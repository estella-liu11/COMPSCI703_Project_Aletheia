"""
Aletheia knowledge-base ingestion + refresh pipeline.

Two modes:

    python ingest.py                # incremental refresh (default)
    python ingest.py --force        # ignore manifest, re-ingest everything
    python ingest.py --scheduled    # cron-friendly: silent on no-op, exit 0
    python ingest.py --dry-run      # report diff, don't touch the vector store

Incremental refresh works via a `DATA-library/_manifest.json` file
recording the SHA-256 of each ingested PDF. On every run:

  1. Scan DATA-library/ for *.pdf
  2. Compute current hashes
  3. Diff against the saved manifest:
       new      → PDF appeared (not in manifest)
       changed  → hash differs from manifest
       removed  → was in manifest, gone from disk
       same     → hash matches → skip
  4. If anything changed, re-ingest the whole library and rewrite the
     manifest. (Whole-library rebuild keeps the implementation tight
     and works identically on both Pinecone and Chroma without needing
     vendor-specific metadata-delete code paths.)
  5. If nothing changed, exit cleanly — safe to wire into cron.

Each emitted chunk carries `source_file` + `source_hash` metadata so a
future enhancement can support per-file incremental deletion without a
manifest rewrite.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from vector_store_factory import (
    build_vector_store_from_documents,
    describe_active_backend,
)
import config

load_dotenv()

DATA_DIR = Path("DATA-library")
MANIFEST_PATH = DATA_DIR / "_manifest.json"


# ────────────────────────────────────────────────────────────────────
# Manifest I/O
# ────────────────────────────────────────────────────────────────────

def _file_sha256(path):
    """Stream-hash a file in 64 KB chunks (handles large PDFs)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _load_manifest():
    """Return the previous manifest dict, or {} if none exists / malformed."""
    if not MANIFEST_PATH.exists():
        return {}
    try:
        payload = json.loads(MANIFEST_PATH.read_text())
        return payload.get("files", {}) if isinstance(payload, dict) else {}
    except (json.JSONDecodeError, OSError):
        # Corrupt manifest → treat as empty so we rebuild from scratch.
        return {}


def _save_manifest(files, backend_label):
    """Write the new manifest with a timestamp and backend snapshot."""
    payload = {
        "schema": 1,
        "last_refreshed": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "backend": backend_label,
        "files": files,
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2))


# ────────────────────────────────────────────────────────────────────
# Diff
# ────────────────────────────────────────────────────────────────────

def _scan_pdfs():
    """Return {filename: sha256} for every PDF currently in DATA-library."""
    return {
        pdf.name: _file_sha256(pdf)
        for pdf in sorted(DATA_DIR.glob("*.pdf"))
    }


def _diff(current, previous):
    """Categorise files into new / changed / removed / unchanged."""
    new      = sorted(set(current) - set(previous))
    removed  = sorted(set(previous) - set(current))
    changed  = sorted(n for n in (set(current) & set(previous))
                      if current[n] != previous[n])
    unchanged = sorted(n for n in (set(current) & set(previous))
                       if current[n] == previous[n])
    return new, changed, removed, unchanged


def _print_diff(new, changed, removed, unchanged):
    """Pretty-print the diff summary."""
    print(f"   New      ({len(new)}):       {', '.join(new) or '—'}")
    print(f"   Changed  ({len(changed)}):   {', '.join(changed) or '—'}")
    print(f"   Removed  ({len(removed)}):   {', '.join(removed) or '—'}")
    print(f"   Unchanged ({len(unchanged)}): {len(unchanged)} file(s)")


# ────────────────────────────────────────────────────────────────────
# Core ingest
# ────────────────────────────────────────────────────────────────────

def _load_and_clean(file_paths):
    """Load a list of PDFs, clean noisy pages, return langchain Documents.

    Per-page filter rules (same heuristics as the original):
      - drop pages < 200 chars after strip
      - drop pages where '.' ratio > 0.2 (OCR noise, TOC dots, etc.)
      - drop pages where alphanumeric ratio < 0.3
    """
    all_pages = []
    for path in file_paths:
        try:
            pages = PyPDFLoader(str(path)).load()
        except Exception as e:
            print(f"   ⚠️  Failed to load {path.name}: {e}")
            continue
        all_pages.extend(pages)
        print(f"   ✓ Loaded {len(pages):4d} pages from {path.name}")

    cleaned = []
    for page in all_pages:
        content = page.page_content.strip()
        if len(content) < config.INGEST_MIN_PAGE_CHARS:
            continue
        if content.count(".") / max(len(content), 1) > config.INGEST_MAX_DOT_RATIO:
            continue
        if sum(c.isalnum() for c in content) / max(len(content), 1) < config.INGEST_MIN_ALPHA_RATIO:
            continue
        cleaned.append(page)
    return all_pages, cleaned


def _attach_metadata(docs, file_to_hash):
    """Tag each chunk with source_file + source_hash for future use."""
    for d in docs:
        src_path = d.metadata.get("source", "")
        name = Path(src_path).name if src_path else ""
        d.metadata["source_file"] = name
        d.metadata["source_hash"] = file_to_hash.get(name, "")
    return docs


def _ingest_full(current_hashes, embeddings, dry_run):
    """Rebuild the vector store from every PDF in DATA-library."""
    if not current_hashes:
        print("⚠️  No PDFs to ingest — DATA-library/ is empty.")
        return False

    file_paths = [DATA_DIR / name for name in current_hashes]
    print(f"\n📥 Loading {len(file_paths)} PDFs...")
    raw_pages, cleaned = _load_and_clean(file_paths)
    print(f"\n🧹 Cleaned: {len(cleaned)}/{len(raw_pages)} pages kept")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.INGEST_CHUNK_SIZE,
        chunk_overlap=config.INGEST_CHUNK_OVERLAP,
    )
    docs = splitter.split_documents(cleaned)
    docs = _attach_metadata(docs, current_hashes)
    print(f"✅ Split into {len(docs)} chunks "
          f"(chunk_size={config.INGEST_CHUNK_SIZE}, overlap={config.INGEST_CHUNK_OVERLAP})")

    if dry_run:
        print(f"\n🟡 --dry-run: would upload {len(docs)} chunks; skipping.")
        return False

    print(f"\n📤 Uploading {len(docs)} chunks to vector store...")
    try:
        build_vector_store_from_documents(documents=docs, embeddings=embeddings)
        print(f"✅ Successfully uploaded {len(docs)} chunks!")
        return True
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False


# ────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────

def start_ingestion(force=False, scheduled=False, dry_run=False):
    """Refresh the knowledge base. Returns exit code (0 = success / no-op)."""
    backend_label = describe_active_backend()

    if not scheduled:
        print("─" * 70)
        print("🛡️  Aletheia Knowledge Base — refresh")
        print(f"    Target:    {backend_label}")
        print(f"    Library:   {DATA_DIR.absolute()}")
        print(f"    Manifest:  {MANIFEST_PATH.absolute()}")
        print("─" * 70)

    current = _scan_pdfs()
    previous = {} if force else _load_manifest()
    new, changed, removed, unchanged = _diff(current, previous)

    if not scheduled:
        print("\n📊 Diff against previous manifest:")
        _print_diff(new, changed, removed, unchanged)

    work_needed = bool(new or changed or removed) or force
    if not work_needed:
        if not scheduled:
            print("\n✅ Nothing changed — vector store is up to date.")
        return 0

    if scheduled:
        # Cron mode: one concise line summarising work
        print(
            f"[{datetime.utcnow().isoformat(timespec='seconds')}Z] "
            f"aletheia-ingest: refreshing — "
            f"new={len(new)} changed={len(changed)} removed={len(removed)}"
        )

    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    ok = _ingest_full(current, embeddings, dry_run=dry_run)

    if ok and not dry_run:
        _save_manifest(current, backend_label)
        if not scheduled:
            print(f"\n📝 Manifest updated → {MANIFEST_PATH}")
        return 0
    elif dry_run:
        return 0
    return 1


def main():
    parser = argparse.ArgumentParser(description="Aletheia KB ingest / refresh")
    parser.add_argument(
        "--force", action="store_true",
        help="Ignore the manifest and re-ingest every PDF in DATA-library.",
    )
    parser.add_argument(
        "--scheduled", action="store_true",
        help="Cron-friendly: minimal output, exit 0 even on no-op.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show the diff and chunk count, but do not write to the store.",
    )
    args = parser.parse_args()

    sys.exit(start_ingestion(
        force=args.force,
        scheduled=args.scheduled,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
