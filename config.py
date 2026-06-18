"""
Centralised hyperparameters.

Single source of truth for every tunable knob in the pipeline. Two reasons:

1. **Reproducibility.** The assignment's appendix-quality criterion asks
   for "enough detail (hyperparameters, …) for a peer to replicate
   your project". With everything in one file the appendix is one
   table copied verbatim, not a treasure hunt through six modules.

2. **Ablation.** Sweeping ``INITIAL_RETRIEVAL_K`` from 3 → 5 → 10 is a
   single-line change here rather than chasing literals across files.

These values were chosen empirically during development; defaults
preserve the original behaviour. Override at runtime by editing this
file (no env var indirection — that's reserved for switching backends
and toggling features, where runtime change matters).

Import policy
-------------
Every module that uses a value here imports it from ``config``. We
deliberately do NOT re-export anything; if you don't see it imported
in a module, it's a hardcoded literal that should be moved here next.
"""

# ════════════════════════════════════════════════════════════
# 1. Embeddings
# ════════════════════════════════════════════════════════════
# HuggingFace sentence-transformers model used to embed both
# ingest-time documents and runtime queries. all-MiniLM-L6-v2 is the
# sweet spot for the dataset size: 384-dim, ~80 MB, fast on CPU.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


# ════════════════════════════════════════════════════════════
# 2. Ingestion (PDF → vector store)
# ════════════════════════════════════════════════════════════
# Chunk size 800 chars ≈ 200 tokens — small enough that the retrieved
# chunk fits inside the 4 K context window Llama 3.2 3B has while
# still carrying a full regulatory clause + surrounding context.
INGEST_CHUNK_SIZE = 800
INGEST_CHUNK_OVERLAP = 80

# Per-page filter heuristics (drop OCR noise from the ingest pass).
INGEST_MIN_PAGE_CHARS   = 200    # discard near-blank pages
INGEST_MAX_DOT_RATIO    = 0.2    # discard pages dominated by '.'
INGEST_MIN_ALPHA_RATIO  = 0.3    # discard pages with mostly non-letters


# ════════════════════════════════════════════════════════════
# 3. Retrieval
# ════════════════════════════════════════════════════════════
# Initial MMR retrieval feeding the Junior auditors. fetch_k controls
# how many candidates MMR considers before re-ranking down to k.
INITIAL_RETRIEVAL_K          = 5
INITIAL_RETRIEVAL_FETCH_K    = 15
INITIAL_CHUNK_TRUNCATE_CHARS = 400  # per-chunk char cap for prompt budget

# Citation re-retrieval (CoVe + simple verifier).
VERIFY_K            = 2
VERIFY_MAX_WORKERS  = 5   # parallel Pinecone queries

# Quick Q&A retrieval.
QA_RETRIEVAL_K       = 4
QA_RETRIEVAL_FETCH_K = 20

# Baseline pipeline retrieval (smaller k — preserves the ablation
# contract between baseline and the new dual-junior + CoVe pipeline).
BASELINE_RETRIEVAL_K       = 3
BASELINE_RETRIEVAL_FETCH_K = 20


# ════════════════════════════════════════════════════════════
# 4. LLM call parameters
# ════════════════════════════════════════════════════════════
# Temperature 0 is non-negotiable for audit-style tasks — we need
# reproducible findings, not creative writing.
LLM_TEMPERATURE = 0


# ════════════════════════════════════════════════════════════
# 5. External tools
# ════════════════════════════════════════════════════════════
# Short timeout: a slow Frankfurter response should not stall an
# entire audit. 3 s is long enough for normal latency, short enough
# to be a no-op when the API is unreachable.
EXTERNAL_TIMEOUT_S = 3

# Cap the FX-evidence block to the N largest non-base amounts so a
# noisy contract doesn't bury the Junior in conversions.
EXTERNAL_MAX_AMOUNTS_REPORTED = 3


# ════════════════════════════════════════════════════════════
# 6. Prompt / output limits
# ════════════════════════════════════════════════════════════
# These are also enforced *inside* the prompts ("Maximum N findings");
# duplicating them here lets the appendix list them as configurable.
JUNIOR_MAX_FINDINGS = 3
CHIEF_MAX_FINDINGS  = 5


# ════════════════════════════════════════════════════════════
# 7. Evaluation thresholds
# ════════════════════════════════════════════════════════════
# Default keyword-match threshold used by both coverage and the
# adversarial-pass gate. Individual TCs can override via the
# per-issue ``match_threshold`` field (see test_data.py TC-009).
KEYWORD_MATCH_THRESHOLD = 0.5
