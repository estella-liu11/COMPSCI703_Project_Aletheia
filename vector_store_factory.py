"""
Vector-database backend factory.

Aletheia supports two interchangeable vector stores:

- ``pinecone`` — Pinecone Serverless (cloud). Default. Fast, fully managed,
  but contract embeddings and queries leave the host. Not acceptable for
  data-sovereignty-sensitive deployments (Group 14's peer review).
- ``chroma``   — Chroma DB persisted to a local directory. Fully air-gapped:
  combined with the ``ollama`` LLM backend, the entire audit can run with
  no network egress at all.

Backend selection is driven by the ``VECTOR_DB_BACKEND`` env var (default:
``pinecone``). Unlike the LLM backend, this one is NOT switchable at
runtime: once you ingest to a particular store, that's where your
embeddings live. Switching the env var only changes which store the
audit/ingest pipelines read from — the other store will appear empty.

Exports:
    * ``get_backend_name()``
    * ``get_vector_store(embeddings)`` — for read-side use in agents.
    * ``build_vector_store_from_documents(docs, embeddings)`` — for ingest.
    * ``describe_active_backend()`` — for the sidebar status line.
"""

import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv

load_dotenv()

PINECONE = "pinecone"
CHROMA = "chroma"

VALID_BACKENDS = (PINECONE, CHROMA)

DEFAULT_INDEX_NAME = "aletheia-legal-db"
DEFAULT_CHROMA_DIR = "./chroma_db"


def get_backend_name() -> str:
    """Resolve the active vector backend from ``VECTOR_DB_BACKEND``."""
    name = os.getenv("VECTOR_DB_BACKEND", PINECONE).strip().lower()
    if name not in VALID_BACKENDS:
        raise ValueError(
            f"Invalid VECTOR_DB_BACKEND={name!r}. "
            f"Must be one of {VALID_BACKENDS}."
        )
    return name


@lru_cache(maxsize=2)
def _build_pinecone(index_name: str, api_key: str, embeddings_id: str) -> Any:
    """Cached PineconeVectorStore. embeddings_id keeps cache safe."""
    from langchain_pinecone import PineconeVectorStore
    # We can't cache the embeddings object directly in the cache key
    # (HuggingFaceEmbeddings is not hashable), so callers must pass a
    # stable string id and re-pass the actual embeddings object below.
    # We rebuild the store with the supplied embeddings each call —
    # cheap because the underlying Pinecone client is itself idempotent.
    return PineconeVectorStore(
        index_name=index_name,
        embedding=_active_embeddings,
        pinecone_api_key=api_key,
    )


# Module-level slot the cached builder reads from. Set in get_vector_store
# right before the cache call so the lru_cache key stays simple.
_active_embeddings: Any = None


def get_vector_store(embeddings) -> Any:
    """Return a vector store configured against the active backend.

    Args:
        embeddings: A langchain Embeddings instance (e.g. HuggingFaceEmbeddings).

    Returns:
        Either a ``PineconeVectorStore`` or a ``Chroma`` instance.
        Both expose ``.similarity_search(query, k)`` and
        ``.as_retriever(...)`` so callers don't have to branch.
    """
    global _active_embeddings
    _active_embeddings = embeddings
    backend = get_backend_name()

    if backend == PINECONE:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "PINECONE_API_KEY is not set. Add it to .env or switch "
                "to VECTOR_DB_BACKEND=chroma for a local store."
            )
        index_name = os.getenv("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME)
        # Force-rebuild rather than relying on the cache key (since
        # embeddings is unhashable — see _build_pinecone docstring).
        from langchain_pinecone import PineconeVectorStore
        return PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
            pinecone_api_key=api_key,
        )

    if backend == CHROMA:
        from langchain_chroma import Chroma
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", DEFAULT_CHROMA_DIR)
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=os.getenv("CHROMA_COLLECTION", "aletheia"),
        )

    raise ValueError(f"Unknown vector backend: {backend!r}")


def build_vector_store_from_documents(documents, embeddings) -> Any:
    """Ingest path: build (or extend) the vector store from a doc list.

    Pinecone: uses ``PineconeVectorStore.from_documents`` which upserts
    into the named index (auto-creates if missing on serverless).
    Chroma: uses ``Chroma.from_documents`` with persist_directory; the
    on-disk store is created if missing and appended to otherwise.
    """
    backend = get_backend_name()

    if backend == PINECONE:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY is not set.")
        index_name = os.getenv("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME)
        from langchain_pinecone import PineconeVectorStore
        return PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=index_name,
            pinecone_api_key=api_key,
        )

    if backend == CHROMA:
        from langchain_chroma import Chroma
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", DEFAULT_CHROMA_DIR)
        return Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_name=os.getenv("CHROMA_COLLECTION", "aletheia"),
        )

    raise ValueError(f"Unknown vector backend: {backend!r}")


def describe_active_backend() -> str:
    """Human-readable label for the sidebar."""
    backend = get_backend_name()
    if backend == PINECONE:
        index_name = os.getenv("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME)
        return f"📚 Pinecone · {index_name}"
    if backend == CHROMA:
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", DEFAULT_CHROMA_DIR)
        return f"📚 Chroma (local) · {persist_dir}"
    return f"📚 {backend}"
