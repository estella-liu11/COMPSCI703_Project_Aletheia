"""
LLM backend factory.

Aletheia supports two interchangeable LLM backends:

- ``ollama``    — local Llama 3.2 via Ollama (default; works offline)
- ``deepseek``  — DeepSeek cloud API (OpenAI-compatible; needs an API key)

Which backend is used is decided by the ``LLM_BACKEND`` environment variable.
The Streamlit UI sets this env var live based on the sidebar selector, and the
``run_*`` functions in ``agent_audit.py`` call :func:`get_llm` each time, so a
switch takes effect on the next audit / Q&A without restarting the app.

Two helpers are exported:

* :func:`get_llm` — returns a cached LLM client for the requested backend.
* :func:`invoke_text` — normalises ``llm.invoke(...)`` to a plain ``str``,
  because ``OllamaLLM`` returns ``str`` whereas ``ChatOpenAI`` returns an
  ``AIMessage``. Centralising this means the rest of the codebase does not
  need to know which backend is active.
"""

import os
from functools import lru_cache
from typing import Any, Optional

from dotenv import load_dotenv

# Load .env on module import so callers don't have to remember.
# Safe to call from any entry point (Streamlit, CLI test scripts,
# evaluation suite) — python-dotenv silently no-ops if .env is missing.
load_dotenv()

# Public backend names
OLLAMA = "ollama"
DEEPSEEK = "deepseek"

VALID_BACKENDS = (OLLAMA, DEEPSEEK)


def get_backend_name() -> str:
    """Resolve the active backend from the ``LLM_BACKEND`` env var.

    Defaults to :data:`OLLAMA`. Raises ``ValueError`` if the env var is set
    to an unknown name so misconfiguration fails loudly.
    """
    name = os.getenv("LLM_BACKEND", OLLAMA).strip().lower()
    if name not in VALID_BACKENDS:
        raise ValueError(
            f"Invalid LLM_BACKEND={name!r}. Must be one of {VALID_BACKENDS}."
        )
    return name


@lru_cache(maxsize=4)
def _build_llm(backend: str) -> Any:
    """Build (and cache) an LLM client for the given backend.

    Cached per backend name, so switching back and forth in the UI does not
    re-instantiate the client every call.
    """
    # Lazy import to avoid forcing every consumer of llm_factory to
    # carry a config dependency.
    import config

    if backend == OLLAMA:
        from langchain_ollama import OllamaLLM
        return OllamaLLM(
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=config.LLM_TEMPERATURE,
        )

    if backend == DEEPSEEK:
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. Add it to .env or switch to "
                "the Ollama backend in the sidebar."
            )
        return ChatOpenAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            temperature=config.LLM_TEMPERATURE,
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        )

    raise ValueError(f"Unknown backend: {backend!r}")


def get_llm(backend: Optional[str] = None) -> Any:
    """Return a cached LLM client.

    Args:
        backend: ``"ollama"``, ``"deepseek"``, or ``None`` to read the
            ``LLM_BACKEND`` env var.
    """
    backend = (backend or get_backend_name()).strip().lower()
    return _build_llm(backend)


def invoke_text(llm: Any, prompt: str) -> str:
    """Run ``llm.invoke(prompt)`` and normalise the result to a string.

    ``OllamaLLM.invoke`` returns ``str``; ``ChatOpenAI.invoke`` returns an
    ``AIMessage`` with a ``.content`` attribute. The rest of the codebase
    treats LLM output as plain text, so this wrapper hides the difference.

    This is also the central chokepoint for **token accounting**
    (see ``cost_estimator.py``). Every LLM call in Aletheia routes
    through ``invoke_text``, so instrumenting here captures Juniors,
    Chief Officer, CoVe question generation, and Quick Q&A in one
    place. Cost reporting happens off the resulting accumulator.
    """
    # Local import — cost_estimator pulls in tiktoken which is heavy.
    # Importing inside the function keeps `import llm_factory` cheap.
    from cost_estimator import count_tokens, record_call

    input_tokens = count_tokens(prompt)
    result = llm.invoke(prompt)
    text = result if isinstance(result, str) else getattr(result, "content", str(result))
    output_tokens = count_tokens(text)
    record_call(input_tokens, output_tokens)
    return text


def describe_active_backend() -> str:
    """Human-readable label for the currently selected backend.

    Used by the sidebar status line in ``app.py``.
    """
    backend = get_backend_name()
    if backend == OLLAMA:
        model = os.getenv("OLLAMA_MODEL", "llama3.2")
        return f"🧠 Ollama · {model}"
    if backend == DEEPSEEK:
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        return f"🧠 DeepSeek · {model}"
    return f"🧠 {backend}"
