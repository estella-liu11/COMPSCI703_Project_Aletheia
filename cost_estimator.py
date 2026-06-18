"""
Token counting + USD cost estimation for Aletheia.

Two responsibilities:

1. **Count tokens** on every LLM call so the assignment's Commercial
   Stress Test ("how much does a single execution cost in tokens?")
   can be answered with real numbers.

2. **Convert to USD** using a published per-1M-token price table so the
   report can produce a cost-vs-human-labour table.

Counting is **provider-agnostic**: we use a single BPE encoder
(``cl100k_base``) regardless of which backend produced the text.
GPT-4, DeepSeek-V3 and Llama 3.x all use BPE tokenizers whose token
boundaries differ from cl100k_base by single-digit-percent on English
text — close enough for cost estimation, and a *consistent*
yardstick across backends (which is what matters for ablation studies).

If the exact cost matters more than consistency (e.g. for production
billing), swap to the provider-reported counts: DeepSeek returns
``response.usage`` and Ollama returns ``prompt_eval_count`` /
``eval_count``.

State model:
    A single process-wide accumulator (guarded by a Lock) collects
    counts from every invoke_text() call. Callers that want per-audit
    numbers wrap their flow with ``reset_usage()`` / ``get_usage()``.
"""

import threading


# ──────────────────────────────────────────────────────────────────────
# Token counter
# ──────────────────────────────────────────────────────────────────────

try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text):
        """Exact token count via tiktoken's cl100k_base BPE."""
        if not text:
            return 0
        return len(_ENCODER.encode(text, disallowed_special=()))

    _COUNT_METHOD = "tiktoken/cl100k_base"

except ImportError:  # pragma: no cover — tiktoken is in requirements
    # Fallback: 4 chars/token is a reasonable English approximation
    # used by many cost-estimation rulers.
    def count_tokens(text):
        return len(text) // 4 if text else 0

    _COUNT_METHOD = "char/4 approximation (tiktoken unavailable)"


# ──────────────────────────────────────────────────────────────────────
# Pricing
# ──────────────────────────────────────────────────────────────────────
#
# Prices in USD per 1 000 000 tokens. Update when providers change rates.
# Source links:
#   DeepSeek  — https://api-docs.deepseek.com/quick_start/pricing
#   Ollama    — local compute, marginal cost ≈ electricity, treated as 0
#
# A backend with no entry here gets reported as "$0 / unknown" rather
# than raising — so an experimental backend doesn't break evaluation.

PRICE_PER_M_TOKENS = {
    # backend, model
    ("ollama",   None):                {"input": 0.00, "output": 0.00,
                                         "note":  "local compute (electricity-only marginal cost)"},
    ("deepseek", "deepseek-chat"):     {"input": 0.27, "output": 1.10},
    ("deepseek", "deepseek-reasoner"): {"input": 0.55, "output": 2.19},
}


def estimate_cost(input_tokens, output_tokens, backend, model=None):
    """Return USD cost for the given token counts and backend/model.

    Falls back to backend-default (model=None) entry if the specific
    model isn't priced. Returns 0.0 if neither match exists.
    """
    rates = PRICE_PER_M_TOKENS.get((backend, model))
    if rates is None:
        rates = PRICE_PER_M_TOKENS.get((backend, None))
    if rates is None:
        return 0.0
    return (
        input_tokens * rates["input"] + output_tokens * rates["output"]
    ) / 1_000_000


# ──────────────────────────────────────────────────────────────────────
# Process-wide accumulator
# ──────────────────────────────────────────────────────────────────────

_lock = threading.Lock()
_usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}


def reset_usage():
    """Zero the accumulator. Call before each audit you want to measure."""
    with _lock:
        _usage["input_tokens"]  = 0
        _usage["output_tokens"] = 0
        _usage["calls"]         = 0


def get_usage():
    """Return a snapshot of the accumulator."""
    with _lock:
        return dict(_usage)


def record_call(input_tokens, output_tokens):
    """Add one LLM call's token counts to the accumulator.

    Thread-safe — the audit pipeline parallelises Junior calls via
    ThreadPoolExecutor, so multiple threads can land here concurrently.
    """
    with _lock:
        _usage["input_tokens"]  += input_tokens
        _usage["output_tokens"] += output_tokens
        _usage["calls"]         += 1


def describe_counter():
    """For appendices / debug: which tokenizer is in use right now."""
    return _COUNT_METHOD
