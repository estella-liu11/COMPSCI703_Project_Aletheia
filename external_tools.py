"""
External-API tooling.

Aletheia treats the regulatory text in Pinecone/Chroma as the "internal"
knowledge base — it's static and curated. But cross-border tax audits
also depend on **live** facts that don't sit in a vector DB:

    • Today's exchange rate (a contract that pays "USD 50 000 quarterly"
      means something different at 1 USD = 1.40 NZD vs 1 USD = 1.62 NZD)
    • Counterparty registration status (is "Wellington Analytics Ltd"
      a real NZ-registered company?)
    • Latest amendments to specific regulations

This module is the first such external-tool entry point. The
``build_external_evidence()`` function is called by the LangGraph
pre-HITL graph's ``enrich`` node and adds a short evidence block to the
context Junior auditors see.

Initial provider:
    Frankfurter (https://www.frankfurter.app/) — ECB-sourced FX rates,
    no API key, no registration, MIT-licensed.

Failure model is **conservative**: every external call has a 3 s
timeout and any failure (DNS, 5xx, timeout, JSON parse error) is logged
and silently returns no evidence. An audit running in an offline
environment will still complete; it just won't get the FX enrichment.

Toggle: ``EXTERNAL_TOOLS_ENABLED`` env var (default: true). Set to
"false" / "0" / "no" to disable external lookups entirely (useful for
fully air-gapped deployments combined with Ollama + Chroma).
"""

import os
import re
from functools import lru_cache

import requests

import config

# ────────────────────────────────────────────────────────────────────
# Configuration (constants here are module-internal; tunables live in
# config.py so the appendix has a single hyperparameter table).
# ────────────────────────────────────────────────────────────────────

FRANKFURTER_URL = "https://api.frankfurter.app/latest"
HTTP_TIMEOUT_S = config.EXTERNAL_TIMEOUT_S

# Currencies we know how to talk about in the evidence block. Anything
# else found in the contract is just ignored (no risky LLM-driven lookup).
SUPPORTED_CURRENCIES = {"USD", "NZD", "EUR", "GBP", "AUD", "JPY", "CNY"}

# Cap evidence so a contract with 30 currency mentions doesn't bury the
# Junior in noise. The top N largest amounts (by value) are picked.
MAX_AMOUNTS_REPORTED = config.EXTERNAL_MAX_AMOUNTS_REPORTED


def external_tools_enabled():
    """Whether to perform live external lookups at all.

    Mirrors the ``COVE_ENABLED`` pattern in ``agent_audit.py``: set to
    "false" / "0" / "no" to disable, anything else (including unset)
    keeps the feature on.
    """
    return os.getenv("EXTERNAL_TOOLS_ENABLED", "true").strip().lower() not in (
        "false", "0", "no",
    )


# ────────────────────────────────────────────────────────────────────
# Currency amount detection
# ────────────────────────────────────────────────────────────────────

# Matches "USD 50,000", "USD 50000.00", "USD$50,000.50", etc.
# Group 1 = currency code, group 2 = amount with separators stripped.
_AMOUNT_RE = re.compile(
    r'\b(USD|NZD|EUR|GBP|AUD|JPY|CNY)\s*\$?\s*([0-9](?:[0-9,]*[0-9])?(?:\.[0-9]+)?)\b',
    re.IGNORECASE,
)


def detect_currency_amounts(text):
    """Find currency amounts written like 'USD 50,000' or 'NZD 1,234.56'.

    Returns a list of (currency_code, float_amount) tuples in the order
    they appear. Duplicates are kept — the caller decides how to summarise.
    """
    out = []
    for match in _AMOUNT_RE.finditer(text):
        ccy = match.group(1).upper()
        raw_amt = match.group(2).replace(",", "")
        try:
            amt = float(raw_amt)
        except ValueError:
            continue
        if ccy in SUPPORTED_CURRENCIES and amt > 0:
            out.append((ccy, amt))
    return out


# ────────────────────────────────────────────────────────────────────
# Frankfurter FX lookup
# ────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=64)
def lookup_exchange_rate(from_ccy, to_ccy):
    """Return today's rate as a dict ``{rate, date, source}``, or None.

    Cached by (from, to) for the lifetime of the Python process, so a
    contract that mentions USD five times only hits the API once.
    """
    if from_ccy == to_ccy:
        return {"rate": 1.0, "date": "n/a", "source": "identity"}

    try:
        resp = requests.get(
            FRANKFURTER_URL,
            params={"from": from_ccy, "to": to_ccy},
            timeout=HTTP_TIMEOUT_S,
        )
        resp.raise_for_status()
        data = resp.json()
        rate = data.get("rates", {}).get(to_ccy)
        if rate is None:
            return None
        return {
            "rate": float(rate),
            "date": data.get("date", "n/a"),
            "source": "Frankfurter / ECB",
        }
    except (requests.RequestException, ValueError, TypeError) as e:
        # Log but don't raise — caller falls back to no evidence.
        print(f"   ⚠️ FX lookup {from_ccy}→{to_ccy} failed: {e}")
        return None


# ────────────────────────────────────────────────────────────────────
# Orchestrator — called by the LangGraph `enrich` node
# ────────────────────────────────────────────────────────────────────

def build_external_evidence(contract_text, base_currency="NZD"):
    """Top-level entry point: scan contract → query APIs → format evidence.

    Returns a short markdown-ish block (under ~500 chars) for inclusion
    in the retrieval context. Returns an empty string if external tools
    are disabled, the contract has no detectable amounts, or all API
    calls failed — caller can drop it into the context unconditionally.

    Args:
        contract_text: The raw contract string the user uploaded.
        base_currency: Convert detected non-base amounts INTO this code.
            Defaults to NZD because Aletheia's primary user base is
            NZ-resident parties auditing cross-border deals.
    """
    if not external_tools_enabled():
        return ""

    amounts = detect_currency_amounts(contract_text)
    if not amounts:
        return ""

    # Deduplicate per currency code: report the largest amount per code
    # (small contractual line-items don't add evaluation signal).
    largest_by_ccy = {}
    for ccy, amt in amounts:
        if ccy == base_currency:
            continue  # nothing to convert
        if ccy not in largest_by_ccy or amt > largest_by_ccy[ccy]:
            largest_by_ccy[ccy] = amt

    if not largest_by_ccy:
        return ""

    # Pick top N largest amounts overall so a busy contract doesn't
    # flood the evidence section.
    ranked = sorted(largest_by_ccy.items(), key=lambda kv: -kv[1])[:MAX_AMOUNTS_REPORTED]

    lines = ["─── External Evidence (live API lookups) ───"]
    any_success = False
    for ccy, amt in ranked:
        rate_info = lookup_exchange_rate(ccy, base_currency)
        if rate_info is None:
            continue
        converted = amt * rate_info["rate"]
        lines.append(
            f"💱 Contract amount {ccy} {amt:,.2f} ≈ {base_currency} "
            f"{converted:,.2f} at 1 {ccy} = {rate_info['rate']:.4f} {base_currency} "
            f"(as of {rate_info['date']}; source: {rate_info['source']})"
        )
        any_success = True

    if not any_success:
        return ""

    lines.append(
        "Note: use these live rates when reasoning about currency clauses, "
        "rate-lock provisions, and economic significance."
    )
    lines.append("───────────────────────────────────────────────")
    return "\n".join(lines)
