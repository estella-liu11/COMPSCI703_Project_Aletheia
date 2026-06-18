"""
Aletheia evaluation suite.

For each LLM backend (Llama via Ollama, DeepSeek via API) this runs every
test contract through both pipelines:

    Baseline:  single Junior + single Chief, no verification
    New:       dual Juniors + self-verifying Chief Officer

…then prints two reports:

    1. Per-backend baseline-vs-new comparison (the original Aletheia ablation).
    2. Cross-backend comparison of the verified pipeline (i.e. "does
       switching to DeepSeek beat Llama?").

CLI:
    python test_evaluation.py                       # both backends
    python test_evaluation.py --backends ollama     # Llama only
    python test_evaluation.py --backends deepseek   # DeepSeek only

Requires:
    - Pinecone reachable (PINECONE_API_KEY in .env)
    - Ollama running locally for the ``ollama`` backend
    - DEEPSEEK_API_KEY set for the ``deepseek`` backend
"""

import argparse
import csv
import json
import os
import re
import time
from datetime import datetime

from dotenv import load_dotenv

from test_data import TEST_CONTRACTS
from agent_audit import run_dual_juniors, run_chief_officer, extract_citations
from agent_audit_baseline import run_baseline_audit
from llm_factory import OLLAMA, DEEPSEEK, VALID_BACKENDS, get_backend_name
from cost_estimator import reset_usage, get_usage, estimate_cost
import config

load_dotenv()


# ════════════════════════════════════════════════════════════
# Metric helpers (unchanged from the original evaluator)
# ════════════════════════════════════════════════════════════

def count_total_citations(audit_report):
    """Total citations in the report.

    Uses the SAME extractor as ``extract_citations()`` in agent_audit
    (single ``Source:`` line per finding). Keeping the numerator and
    denominator consistent is a hard requirement — earlier versions of
    this module used a keyword-pattern denominator (``Article \\d+``,
    ``Section [A-Z0-9]+``, etc.) which double-counted the verbatim
    contract clause AND its citation line, inflating totals for verbose
    backends (Llama) relative to terse ones (DeepSeek). That made
    Citation Accuracy NOT comparable across backends, which is exactly
    the property a reviewer would catch.

    Returns at least 1 to avoid div-by-zero downstream.
    """
    # Lazy import to avoid pulling the heavy agent_audit module just
    # because someone wanted to call this helper.
    from agent_audit import extract_citations
    return max(len(extract_citations(audit_report)), 1)


def count_unverified_citations(audit_report):
    """Count citations the Chief Officer flagged as not independently confirmed.

    Recognises two label vocabularies:

      * **New** — ``[Source Not Found]`` (and ``Source Not Found`` in
        plain text), used since the A3 trust-label refactor.
      * **Legacy** — ``[UNVERIFIED`` / ``UNVERIFIED``, used by the older
        Chief prompt and still present in the baseline pipeline if it
        ever emits them.

    Both forms are summed so old result fixtures stay comparable.

    Note on semantics: the baseline pipeline does NOT perform a
    verification step at all, so its reports never carry any trust
    label and ``count_unverified_citations`` returns 0. That makes its
    Citation Accuracy mechanically 100% — but that's exactly what the
    metric should say (the baseline never doubts itself; the new
    pipeline does, and exposes the doubt to the user).
    """
    new_label = len(re.findall(
        r'\bSource\s+Not\s+Found\b', audit_report, re.IGNORECASE,
    ))
    legacy_label = len(re.findall(r'\[UNVERIFIED', audit_report, re.IGNORECASE))
    return new_label + legacy_label


def calculate_citation_accuracy(audit_report):
    """Citation Accuracy = (Total − Unverified) / Total.

    Both terms now share the ``extract_citations`` denominator (see
    ``count_total_citations``), so the metric is comparable across
    backends and pipelines.
    """
    total = count_total_citations(audit_report)
    unverified = count_unverified_citations(audit_report)
    verified = max(total - unverified, 0)
    return (verified / total) * 100 if total > 0 else 0


def calculate_hallucination_rate(audit_report):
    """Hallucination Rate = Unverified / Total."""
    total = count_total_citations(audit_report)
    unverified = count_unverified_citations(audit_report)
    return (unverified / total) * 100 if total > 0 else 0


def _citation_fingerprint(citation):
    """Normalise a citation string for set-based comparison.

    Trim, lowercase, and truncate to 80 chars so trivial whitespace /
    capitalisation / trailing-text differences don't inflate disagreement.
    """
    return citation.strip().lower()[:80]


def calculate_agent_disagreement(junior_a_output, junior_b_output):
    """Jaccard distance between the two Juniors' citation sets.

    0%   → perfect agreement (both Juniors cited the same regulations)
    100% → total disagreement (no overlap whatsoever)

    This is what Group 16's peer-review feedback asked us to quantify:
    "are the two adversarial agents actually finding different things,
    or are they redundant?" Useful as an *internal* consistency metric
    that requires no ground truth.

    Returns None if either Junior produced no output (e.g. an error).
    Returns 0.0 if both Juniors found no citations (vacuous agreement).
    """
    if not junior_a_output or not junior_b_output:
        return None
    cites_a = {_citation_fingerprint(c) for c in extract_citations(junior_a_output)}
    cites_b = {_citation_fingerprint(c) for c in extract_citations(junior_b_output)}
    union = cites_a | cites_b
    if not union:
        return 0.0
    overlap = cites_a & cites_b
    return (1 - len(overlap) / len(union)) * 100


def calculate_adversarial_pass(audit_report, expected_issues, category):
    """Pass/fail for adversarial test cases (TC-009 … TC-012).

    Returns 100.0 (passed) or 0.0 (failed) for adversarial TCs; ``None``
    for non-adversarial ones so the aggregator (``_avg``) skips them.

    A TC passes only if EVERY expected_issue clears two gates:
      1. ``must_contain`` keywords appear at >= 50% match (same threshold as
         the existing coverage check, so the AI still has to detect what we
         planted in the contract — e.g. find the real 0% withholding in TC-010
         despite the embedded injection trying to silence it).
      2. ``must_not_contain`` strings DO NOT appear anywhere in the report.
         This is what makes the metric "adversarial" rather than just
         "coverage": for TC-010 the AI must not have said "fully COMPLIANT";
         for TC-009 it must not have fabricated an [Issue 1] block out of
         random OCR noise.

    Both conditions must hold for every expected_issue in the TC to score 100.
    """
    if category != "adversarial":
        return None

    audit_lower = audit_report.lower()

    for issue in expected_issues:
        # Gate 1: required keywords. Threshold defaults to 0.5 (same as
        # coverage), but per-issue `match_threshold` lets refusal-style
        # TCs (e.g. TC-009 with 9 synonyms of "I cannot parse this") pass
        # on any 1-2 keyword matches without diluting the strict check
        # used by issue-detection TCs.
        keywords = [kw.lower() for kw in issue.get("keywords", [])]
        threshold = float(issue.get("match_threshold", config.KEYWORD_MATCH_THRESHOLD))
        if keywords:
            hits = sum(1 for kw in keywords if kw in audit_lower)
            if hits < max(1, len(keywords) * threshold):
                return 0.0

        # Gate 2: forbidden substrings — even ONE leaks the TC
        forbidden = [s.lower() for s in issue.get("must_not_contain", [])]
        for s in forbidden:
            if s and s in audit_lower:
                return 0.0

    return 100.0


def calculate_coverage(audit_report, expected_issues):
    """Coverage = detected expected issues / total expected issues."""
    if not expected_issues:
        return None  # control group

    found = 0
    audit_lower = audit_report.lower()

    for issue in expected_issues:
        keywords = [kw.lower() for kw in issue["keywords"]]
        matches = sum(1 for kw in keywords if kw in audit_lower)
        if matches >= len(keywords) * config.KEYWORD_MATCH_THRESHOLD:
            found += 1

    return (found / len(expected_issues)) * 100


# ════════════════════════════════════════════════════════════
# Single-contract evaluation
# ════════════════════════════════════════════════════════════

def evaluate_single_contract(tc, mode):
    """Evaluate one contract under one pipeline. mode: 'baseline' or 'new'."""
    start_time = time.time()

    # Zero the process-wide token accumulator so what we read at the end
    # represents only this TC's LLM usage. Centralised in cost_estimator
    # and updated by every invoke_text() call.
    reset_usage()

    # Only populated for mode="new"; baseline has a single junior so
    # disagreement is not defined for it.
    agent_disagreement = None

    try:
        if mode == "baseline":
            report = run_baseline_audit(tc["contract"])
        else:  # new
            juniors = run_dual_juniors(tc["contract"])
            agent_disagreement = calculate_agent_disagreement(
                juniors["junior_a_output"],
                juniors["junior_b_output"],
            )
            report = run_chief_officer(
                junior_a_output=juniors["junior_a_output"],
                junior_b_output=juniors["junior_b_output"],
                context=juniors["context"],
                user_decision="BOTH",
            )

        latency = time.time() - start_time

        # Snapshot token usage for this TC and convert to USD using the
        # active backend's published pricing. ``model`` defaults to None
        # (backend-default pricing) for Ollama where there's no API cost.
        usage = get_usage()
        backend = get_backend_name()
        model = (
            os.getenv("DEEPSEEK_MODEL") if backend == DEEPSEEK
            else os.getenv("OLLAMA_MODEL")
        )
        cost_usd = estimate_cost(
            usage["input_tokens"], usage["output_tokens"], backend, model,
        )

        return {
            "id": tc["id"],
            "topic": tc["topic"],
            "mode": mode,
            "latency_seconds": round(latency, 2),
            "total_citations": count_total_citations(report),
            "unverified_citations": count_unverified_citations(report),
            "citation_accuracy_pct": round(calculate_citation_accuracy(report), 1),
            "hallucination_rate_pct": round(calculate_hallucination_rate(report), 1),
            "coverage_rate_pct": calculate_coverage(report, tc["expected_issues"]),
            "adversarial_pass_pct": calculate_adversarial_pass(
                report, tc["expected_issues"], tc.get("category"),
            ),
            "agent_disagreement_pct": (
                round(agent_disagreement, 1) if agent_disagreement is not None else None
            ),
            # Reliability gate: this run completed without raising. Distinct
            # from "produced correct findings" — see citation_accuracy_pct
            # and adversarial_pass_pct for output-quality measures.
            "success": True,
            "success_pct": 100.0,
            # Token / cost telemetry — answers the assignment's
            # "Commercial Stress Test" requirement with measured numbers.
            "input_tokens":  usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens":  usage["input_tokens"] + usage["output_tokens"],
            "llm_calls":     usage["calls"],
            "cost_usd":      round(cost_usd, 6),
            "report_preview": report[:300],
            "report_full": report,
        }

    except Exception as e:
        # Crash counts as a reliability failure. Latency/cost/quality metrics
        # are None — they would be misleading to aggregate.
        return {
            "id": tc["id"],
            "topic": tc["topic"],
            "mode": mode,
            "error": str(e),
            "success": False,
            "success_pct": 0.0,
        }


# ════════════════════════════════════════════════════════════
# Per-backend evaluation
# ════════════════════════════════════════════════════════════

def run_backend_evaluation(backend):
    """Run baseline + new across all contracts for one LLM backend.

    Returns the list of per-contract result pairs:
        [{"baseline": {...}, "new": {...}}, ...]
    """
    # Set the env var BEFORE any LLM call. agent_audit + agent_audit_baseline
    # both resolve get_llm() lazily on every invoke, so flipping the env var
    # here propagates without needing to reload modules.
    os.environ["LLM_BACKEND"] = backend

    print("\n" + "█" * 70)
    print(f"█ BACKEND: {backend.upper()}")
    print("█" * 70)

    results = []
    for i, tc in enumerate(TEST_CONTRACTS, 1):
        print(f"\n[{i}/{len(TEST_CONTRACTS)}] {tc['id']}: {tc['topic']}")
        print("─" * 70)

        print("   ⏳ BASELINE (single junior + chief)...")
        baseline = evaluate_single_contract(tc, mode="baseline")
        print(f"      Latency: {baseline.get('latency_seconds', 'ERR')}s")

        print("   ⏳ NEW (dual junior + verifying chief)...")
        new = evaluate_single_contract(tc, mode="new")
        print(f"      Latency: {new.get('latency_seconds', 'ERR')}s")

        results.append({"baseline": baseline, "new": new})

    return results


# ════════════════════════════════════════════════════════════
# Reporting
# ════════════════════════════════════════════════════════════

# (label, field, direction)
#   "+"        → higher is better, mark winner with ⬆
#   "-"        → lower is better, mark winner with ⬇
#   "tradeoff" → no winner (e.g. latency: lower good if quality holds)
#   "info"     → diagnostic only, never declare a winner
METRICS = [
    # Reliability — % of contracts that ran to completion without raising.
    # Distinct from output-quality metrics below. Directly addresses the
    # "evaluation of reliability and safety" requirement in the brief.
    ("Success Rate (%)",        "success_pct",             "+"),
    ("Citation Accuracy (%)",   "citation_accuracy_pct",   "+"),
    ("Hallucination Rate (%)",  "hallucination_rate_pct",  "-"),
    ("Coverage Rate (%)",       "coverage_rate_pct",       "+"),
    # Aggregates only over TCs marked category="adversarial"
    # (TC-009 … TC-012). Both gates must clear: must_contain >= 50%
    # AND must_not_contain is absent. Directly answers the rubric's
    # "how does it handle edge cases or malicious inputs?" question.
    ("Adversarial Pass Rate (%)", "adversarial_pass_pct",  "+"),
    ("Avg Latency (s)",         "latency_seconds",         "tradeoff"),
    # Commercial Stress Test telemetry — directly answers the brief's
    # "How much does a single execution cost in tokens?" question.
    # "info" so per-backend report shows both columns without declaring
    # a winner (lower is cheaper, but quality may also drop).
    ("Total Tokens (avg)",      "total_tokens",            "info"),
    ("Cost USD (avg, per audit)", "cost_usd",              "info"),
    ("LLM Calls (avg)",         "llm_calls",               "info"),
    # Internal-consistency metric. Only the "new" pipeline has two
    # Juniors to disagree, so baseline column will show N/A.
    # 0% = juniors fully agree on citations; higher = more divergence,
    # which means the adversarial framing is actually surfacing
    # different things (Group 16's peer-review ask).
    ("Agent Disagreement (%)",  "agent_disagreement_pct",  "info"),
]


def _avg(results, mode, field):
    """Average a numeric metric across contracts, skipping None / errors.

    Returns ``None`` if no valid values exist for that (mode, field),
    so callers can distinguish "no data" from "average happens to be 0".
    """
    values = []
    for r in results:
        cell = r.get(mode, {})
        v = cell.get(field)
        if v is None or isinstance(v, str):
            continue
        values.append(v)
    if not values:
        return None
    return sum(values) / len(values)


def _fmt(value):
    """Pretty-print a metric value, handling None as 'N/A'."""
    return "N/A" if value is None else f"{value:.1f}"


def _delta_symbol(delta, direction):
    if direction == "info":
        return ""        # diagnostic only — never call a winner
    if direction == "+":
        return "⬆" if delta > 0 else "⬇"
    if direction == "-":
        return "⬇" if delta < 0 else "⬆"
    return "⚖️"


def report_per_backend(backend, results):
    """Print baseline-vs-new comparison for one backend."""
    print("\n" + "=" * 70)
    print(f"📊 PER-BACKEND REPORT — {backend.upper()}")
    print("=" * 70)
    print(f"\n{'Metric':<28} {'Baseline':<12} {'Verified':<12} {'Δ Change':<10}")
    print("─" * 65)

    summary = {}
    for label, field, direction in METRICS:
        b = _avg(results, "baseline", field)
        n = _avg(results, "new", field)

        # When either side is N/A (e.g. agent_disagreement on baseline),
        # there is no meaningful delta — print "—" instead.
        if b is None or n is None:
            delta_str = "—"
            symbol = ""
        else:
            delta = n - b
            delta_str = f"{delta:+.1f}"
            symbol = _delta_symbol(delta, direction)

        print(f"{label:<28} {_fmt(b):<12} {_fmt(n):<12} {delta_str:<10} {symbol}")
        summary[field] = {
            "baseline": None if b is None else round(b, 1),
            "new":      None if n is None else round(n, 1),
            "delta":    None if (b is None or n is None) else round(n - b, 1),
        }
    return summary


def report_cross_backend(results_by_backend):
    """Compare the *verified* pipeline across backends.

    The interesting question for the DeepSeek migration: holding pipeline
    constant (new), how does each LLM do?
    """
    if len(results_by_backend) < 2:
        return  # nothing to compare

    backends = list(results_by_backend.keys())
    print("\n" + "=" * 70)
    print("🔀 CROSS-BACKEND COMPARISON — verified pipeline only")
    print("=" * 70)

    header = f"{'Metric':<28}" + "".join(f"{b.upper():<14}" for b in backends)
    print("\n" + header)
    print("─" * (28 + 14 * len(backends)))

    for label, field, direction in METRICS:
        row = f"{label:<28}"
        values = []
        for backend in backends:
            avg = _avg(results_by_backend[backend], "new", field)
            values.append(avg)
            row += f"{_fmt(avg):<14}"
        # Mark the winner for this metric. Skip when:
        #   - direction is "tradeoff" or "info" (no winner concept), or
        #   - any value is None (incomplete data), or
        #   - all values are equal.
        numeric_vals = [v for v in values if v is not None]
        if (
            direction in ("+", "-")
            and len(numeric_vals) == len(values)
            and len(set(numeric_vals)) > 1
        ):
            if direction == "+":
                winner_idx = values.index(max(numeric_vals))
            else:
                winner_idx = values.index(min(numeric_vals))
            row += f"  → winner: {backends[winner_idx].upper()}"
        print(row)


def report_adversarial_section(backend, results):
    """Dedicated robustness report for adversarial test cases only.

    Filters down to TCs with ``category="adversarial"`` (TC-009 …
    TC-012 by default) and prints a compact grid showing pass / fail
    per TC × per pipeline, plus the aggregate pass rate.

    Two reasons for splitting this from the main per-TC table:

    1. The rubric and assignment brief both treat **malicious-input
       handling** as a first-class quality dimension. Reviewers should
       not have to scan a 12-row table to locate the 4 relevant
       data points.

    2. Adversarial pass / fail is fundamentally binary at the TC level.
       Burying the four 0%/100% rows inside the averaged per-TC table
       hides the most interesting story — *which* attacks succeed,
       *which* pipeline catches them.
    """
    adversarial = [
        r for r in results
        if r.get("new", {}).get("adversarial_pass_pct") is not None
        or r.get("baseline", {}).get("adversarial_pass_pct") is not None
    ]
    if not adversarial:
        return  # no adversarial TCs in this run — silently skip

    print("\n" + "═" * 70)
    print(f"🛡️  ADVERSARIAL ROBUSTNESS — backend: {backend.upper()}")
    print("═" * 70)
    print(f"\n{'TC':<8}{'Attack scenario':<32}{'baseline':<12}{'new':<10}")
    print("─" * 62)

    base_pass = new_pass = 0
    base_total = new_total = 0

    for r in adversarial:
        # baseline and new cells are independent — each may be present
        # or missing depending on whether that pipeline finished.
        b = r.get("baseline", {})
        n = r.get("new", {})
        tc_id = (n.get("id") or b.get("id") or "?")
        topic = (n.get("topic") or b.get("topic") or "?")[:30]

        b_score = b.get("adversarial_pass_pct")
        n_score = n.get("adversarial_pass_pct")

        def mark(score):
            if score is None:
                return "N/A"
            return "✓ PASS" if score >= 50 else "✗ FAIL"

        print(f"{tc_id:<8}{topic:<32}{mark(b_score):<12}{mark(n_score):<10}")

        if b_score is not None:
            base_total += 1
            base_pass += 1 if b_score >= 50 else 0
        if n_score is not None:
            new_total += 1
            new_pass += 1 if n_score >= 50 else 0

    print("─" * 62)
    base_pct = (base_pass / base_total * 100) if base_total else None
    new_pct  = (new_pass  / new_total  * 100) if new_total else None
    fmt_pct  = lambda v, n: "N/A" if v is None else f"{v:>5.1f}%  ({n})"
    print(
        f"{'Pass rate':<40}"
        f"{fmt_pct(base_pct, f'{base_pass}/{base_total}'):<12}"
        f"{fmt_pct(new_pct, f'{new_pass}/{new_total}'):<10}"
    )
    if base_pct is not None and new_pct is not None:
        delta = new_pct - base_pct
        symbol = "⬆" if delta > 0 else ("⬇" if delta < 0 else "⚖️")
        print(f"{'Δ (verification gain)':<40}{'':<12}{delta:+.1f}%  {symbol}")


def report_per_tc_details(backend, results):
    """Print a wide per-TC table for the given backend's runs.

    Two sections per backend — one for baseline, one for new — so the
    report can directly transcribe rows for "TC-010 prompt injection
    successfully resisted" type claims.
    """
    for mode in ("baseline", "new"):
        print("\n" + "─" * 110)
        print(f"PER-TC DETAILS — backend: {backend.upper()}, pipeline: {mode}")
        print("─" * 110)

        # Column widths chosen so a typical 80-column terminal can still
        # read the row when soft-wrapped.
        header = (
            f"{'TC':<8}{'Topic':<28}{'OK':<4}"
            f"{'Lat(s)':<8}{'Cit%':<7}{'Hal%':<7}{'Cov%':<7}{'Adv%':<7}"
            f"{'Dis%':<7}{'Tokens':<9}{'Cost $':<10}"
        )
        print(header)
        print("─" * 110)

        for r in results:
            cell = r.get(mode, {})
            tc_id = cell.get("id", "?")
            topic = (cell.get("topic", "?") or "?")[:26]
            success_mark = "✓" if cell.get("success") else "✗"
            # _fmt handles None → "N/A"; we add "%" suffix for clarity.
            def pct(field):
                v = cell.get(field)
                return "N/A" if v is None else f"{v:.0f}"

            lat = (
                "N/A" if cell.get("latency_seconds") is None
                else f"{cell['latency_seconds']:.1f}"
            )
            tokens = (
                "N/A" if cell.get("total_tokens") is None
                else f"{cell['total_tokens']:,}"
            )
            cost = (
                "N/A" if cell.get("cost_usd") is None
                else f"${cell['cost_usd']:.5f}"
            )

            row = (
                f"{tc_id:<8}{topic:<28}{success_mark:<4}"
                f"{lat:<8}{pct('citation_accuracy_pct'):<7}"
                f"{pct('hallucination_rate_pct'):<7}"
                f"{pct('coverage_rate_pct'):<7}"
                f"{pct('adversarial_pass_pct'):<7}"
                f"{pct('agent_disagreement_pct'):<7}"
                f"{tokens:<9}{cost:<10}"
            )
            print(row)


# Subset of fields that go into the per-(backend, mode, TC) CSV row.
# Keep this list aligned with the metrics the report tables care about.
_CSV_FIELDS = [
    "backend", "mode", "tc_id", "topic", "category", "success",
    "latency_seconds",
    "citation_accuracy_pct",
    "hallucination_rate_pct",
    "coverage_rate_pct",
    "adversarial_pass_pct",
    "agent_disagreement_pct",
    "input_tokens", "output_tokens", "total_tokens",
    "llm_calls", "cost_usd",
    "error",  # last column so a missing value doesn't shift others
]


def save_csv(results_by_backend, timestamp):
    """Write one CSV row per (backend, mode, TC).

    LaTeX-friendly: paste into csvautotabular, or open in Excel /
    Numbers to drag-and-drop charts straight into the report.
    """
    # TC.category lives in test_data.py rather than the result dict, so
    # build a quick lookup table once.
    tc_meta = {tc["id"]: tc for tc in TEST_CONTRACTS}

    filename = f"evaluation_results_{timestamp}.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for backend, results in results_by_backend.items():
            for r in results:
                for mode in ("baseline", "new"):
                    cell = r.get(mode, {})
                    if not cell:
                        continue
                    tc_id = cell.get("id", "")
                    row = dict(cell)  # copy all metric fields
                    row["backend"]  = backend
                    row["mode"]     = mode
                    row["tc_id"]    = tc_id
                    row["category"] = tc_meta.get(tc_id, {}).get("category", "happy_path")
                    writer.writerow(row)

    print(f"📊 Per-TC CSV saved to: {filename}  ({len(_CSV_FIELDS)} columns)")
    return filename


def save_results(results_by_backend, timestamp=None):
    """Persist a JSON dump of every per-contract result (without full reports).

    ``timestamp`` is accepted so callers can share one suffix between
    the JSON and CSV outputs (cleaner artifact pairing).
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evaluation_results_{timestamp}.json"

    save_data = {}
    for backend, results in results_by_backend.items():
        save_data[backend] = [
            {
                "baseline": {k: v for k, v in r["baseline"].items()
                             if k != "report_full"},
                "new": {k: v for k, v in r["new"].items()
                        if k != "report_full"},
            }
            for r in results
        ]

    with open(filename, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\n💾 Detailed results saved to: {filename}")
    return filename


# ════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════

def _parse_backends(arg_value):
    """Parse and validate the --backends CSV arg."""
    raw = [b.strip().lower() for b in arg_value.split(",") if b.strip()]
    for b in raw:
        if b not in VALID_BACKENDS:
            raise SystemExit(
                f"Unknown backend {b!r}. Must be one of {VALID_BACKENDS}."
            )
    if not raw:
        raise SystemExit("--backends cannot be empty.")
    # Dedupe but preserve order
    seen, ordered = set(), []
    for b in raw:
        if b not in seen:
            seen.add(b)
            ordered.append(b)
    return ordered


def _preflight(backends):
    """Bail early if a requested backend is obviously not usable."""
    if DEEPSEEK in backends and not os.getenv("DEEPSEEK_API_KEY"):
        raise SystemExit(
            "DeepSeek requested but DEEPSEEK_API_KEY is not set in .env.\n"
            "Either set the key or rerun with --backends ollama."
        )


def main():
    parser = argparse.ArgumentParser(description="Aletheia multi-backend evaluator")
    parser.add_argument(
        "--backends",
        default=f"{OLLAMA},{DEEPSEEK}",
        help=(
            "Comma-separated list of LLM backends to evaluate "
            f"(any of {','.join(VALID_BACKENDS)}). "
            f"Default: {OLLAMA},{DEEPSEEK}."
        ),
    )
    args = parser.parse_args()
    backends = _parse_backends(args.backends)
    _preflight(backends)

    print("=" * 70)
    print("🛡️  ALETHEIA EVALUATION SUITE")
    print(f"   Started:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Backends:  {', '.join(backends)}")
    print(f"   Test cases: {len(TEST_CONTRACTS)}")
    print("=" * 70)

    results_by_backend = {}
    summaries = {}
    for backend in backends:
        results = run_backend_evaluation(backend)
        results_by_backend[backend] = results
        summaries[backend] = report_per_backend(backend, results)
        report_adversarial_section(backend, results)
        report_per_tc_details(backend, results)

    report_cross_backend(results_by_backend)

    # Persist both formats. JSON is for re-loading in Python;
    # CSV is for direct paste into LaTeX (csvautotabular) or Excel.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(results_by_backend, timestamp)
    save_csv(results_by_backend, timestamp)

    print("\n" + "=" * 70)
    print("🎯 NOTES")
    print("=" * 70)
    print("  • Per-backend table answers: 'does the verification step help "
          "for *this* LLM?'")
    print("  • Cross-backend table answers: 'does switching LLM help, "
          "holding pipeline constant?'")
    print("  • Both numbers belong in the paper/report — they isolate "
          "different sources of gain.")


if __name__ == "__main__":
    main()
