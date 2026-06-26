"""
2x2 ablation study (component contribution analysis).

Cell  Dual-junior  Verification  Where the data comes from
M1    No           None          existing evaluation_results_*.json (baseline)
M2    Yes          CoVe          existing evaluation_results_*.json (new)
M3    No           CoVe          [THIS SCRIPT] ONLY_A route + VERIFY_MODE=cove
M4    Yes          None          [THIS SCRIPT] BOTH route + VERIFY_MODE=none

Run:
    python scripts/run_ablation.py deepseek
    python scripts/run_ablation.py ollama
    python scripts/run_ablation.py both
"""

import os
import sys
import json
import time
from datetime import datetime

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_data import TEST_CONTRACTS
from test_evaluation import (
    count_total_citations,
    count_unverified_citations,
    calculate_citation_accuracy,
    calculate_hallucination_rate,
    calculate_coverage,
    calculate_adversarial_pass,
    calculate_agent_disagreement,
)
from cost_estimator import reset_usage, get_usage, estimate_cost
from llm_factory import get_backend_name, DEEPSEEK


def _run_one_audit(tc, ablation_cell):
    """Run TC under one ablation configuration and return metric dict."""
    # Late import so env-var changes take effect on the imported module's
    # `os.getenv` reads (they read every time, so this is actually safe).
    from agent_audit import run_dual_juniors, run_chief_officer

    # ── Configure ablation cell via env vars ──
    if ablation_cell == "single_cove":
        os.environ["VERIFY_MODE"] = "cove"
        decision = "ONLY_A"      # discard Junior B
    elif ablation_cell == "dual_nocove":
        os.environ["VERIFY_MODE"] = "none"
        decision = "BOTH"        # keep Junior B, skip verification
    else:
        raise ValueError(f"Unknown ablation_cell: {ablation_cell}")

    start = time.time()
    reset_usage()
    disagreement = None

    try:
        juniors = run_dual_juniors(tc["contract"])

        # Disagreement only meaningful when both juniors influence the final report
        if ablation_cell == "dual_nocove":
            disagreement = calculate_agent_disagreement(
                juniors["junior_a_output"],
                juniors["junior_b_output"],
            )

        report = run_chief_officer(
            junior_a_output=juniors["junior_a_output"],
            junior_b_output=juniors["junior_b_output"],
            context=juniors["context"],
            user_decision=decision,
        )

        latency = time.time() - start
        usage = get_usage()
        backend = get_backend_name()
        model = (
            os.getenv("DEEPSEEK_MODEL") if backend == DEEPSEEK
            else os.getenv("OLLAMA_MODEL")
        )
        cost = estimate_cost(
            usage["input_tokens"], usage["output_tokens"], backend, model,
        )

        return {
            "id": tc["id"],
            "topic": tc["topic"],
            "mode": ablation_cell,
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
                round(disagreement, 1) if disagreement is not None else None
            ),
            "success": True,
            "success_pct": 100.0,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["input_tokens"] + usage["output_tokens"],
            "llm_calls": usage["calls"],
            "cost_usd": round(cost, 6),
            "report_preview": report[:300],
        }
    except Exception as e:
        return {
            "id": tc["id"],
            "topic": tc["topic"],
            "mode": ablation_cell,
            "error": str(e),
            "success": False,
            "success_pct": 0.0,
        }
    finally:
        # Restore defaults so subsequent imports / runs are not polluted
        os.environ.pop("VERIFY_MODE", None)


def run_backend(backend):
    """Run both ablation cells across all 12 TCs on one backend."""
    os.environ["LLM_BACKEND"] = backend
    print(f"\n{'='*60}")
    print(f"  Ablation run on backend: {backend}")
    print(f"{'='*60}\n")

    results = {"single_cove": [], "dual_nocove": []}

    for tc in TEST_CONTRACTS:
        for cell in ("single_cove", "dual_nocove"):
            print(f"  [{tc['id']}] {cell:<15} ...", end=" ", flush=True)
            r = _run_one_audit(tc, cell)
            status = "ok" if r.get("success") else f"ERR ({r.get('error', '?')[:40]})"
            lat = r.get("latency_seconds", "—")
            print(f"{status:<10}  {lat}s")
            results[cell].append(r)

    return results


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "deepseek"
    backends = ["deepseek", "ollama"] if arg == "both" else [arg]

    all_results = {}
    for b in backends:
        all_results[b] = run_backend(b)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = (
        f"/Users/liuhanyu/Desktop/COMPSCI703_Project/"
        f"ablation_results_{ts}.json"
    )
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Ablation complete. Saved to: {out_path}")
    print("\nNext: run scripts/summarise_ablation.py to print the 2x2 table.")


if __name__ == "__main__":
    main()
