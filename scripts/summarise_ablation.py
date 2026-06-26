"""
Merge the existing baseline+verified evaluation with the 2 new ablation cells
and print the full 2x2 ablation table to stdout.

Usage:
    python scripts/summarise_ablation.py \
        evaluation_results_20260615_191917.json \
        ablation_results_*.json
"""

import sys
import json
import glob
import statistics
import math


def _safe_mean(vals):
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else float("nan")


def _safe_sd(vals):
    vals = [v for v in vals if v is not None]
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def _adv_pass(audits):
    """Pass rate over adversarial TCs only (4 of 12)."""
    adv = [a for a in audits if a.get("id") in ("TC-009", "TC-010", "TC-011", "TC-012")]
    if not adv:
        return float("nan")
    return 100.0 * sum(1 for a in adv if a.get("adversarial_pass_pct") == 100.0) / len(adv)


def _aggregate(audits):
    return {
        "n": len(audits),
        "cov":  (_safe_mean([a.get("coverage_rate_pct") for a in audits]),
                 _safe_sd([a.get("coverage_rate_pct") for a in audits])),
        "ca":   (_safe_mean([a.get("citation_accuracy_pct") for a in audits]),
                 _safe_sd([a.get("citation_accuracy_pct") for a in audits])),
        "hr":   (_safe_mean([a.get("hallucination_rate_pct") for a in audits]),
                 _safe_sd([a.get("hallucination_rate_pct") for a in audits])),
        "lat":  (_safe_mean([a.get("latency_seconds") for a in audits]),
                 _safe_sd([a.get("latency_seconds") for a in audits])),
        "tok":  _safe_mean([a.get("total_tokens") for a in audits]),
        "calls":_safe_mean([a.get("llm_calls") for a in audits]),
        "cost": _safe_mean([a.get("cost_usd") for a in audits]),
        "adv":  _adv_pass(audits),
    }


def _print_table(title, cells):
    print(f"\n{'='*78}\n{title}\n{'='*78}")
    print(f"{'Cell':<32} {'Cov%':>14} {'CA%':>14} {'HR%':>13} {'Lat(s)':>12} {'AdvPR%':>8}")
    print("-" * 78)
    for name, agg in cells:
        cov = f"{agg['cov'][0]:5.1f} ±{agg['cov'][1]:4.1f}"
        ca  = f"{agg['ca'][0]:5.1f} ±{agg['ca'][1]:4.1f}"
        hr  = f"{agg['hr'][0]:5.1f} ±{agg['hr'][1]:4.1f}"
        lat = f"{agg['lat'][0]:6.1f}"
        adv = f"{agg['adv']:5.1f}"
        print(f"{name:<32} {cov:>14} {ca:>14} {hr:>13} {lat:>12} {adv:>8}")
    print("-" * 78)
    print(f"  Extra: cost (USD/audit), llm_calls, total_tokens")
    for name, agg in cells:
        print(f"  {name:<30} ${agg['cost']:.5f}  calls={agg['calls']:.1f}  tokens={agg['tok']:.0f}")


def _delta(after, before):
    """Print Δ between two cells for the key metrics."""
    return {
        "cov":  after["cov"][0] - before["cov"][0],
        "lat":  after["lat"][0] - before["lat"][0],
        "adv":  after["adv"]    - before["adv"],
        "tok":  after["tok"]    - before["tok"],
    }


def main():
    eval_path = sys.argv[1] if len(sys.argv) > 1 else \
        glob.glob("evaluation_results_*.json")[-1]
    abl_path = sys.argv[2] if len(sys.argv) > 2 else \
        sorted(glob.glob("ablation_results_*.json"))[-1]

    print(f"Loading evaluation: {eval_path}")
    print(f"Loading ablation:   {abl_path}")
    with open(eval_path) as f:
        ev = json.load(f)
    with open(abl_path) as f:
        ab = json.load(f)

    for backend in ("deepseek", "ollama"):
        if backend not in ev or backend not in ab:
            continue
        # 4 cells: 2 from existing eval, 2 from ablation
        m1 = _aggregate([a["baseline"] for a in ev[backend]])
        m2 = _aggregate([a["new"] for a in ev[backend]])
        m3 = _aggregate(ab[backend]["single_cove"])
        m4 = _aggregate(ab[backend]["dual_nocove"])

        _print_table(
            f"{backend.upper()} 2x2 ablation",
            [
                ("M1: single + no verify (baseline)", m1),
                ("M3: single + CoVe", m3),
                ("M4: dual + no verify", m4),
                ("M2: dual + CoVe (full Aletheia)", m2),
            ],
        )

        print(f"\n  Δ contributions (vs M1 baseline):")
        for name, m, against in [
            ("dual-junior alone (M4-M1)", m4, m1),
            ("CoVe alone (M3-M1)", m3, m1),
            ("dual + CoVe combined (M2-M1)", m2, m1),
        ]:
            d = _delta(m, against)
            print(f"    {name:<30}  ΔCov={d['cov']:+5.1f}pp  Δlat={d['lat']:+6.1f}s  ΔadvPR={d['adv']:+5.1f}pp")


if __name__ == "__main__":
    main()
