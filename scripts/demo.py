"""
Aletheia recorded-demo driver.

Runs a scripted end-to-end audit straight from the terminal so the
3-minute road-show video doesn't have to chase Streamlit UI delays.

Steps shown (one short scene per step):
    Scene 1  Banner — what the system is + which backends are live
    Scene 2  Contract upload — load a real fixture from test_data.py
    Scene 3  Live external API — show today's USD→NZD rate
    Scene 4  Pre-HITL graph — Risk Hunter + Compliance Maven in parallel
    Scene 5  Human-in-the-loop — pick BOTH (announced, not interactive)
    Scene 6  Post-HITL graph — CoVe verification + Chief Officer
    Scene 7  Final report — colour-coded by severity, with trust labels
    Scene 8  Cost + telemetry — tokens, dollars, LLM-call count

Usage::

    python scripts/demo.py                       # default scenario
    python scripts/demo.py --tc TC-010           # adversarial demo
    python scripts/demo.py --backend deepseek    # cross-backend live switch

Designed for screen recording: deliberate pauses (PAUSE_S) give the
narrator time, and ANSI colours make the output visually distinct from
the surrounding terminal. Each scene is gated by ``--no-pause`` for
non-interactive (e.g. CI) runs.
"""

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────
# ANSI colour helpers (no external dep)
# ─────────────────────────────────────────────────────────────────

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    YEL = "\033[33m"
    BLU = "\033[34m"
    GRN = "\033[32m"
    CYN = "\033[36m"
    MAG = "\033[35m"


PAUSE_S = 1.5


def _pause(seconds=PAUSE_S):
    if not _NO_PAUSE:
        time.sleep(seconds)


def _scene(num, title):
    print()
    print(f"{C.MAG}{C.BOLD}━━ Scene {num}  {title} ━━{C.RESET}")
    _pause(0.5)


def _say(text, c=C.CYN):
    print(f"{c}{text}{C.RESET}")


_NO_PAUSE = False  # set by --no-pause


# ─────────────────────────────────────────────────────────────────
# Scenes
# ─────────────────────────────────────────────────────────────────

def scene_banner():
    _scene(1, "Aletheia — Compound AI Audit System")
    print()
    print(f"  {C.BOLD}🛡️  Aletheia{C.RESET}")
    print(f"  Cross-border tax-contract auditor")
    print(f"  Compound AI: vector DB + LLM + external API")
    print()

    from llm_factory import describe_active_backend
    from vector_store_factory import describe_active_backend as describe_vec
    print(f"  LLM        : {describe_active_backend()}")
    print(f"  Vector DB  : {describe_vec()}")
    _pause()


def scene_contract(tc_id):
    _scene(2, f"Load contract fixture ({tc_id})")
    from test_data import TEST_CONTRACTS
    tc = next((t for t in TEST_CONTRACTS if t["id"] == tc_id), TEST_CONTRACTS[0])
    print(f"  {C.DIM}{tc['topic']}{C.RESET}")
    print()
    # show first ~12 lines
    for line in tc["contract"].strip().splitlines()[:12]:
        print(f"  {C.DIM}│{C.RESET} {line}")
    print()
    _pause()
    return tc


def scene_external_api():
    _scene(3, "External-API enrichment — live FX")
    from external_tools import lookup_exchange_rate
    info = lookup_exchange_rate("USD", "NZD")
    if info:
        print(f"  {C.GRN}💱 Live rate:{C.RESET} "
              f"1 USD = {info['rate']:.4f} NZD ({info['date']}, "
              f"{info['source']})")
    else:
        print(f"  {C.YEL}⚠️  FX API unreachable — audit will continue without enrichment{C.RESET}")
    _pause()


def scene_juniors(tc):
    _scene(4, "Pre-HITL graph: Risk Hunter ⊥ Compliance Maven (parallel)")
    from agent_audit import run_dual_juniors
    from cost_estimator import reset_usage
    reset_usage()
    t0 = time.time()
    result = run_dual_juniors(tc["contract"])
    dt = time.time() - t0
    print()
    print(f"  {C.RED}🔴 Risk Hunter (Junior A){C.RESET}")
    print(f"  {C.DIM}{result['junior_a_output'][:400]}{C.RESET}")
    print()
    print(f"  {C.BLU}🔵 Compliance Maven (Junior B){C.RESET}")
    print(f"  {C.DIM}{result['junior_b_output'][:400]}{C.RESET}")
    print()
    print(f"  {C.DIM}(parallel execution: {dt:.1f}s){C.RESET}")
    _pause()
    return result


def scene_hitl(decision):
    _scene(5, "Human-in-the-loop adjudication")
    print(f"  User picks: {C.BOLD}{decision}{C.RESET}")
    print(f"  {C.DIM}(BOTH / ONLY_A / ONLY_B / NEITHER — UI chip shows checkmarks){C.RESET}")
    _pause()


def scene_chief(juniors, decision):
    _scene(6, "Post-HITL graph: CoVe verify → Chief Officer")
    from agent_audit import run_chief_officer
    t0 = time.time()
    final = run_chief_officer(
        junior_a_output=juniors["junior_a_output"],
        junior_b_output=juniors["junior_b_output"],
        context=juniors["context"],
        user_decision=decision,
    )
    dt = time.time() - t0
    print(f"  {C.DIM}(chief synthesis: {dt:.1f}s){C.RESET}")
    return final


def scene_report(final):
    _scene(7, "Final report (severity-coloured + trust labels)")
    print()
    # Quick syntax-highlight the [N] [SEV] [Trust] tokens
    for line in final.splitlines():
        stripped = line.strip()
        if "[HIGH]" in stripped:
            print(f"  {C.RED}{stripped}{C.RESET}")
        elif "[MED]" in stripped:
            print(f"  {C.YEL}{stripped}{C.RESET}")
        elif "[LOW]" in stripped:
            print(f"  {C.BLU}{stripped}{C.RESET}")
        else:
            print(f"  {stripped}")
    _pause(2.0)


def scene_cost_telemetry():
    _scene(8, "Commercial Stress Test — tokens, dollars, calls")
    from cost_estimator import get_usage, estimate_cost
    from llm_factory import get_backend_name, DEEPSEEK
    usage = get_usage()
    backend = get_backend_name()
    model = (
        os.getenv("DEEPSEEK_MODEL") if backend == DEEPSEEK
        else os.getenv("OLLAMA_MODEL")
    )
    cost = estimate_cost(usage["input_tokens"], usage["output_tokens"],
                         backend, model)
    print()
    print(f"  {C.BOLD}Per-audit telemetry:{C.RESET}")
    print(f"    Backend         : {backend} ({model or 'default'})")
    print(f"    Input tokens    : {usage['input_tokens']:>7,}")
    print(f"    Output tokens   : {usage['output_tokens']:>7,}")
    print(f"    LLM calls       : {usage['calls']:>7}")
    print(f"    {C.GRN}Cost (USD)      : ${cost:>10.5f}{C.RESET}")
    print()
    print(f"  {C.DIM}Human equivalent (Big-4 senior associate, NZD 250/hr × 3 h)"
          f" ≈ NZD 750  →  {C.GRN}~125,000× cheaper{C.DIM} per audit.{C.RESET}")
    _pause(2.0)


# ─────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Aletheia recorded-demo driver")
    parser.add_argument("--tc", default="TC-001",
                        help="Test contract id to demo (default: TC-001)")
    parser.add_argument("--decision", default="BOTH",
                        choices=["BOTH", "ONLY_A", "ONLY_B", "NEITHER"],
                        help="Simulated HITL decision (default: BOTH)")
    parser.add_argument("--backend", choices=["ollama", "deepseek"],
                        help="Force a specific LLM backend for this run.")
    parser.add_argument("--no-pause", action="store_true",
                        help="Skip the per-scene pauses (for CI / smoke tests).")
    args = parser.parse_args()

    global _NO_PAUSE
    _NO_PAUSE = args.no_pause

    if args.backend:
        os.environ["LLM_BACKEND"] = args.backend

    scene_banner()
    tc = scene_contract(args.tc)
    scene_external_api()
    juniors = scene_juniors(tc)
    scene_hitl(args.decision)
    final = scene_chief(juniors, args.decision)
    scene_report(final)
    scene_cost_telemetry()

    print()
    print(f"{C.MAG}{C.BOLD}━━ Demo complete ━━{C.RESET}")


if __name__ == "__main__":
    main()
