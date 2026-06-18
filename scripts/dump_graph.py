"""
Dump LangGraph topologies for the report.

Usage:
    python scripts/dump_graph.py              # → write Mermaid + PNG to ./figures/
    python scripts/dump_graph.py --out docs   # → write into docs/ instead
    python scripts/dump_graph.py --no-png     # → skip PNG (no Mermaid CLI needed)

Output files:
    figures/pre_hitl_graph.mmd
    figures/pre_hitl_graph.png   (only with --png; needs Mermaid CLI / Graphviz)
    figures/post_hitl_graph.mmd
    figures/post_hitl_graph.png

The Mermaid source is what you paste into the LaTeX report (with the
mermaid LaTeX package, or render once on https://mermaid.live/ and
embed the SVG). Mermaid is reproducible, version-controllable, and
beats hand-drawn ASCII diagrams in the appendix.
"""

import argparse
import sys
from pathlib import Path

# Allow running from project root or from scripts/ — add project root to path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import agent_audit  # noqa: E402  — must come after sys.path tweak


def _write(path, content):
    """Write text to path, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"   ✓ wrote {path.relative_to(PROJECT_ROOT)} ({len(content)} bytes)")


def dump_graph(graph, name, out_dir, want_png):
    """Dump one compiled graph as Mermaid (+ optional PNG)."""
    g = graph.get_graph()

    print(f"\n📐 {name}")
    mermaid = g.draw_mermaid()
    _write(out_dir / f"{name}.mmd", mermaid)

    if want_png:
        try:
            png_bytes = g.draw_mermaid_png()
            png_path = out_dir / f"{name}.png"
            png_path.parent.mkdir(parents=True, exist_ok=True)
            png_path.write_bytes(png_bytes)
            print(f"   ✓ wrote {png_path.relative_to(PROJECT_ROOT)} "
                  f"({len(png_bytes)} bytes)")
        except Exception as e:
            print(f"   ⚠️  PNG rendering failed (Mermaid CLI not installed?): {e}")
            print(f"      Tip: paste the .mmd content into https://mermaid.live/")


def main():
    parser = argparse.ArgumentParser(description="Dump Aletheia LangGraph topologies")
    parser.add_argument(
        "--out", default="figures",
        help="Output directory (default: figures/)",
    )
    parser.add_argument(
        "--no-png", action="store_true",
        help="Skip PNG rendering (Mermaid source only, no external deps).",
    )
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out
    want_png = not args.no_png

    print("─" * 60)
    print("🛡️  Aletheia LangGraph topology dump")
    print(f"    Output: {out_dir}")
    print(f"    PNG:    {'yes' if want_png else 'skipped'}")
    print("─" * 60)

    pre = agent_audit._build_pre_hitl_graph()
    dump_graph(pre, "pre_hitl_graph", out_dir, want_png)

    post = agent_audit._build_post_hitl_graph()
    dump_graph(post, "post_hitl_graph", out_dir, want_png)

    print("\n✅ Done.")
    if want_png:
        print("   Paste the .mmd files into https://mermaid.live/ if PNG "
              "rendering didn't work.")


if __name__ == "__main__":
    main()
