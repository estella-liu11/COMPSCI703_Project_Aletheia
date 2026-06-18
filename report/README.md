# Aletheia — Final Report (NeurIPS 2025 Format)

## Files in this folder

- `aletheia_report.tex` — the report, ready to compile with NeurIPS 2025 style.

## Overleaf usage (5 minutes)

1. **Get NeurIPS 2025 style file**:
   - Download from https://nips.cc/Conferences/2025/PaperInformation/StyleFiles
   - You only need `neurips_2025.sty` (plus optionally `neurips_2025.bst` if you switch to BibTeX later)

2. **Create Overleaf project**:
   - New project → "Blank Project" → name "Aletheia Report"
   - In file tree (left panel): right-click → Upload → upload `neurips_2025.sty`
   - Right-click → Upload → upload `aletheia_report.tex`
   - Delete the default `main.tex` Overleaf created
   - Click the gear icon → "Main document" → set to `aletheia_report.tex`
   - Click "Recompile"

3. **Should compile clean**. If you see errors, almost always one of:
   - `neurips_2025.sty` not uploaded → re-upload
   - PDF mode confusion → top of `.tex`, the `final` option means de-anonymised; change to `[preprint]` if you want author name visible without `final`

## Before submitting — find-and-replace these

The `.tex` file contains `[VERIFY: ...]` and `[REPLACE: ...]` markers wherever
you need to either confirm a citation or insert your real text. Use Ctrl+F in
Overleaf to find them.

### `[REPLACE: ...]` — you must fill these in

| Marker | What to put |
|---|---|
| `[REPLACE: email]` (line ~62) | Your university email |
| `[REPLACE: GitHub URL]` (×2, around line ~430 and conclusion) | Your GitHub repo URL |

### `[VERIFY: ...]` — citations that need to be real

| Marker | Suggested source |
|---|---|
| Big-4 hourly rates | https://www.legalbusiness.co.uk/rate-cards/ or Deloitte NZ partner-level filings |
| Cross-border tax practitioner survey | RegTech Analyst 2024 report, or BIS legal-tech survey |
| RegTech market size | Grand View Research RegTech Market Report 2024 |
| LegalTech market size | Allied Market Research LegalTech 2023 |
| Big-4 NZ rate card | Deloitte NZ public salary survey, or NZ Law Society fee guide |
| 2025 multi-agent survey | Pick from recent arXiv 2024-2025 surveys |

If you can't find verifiable numbers for some, soften the language:
- "Industry estimates place X at ~$Y" instead of citing a specific report
- Drop the precise number and use qualitative language

## What this report covers (matches your eval data)

- ✅ Real numbers from your 2026-06-15 evaluation run
- ✅ All 4 core requirements (Agentic Autonomy, Profit Logic, Trust & Robustness, Commercial Stress Test)
- ✅ Honest discussion of the DeepSeek adversarial regression as a research finding (not a hidden bug)
- ✅ NeurIPS Checklist filled in
- ✅ Hyperparameters from `config.py` copied verbatim
- ✅ Test contracts table from `test_data.py`
- ✅ Sample prompt from `agent_audit.py`
- ✅ References to CoVe, Self-RAG, RAG, LangGraph, prompt-injection papers

## Page budget (rough estimate)

| Section | Estimated pages |
|---|---|
| Abstract + Introduction | 1.5 |
| Related Work | 0.5 |
| System Design | 2.0 (incl. figure 1) |
| Implementation | 0.5 |
| Evaluation (3 tables) | 2.5 |
| Cost-Benefit (1 table) | 0.5 |
| Discussion | 1.0 |
| Conclusion | 0.3 |
| **Main body total** | **~8.3 of 9 allowed** |
| References | separate, doesn't count |
| Appendix | separate, doesn't count |

If you exceed 9 pages, the easiest cuts (no information loss):
- Compress the metric-definition list (`\begin{description}`) into a paragraph
- Remove some prose in the Related Work paragraphs
- Move the adversarial table to the appendix and reference it from text

## Figures you should add (optional but high-impact)

- **Figure 1** (currently ASCII): render the LangGraph topology from `figures/pre_hitl_graph.mmd` and `figures/post_hitl_graph.mmd`:
  1. Open https://mermaid.live/
  2. Paste contents of `pre_hitl_graph.mmd`
  3. Click "Actions" → "Download PNG"
  4. Upload to Overleaf, replace the `\fbox{...}` ASCII art with `\includegraphics`
- **Figure 2** (optional): bar chart of per-backend Coverage / Adversarial Pass / Latency. Use the CSV `evaluation_results_20260615_191917.csv` in Excel or matplotlib.
- **Figure 3** (optional): Streamlit screenshot showing the trust-tier-coloured final report.

## What's already pulled from your code

- Citation extractor consistency (fixed yesterday)
- Adversarial pass-rate metric (configurable threshold)
- Token cost instrumentation with cl100k_base
- LangGraph pre-HITL + post-HITL graph topology
- All 12 test cases including the 4 adversarial ones
- Real DeepSeek + Llama numbers from your 6/15 run

## Demo video tips (separate deliverable, 3 min max)

Your 3-min "roadshow" video should follow the demo script
(`scripts/demo.py`). Suggested structure:

1. **0:00-0:30** — Problem ("$750 per contract review × 50 contracts/yr...")
2. **0:30-1:30** — Live demo (upload contract → dual juniors → HITL → final report)
3. **1:30-2:15** — Switch backend live (sidebar Llama → DeepSeek, show speedup)
4. **2:15-2:45** — Cost number ($0.0023 vs $437)
5. **2:45-3:00** — Call to action (link to GitHub / pricing)

Use OBS or QuickTime to record. Use `scripts/demo.py --no-pause` if you
want the demo to run faster on camera.

---

Last updated: 2026-06-15 (after the 12-TC × 2-backend evaluation run).
