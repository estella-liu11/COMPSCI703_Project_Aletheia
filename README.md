# 🛡️ Aletheia: HITL Multi-Agent Compliance Auditor

**Aletheia** (Greek: "Truth") is an AI-driven compliance system that audits financial
contracts against NZ–US cross-border tax regulations. It uses a
**Multi-Agent RAG + Human-in-the-Loop (HITL)** architecture, with an independent
citation re-retrieval step in the Chief Officer to minimise legal hallucinations.

---

## 🎯 Project Overview

Cross-border contract auditing is expensive and slow. Traditional single-pass RAG
systems hallucinate citations and miss obligations from different angles. Aletheia
fixes this with:

1. **Two parallel Junior Auditors** with opposing mindsets (Risk Hunter / Compliance Maven)
2. **A human adjudication step** — the user decides which auditor(s) to trust
3. **A self-verifying Chief Officer** that re-retrieves each citation independently
   from the vector DB before signing off
4. **A separate Quick Q&A mode** for ad-hoc regulatory questions (no upload needed)

---

## 🧠 Core Concepts

### 1. Dual-Junior Adversarial Audit
Two Junior agents read the same contract + the same regulatory context, but with
opposite framings:

| Agent | Role | Output |
|-------|------|--------|
| 🔴 **Risk Hunter** (Junior A) | Assume contract has hidden violations | `[Risk #N]` items |
| 🔵 **Compliance Maven** (Junior B) | Assume contract is missing required clauses | `[Gap #N]` items |

### 2. Human-in-the-Loop Adjudication
After the Juniors finish, the UI shows both outputs side-by-side. The user picks
one of four routes before the Chief Officer runs:

- `BOTH` — merge findings from both, flag contradictions
- `ONLY_A` — trust only the Risk Hunter
- `ONLY_B` — trust only the Compliance Maven
- `NEITHER` — escalate to a human reviewer (no AI final report generated)

### 3. Self-Verifying Chief Officer
The Chief Officer does **not** trust the Juniors' citations. For every citation
extracted from the selected Junior(s), it independently re-queries Pinecone for
the top-3 most relevant chunks (`re_retrieve_citations()`), then classifies each
citation as `✅ VERIFIED`, `⚠️ PARTIAL`, or `❌ UNVERIFIED` in the final report.

### 4. MMR Diversity Retrieval
Initial context retrieval uses Maximal Marginal Relevance to avoid redundant chunks:

```python
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20},
)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **LLM** | Llama 3.2 via Ollama (local, `temperature=0`) |
| **Vector DB** | Pinecone Serverless, index `aletheia-legal-db` (384-dim) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` |
| **Orchestration** | LangChain (Pinecone + Ollama + HuggingFace integrations) |
| **UI** | Streamlit (two tabs: Contract Audit / Quick Q&A) |
| **Language** | Python 3.13+ |

---

## 📁 Project Structure

```
COMPSCI703_Project/
├── app.py                       # Streamlit UI — two tabs (Audit + Quick Q&A)
├── agent_audit.py               # Core engine: dual juniors + verifying chief + quick QA
├── agent_audit_baseline.py      # Old single-junior version (used only by evaluation)
├── ingest.py                    # PDF → cleaned chunks → Pinecone pipeline
├── test_data.py                 # 8 test contract fixtures (TC-001…TC-008)
├── test_evaluation.py           # Baseline-vs-new evaluator + metrics report
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example                 # PINECONE_API_KEY template
├── .github/workflows/
│   ├── test.yml                 # CI: lint + import smoke test
│   └── docker.yml               # CD: build & push Docker image
├── DATA-library/                # Regulatory PDFs ingested into Pinecone
│   ├── Double Taxation Relief (United States of America) Order 1983 …pdf
│   └── compliance-simplification-bill-act-commentary.pdf
├── README.md
├── TECH_DOC.md
└── IDEA_DOC.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.13+
- [Ollama](https://ollama.ai/) with `llama3.2` pulled
- A [Pinecone](https://www.pinecone.io/) account (free tier works)

### Installation

```bash
# 1. Clone
git clone <your-repo-url>
cd COMPSCI703_Project

# 2. Virtual env
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 3. Dependencies
pip install -r requirements.txt

# 4. Ollama
ollama pull llama3.2
ollama serve                         # leave running

# 5. Env vars
cp .env.example .env
# edit .env and set PINECONE_API_KEY=...

# 6. Ingest the regulatory PDFs into Pinecone (one-time)
python ingest.py

# 7. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 🐳 Docker Deployment

The Docker image runs the Streamlit app only. Ollama still has to run on the
host (or be wired up separately) because `agent_audit.py` connects to it at
`http://localhost:11434`.

```bash
# With compose (recommended)
docker-compose up --build

# Or manual
docker build -t aletheia .
docker run -p 8501:8501 --env-file .env aletheia
```

---

## 🧪 Evaluation Suite

`test_evaluation.py` runs all 8 test contracts in `test_data.py` against both
the old single-junior baseline (`agent_audit_baseline.py`) and the current
multi-agent + verification version, then prints a comparison report and saves
a JSON dump.

```bash
python test_evaluation.py
```

**Metrics reported:**

| Metric | Definition |
|--------|------------|
| Citation Accuracy (%) | `(total - unverified) / total` |
| Hallucination Rate (%) | `unverified / total` (counts `[UNVERIFIED]` flags) |
| Coverage Rate (%) | % of designed expected issues that were detected by keyword match |
| Avg Latency (s) | Per-contract end-to-end wall time |

> Requires both Pinecone and Ollama to be reachable — does **not** run in CI.

---

## 📊 How It Works

```
┌──────────────────────────────────────────────────────────────┐
│  User uploads contract (PDF / DOCX / TXT)                    │
└─────────────────────────────────┬────────────────────────────┘
                                  ▼
┌──────────────────────────────────────────────────────────────┐
│  MMR retrieval (fetch_k=20 → k=5 diverse chunks)             │
└─────────────────────────────────┬────────────────────────────┘
                                  ▼
┌────────────────────────────┬─────────────────────────────────┐
│ 🔴 Risk Hunter (Junior A)  │ 🔵 Compliance Maven (Junior B)  │
│    finds violations        │    finds gaps                    │
└────────────────────────────┴─────────────────────────────────┘
                                  ▼
┌──────────────────────────────────────────────────────────────┐
│  HITL: user picks BOTH / ONLY_A / ONLY_B / NEITHER           │
└─────────────────────────────────┬────────────────────────────┘
                                  ▼
┌──────────────────────────────────────────────────────────────┐
│  Chief Officer:                                              │
│   1. extract_citations() from selected Junior(s)             │
│   2. re_retrieve_citations() — independent Pinecone query    │
│      per citation                                            │
│   3. Classify each: ✅ VERIFIED / ⚠️ PARTIAL / ❌ UNVERIFIED  │
└─────────────────────────────────┬────────────────────────────┘
                                  ▼
┌──────────────────────────────────────────────────────────────┐
│  Final report grouped by HIGH / MEDIUM / LOW                 │
│  + Unverifiable Claims section + Recommendations             │
└──────────────────────────────────────────────────────────────┘
```

The Quick Q&A tab is a simpler one-shot path: MMR retrieve (`k=4`) → single
LLM call with strict source-grounding prompt → answer with Short Answer /
Explanation / Sources / Confidence.

---

## 🎓 Academic Context

Developed for **COMPSCI 703 — Advanced Topics in Computer Science**.
Focus areas:

- Agentic / multi-agent RAG
- Human-in-the-loop verification
- Hallucination reduction via independent citation re-retrieval
- System 2 reasoning patterns in LLM pipelines

---

## 📝 License

Academic use only.

---

## 👤 Author

**Estella Liu** — COMPSCI 703, 2025
