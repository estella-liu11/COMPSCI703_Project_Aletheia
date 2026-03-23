# 🛡️ Aletheia: Agentic Financial Compliance Auditor

**Aletheia** (Greek: "Truth") is an advanced AI-driven compliance system designed to audit financial contracts against complex regulatory frameworks. It implements a **Multi-Stage Agentic RAG** architecture, specifically optimized to minimize legal hallucinations and ensure source-grounded accuracy.

---

## 🎯 Project Overview

This project addresses the challenge of **expensive cross-border contract auditing** by leveraging AI agents to automate compliance verification. By using a dual-agent reflection loop, Aletheia significantly reduces legal hallucinations common in traditional RAG systems.

---

## 🧠 Core AGI Concepts

### 1. System 2 Thinking (Slow Reasoning)
Most LLMs operate on "System 1" (fast, intuitive but error-prone). Aletheia forces **System 2 thinking** by breaking the audit into:
1. **Retrieval Stage** - Diverse context gathering via MMR
2. **Analysis Stage** - Junior Auditor initial assessment
3. **Reflection Stage** - Chief Compliance Officer rigorous review

### 2. Agentic Reflection Loop
The system utilizes a **Junior Auditor → Chief Compliance Officer** workflow:
- **Junior Auditor**: Performs initial contract analysis against regulatory basis
- **Chief Compliance Officer**: Reviews the junior's findings, filters hallucinations, and provides verified recommendations

### 3. Diversity-Aware Retrieval (MMR)
To avoid "Information Redundancy," the system uses **Maximal Marginal Relevance (MMR)**:
- Scans a candidate pool of 20 vector segments
- Selects the top 3 most diverse snippets
- Ensures AI considers multiple regulatory angles simultaneously

### 4. Hallucination Reduction
Unlike traditional RAG systems, Aletheia implements **STRICT SOURCE GROUNDING**:
- All findings must be traceable to regulatory documents
- Chief Officer rejects unsupported claims
- Numerical values are cross-verified

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **LLM (Brain)** | Llama 3.2 (via Ollama, local inference) |
| **Vector DB (Memory)** | Pinecone (Serverless, 384 dimensions) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` |
| **Orchestration** | LangChain (Agentic Workflows) |
| **UI** | Streamlit |
| **Language** | Python 3.13+ |

---

## 📁 Project Structure

```
Aletheia/
├── app.py                 # Streamlit Web Application
├── agent_audit.py        # Core Agentic Audit Engine
├── ingest.py             # Data Ingestion Pipeline
├── test_security.py      # Security & Poison Testing ⭐
├── test_brain.py         # LLM Connection Test
├── requirements.txt       # Python Dependencies
├── .env.example          # Environment Variables Template
├── .gitignore           # Git Ignore Rules
├── Dockerfile            # Docker Container ⭐
├── docker-compose.yml    # Docker Compose ⭐
├── .github/
│   └── workflows/
│       ├── test.yml     # CI: Automated Testing ⭐
│       └── docker.yml    # CD: Docker Build ⭐
├── README.md            # This File
├── TECH_DOC.md          # Technical Architecture
├── DATA-library/        # Regulatory Documents
│   └── test.pdf        # Sample Test Data
└── .streamlit/
    └── config.toml      # Streamlit Configuration
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.13+
- [Ollama](https://ollama.ai/) installed with Llama 3.2 model
- A [Pinecone](https://www.pinecone.io/) account (free tier available)

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/estella-liu11/COMPSCI703_Project_Aletheia.git
   cd COMPSCI703_Project_Aletheia
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Ollama**
   ```bash
   # Install Ollama from https://ollama.ai
   ollama pull llama3.2
   ollama serve
   ```

5. **Configure Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your Pinecone API Key
   ```

6. **Ingest Regulatory Documents**
   ```bash
   python ingest.py
   ```

7. **Run the Application**
   ```bash
   streamlit run app.py
   ```

---

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
docker-compose up
```

### Manual Docker Build

```bash
docker build -t aletheia .
docker run -p 8501:8501 aletheia
```

---

## 🧪 Security Testing

Run the comprehensive security test suite:

```bash
python test_security.py
```

### Test Categories

| Test Type | Description |
|-----------|-------------|
| **Prompt Injection** | Tests defense against malicious instruction injection |
| **Data Poisoning** | Tests resilience against corrupted contract data |
| **Hallucination Detection** | Verifies hallucination reduction effectiveness |

---

## 📊 How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    User uploads contract                      │
└─────────────────────────────────┬───────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│              Text Extraction (PDF/DOCX/TXT)                  │
└─────────────────────────────────┬───────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│         MMR Retrieval (20 candidates → 3 diverse)           │
└─────────────────────────────────┬───────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│           Junior Auditor: Initial Analysis                   │
│    "Identify contradictions with regulatory basis"           │
└─────────────────────────────────┬───────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│        Chief Compliance Officer: Reflection Loop             │
│    "Verify sources, reject hallucinations, recommend"        │
└─────────────────────────────────┬───────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│                  Final Audit Report                          │
│    ✓ Source-grounded findings                               │
│    ✓ Specific modification recommendations                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎓 Academic Context

This project was developed for **COMPSCI 703 - Advanced Topics in Computer Science**, focusing on practical implementations of:

- Large Language Model (LLM) applications
- Retrieval-Augmented Generation (RAG)
- Agentic AI architectures
- System 2 Thinking in AI

---

## 📝 License

This project is for academic purposes.

---

## 👤 Author

**Estella Liu**  
COMPSCI 703, 2025

---

## 🙏 Acknowledgments

- LangChain Community for agentic workflow frameworks
- Ollama for local LLM inference
- Pinecone for vector database infrastructure
