# 🛡️ Aletheia: Agentic Financial Compliance Auditor

**Aletheia** (Ancient Greek for "Truth") is an advanced AI-driven compliance system designed to audit financial contracts against complex regulatory frameworks. It moves beyond simple chatbots by implementing a **Multi-Stage Agentic RAG** architecture, specifically optimized to minimize legal hallucinations and ensure source-grounded accuracy.

---

## 🧠 Core AGI Concepts & Architecture

This project is a practical implementation of several high-level AI concepts:

### 1. System 2 Thinking (Slow Reasoning)
Most LLMs operate on "System 1" (fast, intuitive but error-prone). Aletheia forces **System 2 thinking** by breaking the audit into retrieval, analysis, and reflection stages, allowing for deep logical verification.

### 2. Agentic Reflection Loop
The system utilizes a **Junior Auditor → Chief Compliance Officer** workflow. The "Chief" agent audits the "Junior" agent's initial findings, performing **Self-Correction** and **Rigor Review** to filter out hallucinations.

### 3. Diversity-Aware Retrieval (MMR)
To avoid "Information Redundancy," the system uses **Maximal Marginal Relevance (MMR)**. It scans a candidate pool of 20 vector segments and精选 out the top 3 most diverse snippets, ensuring the AI considers multiple regulatory angles (e.g., interest rates, late fees, and grace periods) simultaneously.

---

## 🛠️ Technical Stack

| Layer | Technology |
| :--- | :--- |
| **Brain (LLM)** | Llama 3.2 (Running locally via Ollama) |
| **Memory (Vector DB)** | Pinecone (Serverless Vector Index) |
| **Orchestration** | LangChain (Agentic Workflows) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` (384 Dimensions) |
| **UI** | Streamlit |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.13
- [Ollama](https://ollama.ai/) installed with the Llama 3.2 model (`ollama run llama3.2`)
- A Pinecone API Key (Index dimension should be **384**)

### Installation
1. **Clone the Repo**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Aletheia-Auditor.git](https://github.com/YOUR_USERNAME/Aletheia-Auditor.git)
   cd Aletheia-Auditor