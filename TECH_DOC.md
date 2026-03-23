# Technical Architecture 技术架构

## Overview

Aletheia implements a **Multi-Stage Agentic RAG** architecture designed specifically for financial contract compliance auditing. This document details the technical implementation.

---

## System Architecture

### Layer 1: Document Processing

```
Input: PDF/DOCX/TXT files
    │
    ▼
┌─────────────────────────────────┐
│   Document Loader (pypdf,       │
│   python-docx)                  │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│   Text Splitter (LangChain)     │
│   - chunk_size: 500 chars       │
│   - chunk_overlap: 50 chars     │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│   Embedding Model               │
│   HuggingFace all-MiniLM-L6-v2 │
│   Output: 384-dimensional       │
└─────────────────────────────────┘
    │
    ▼
Pinecone Vector Database
```

### Layer 2: Retrieval System

The retrieval system uses **Maximal Marginal Relevance (MMR)** to ensure diverse context selection:

```python
# Configuration
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 3, 'fetch_k': 20}
)
```

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `fetch_k` | 20 | Initial candidate pool size |
| `k` | 3 | Final selected documents |
| `lambda_mult` | 0.5 | Balance relevance vs diversity |

### Layer 3: Dual-Agent Audit Engine

```
Contract Text
    │
    ▼
┌─────────────────────────────────────────────────────┐
│              Junior Auditor Agent                     │
│  - Role: Initial contract analysis                  │
│  - Task: Identify contradictions with regulatory     │
│          basis                                       │
│  - Model: Llama 3.2 (temperature=0)                │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│          Chief Compliance Officer Agent               │
│  - Role: Reflection and verification                │
│  - Task: Filter hallucinations, verify sources,     │
│          provide recommendations                     │
│  - Model: Llama 3.2 (temperature=0)                │
│  - Constraint: STRICT SOURCE GROUNDING              │
└─────────────────────────────────────────────────────┘
    │
    ▼
Final Audit Report
```

---

## Component Specifications

### Agent Audit Engine (`agent_audit.py`)

| Component | Specification |
|-----------|---------------|
| LLM | Ollama Llama 3.2 |
| Temperature | 0 (deterministic) |
| Context Length | First 3000 characters of contract |
| Vector Store | Pinecone (aletheia-legal-db) |
| Embedding Model | all-MiniLM-L6-v2 (384 dims) |
| Retrieval | MMR (k=3, fetch_k=20) |

### Data Ingestion Pipeline (`ingest.py`)

| Parameter | Value |
|-----------|-------|
| Chunk Size | 500 characters |
| Chunk Overlap | 50 characters |
| Text Splitter | RecursiveCharacterTextSplitter |
| Source Directory | DATA-library/ |
| Supported Formats | PDF, DOCX |

---

## Security Considerations

### Hallucination Prevention

1. **Source Grounding**: All claims must be traceable to regulatory documents
2. **Numerical Verification**: Interest rates, fee limits are cross-checked
3. **Reflection Loop**: Chief Officer reviews and corrects Junior's findings

### Data Protection

- API keys stored in environment variables (`.env`)
- No sensitive data in code repository
- Streamlit Secrets for deployment

---

## Deployment Options

### Option 1: Local Development

```bash
# Start Ollama
ollama serve

# Run Streamlit
streamlit run app.py
```

### Option 2: Docker

```bash
docker-compose up
```

### Option 3: Streamlit Cloud

- Free hosting for Streamlit apps
- Supports Secrets for API key management
- Automatic HTTPS

---

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Document Processing | ~1-2 seconds per page |
| Retrieval Latency | ~200-500ms |
| Audit Generation | ~10-30 seconds |
| Total Pipeline | ~15-45 seconds |

---

## Future Improvements

1. **Extended Context**: Handle longer contracts with chunking
2. **Multi-language Support**: Process contracts in various languages
3. **Fine-tuned Model**: Train on financial/legal domain data
4. **Real-time Updates**: Dynamic regulatory database updates
