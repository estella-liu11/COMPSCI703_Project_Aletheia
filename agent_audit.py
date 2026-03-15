import os
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# 1. Initialize the "Legal Brain" (Pinecone)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(
    index_name="aletheia-legal-db",
    embedding=embeddings,
    pinecone_api_key="pcsk_vwEEW_Nm8c5qsyQ4dHN1itWQfepSATHBk2gnoSMyGzpvQ5abry7j7hTCyS9gGWgNkJshw"
)

# 2. Initialize the LLM (Llama 3.2)
llm = OllamaLLM(model="llama3.2", temperature=0)


def run_audit(contract_segment):
    """
    Executes a dual-stage agentic audit process with MMR Retrieval.
    """

    # --- Step 1: Diverse Retrieval (MMR) ---
    # We turn the vector store into a retriever using Maximal Marginal Relevance.
    # fetch_k=20: Look at the top 20 candidates globally.
    # k=3: Select the 3 most relevant but DIVERSE chunks to avoid redundancy.
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 3, 'fetch_k': 20}
    )

    docs = retriever.invoke(contract_segment)
    context = "\n".join([d.page_content for d in docs])

    # --- Step 2: Initial Reasoning (Junior Auditor) ---
    prompt_1 = f"""You are a Senior Financial Compliance Auditor.
    Regulatory Basis: {context}
    Contract Content to Audit: {contract_segment}

    Task: Identify any contradictions between the contract and the regulatory basis. 
    If no risks are found, reply with "No Risks Identified"."""

    initial_opinion = llm.invoke(prompt_1)

    # --- Step 3: Reflection Loop (Chief Compliance Officer) ---
    # Added "STRICT SOURCE GROUNDING" to minimize the hallucinations we discussed earlier.
    prompt_2 = f"""You are the Chief Compliance Officer. Please audit the following initial draft for rigor:
    Initial Draft: {initial_opinion}
    Regulatory Basis: {context}

    Your Tasks:
    1. STRICT SOURCE GROUNDING: Only use facts provided in the Regulatory Basis above.
    2. Verify if the draft missed any critical numbers (e.g., interest rate caps, fee limits).
    3. If the initial draft is incorrect, provide a corrected version.
    4. Provide specific recommendations for contract modification."""

    final_audit = llm.invoke(prompt_2)
    return final_audit