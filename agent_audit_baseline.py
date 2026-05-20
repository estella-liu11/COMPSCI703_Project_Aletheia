"""
Baseline version: the old Aletheia pipeline
(single Junior + single Chief, no verification step).
Used only for evaluation comparison — not imported by app.py.
"""

import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(
    index_name="aletheia-legal-db",
    embedding=embeddings,
    pinecone_api_key=os.getenv("PINECONE_API_KEY"),
)
llm = OllamaLLM(model="llama3.2", temperature=0)


def run_baseline_audit(contract_segment):
    """
    Old Aletheia pipeline:
    - Single Junior Auditor (finds issues)
    - Single Chief Officer (only synthesises, no verification)
    - No multi-agent, no self-verification
    """

    # 1. Retrieve top-3 (the old parameters)
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 20}
    )
    docs = retriever.invoke(contract_segment)
    context = "\n".join([d.page_content for d in docs])

    # 2. Single Junior (the old simple prompt)
    prompt_1 = f"""You are a Senior Financial Compliance Auditor.
Regulatory Basis: {context}
Contract Content to Audit: {contract_segment}

Task: Identify any contradictions between the contract and the 
regulatory basis. For each issue, cite the specific regulation.

If no risks are found, reply with "No Risks Identified"."""

    initial_opinion = llm.invoke(prompt_1)

    # 3. Single Chief (old version: just a simple review, no re-retrieval)
    prompt_2 = f"""You are the Chief Compliance Officer. Review the 
following audit and provide a final report:

Initial Audit: {initial_opinion}
Regulatory Basis: {context}

Tasks:
1. STRICT SOURCE GROUNDING: Only use facts in the Regulatory Basis above.
2. Verify critical numbers and citations.
3. Provide final findings and recommendations."""

    final_audit = llm.invoke(prompt_2)
    return final_audit