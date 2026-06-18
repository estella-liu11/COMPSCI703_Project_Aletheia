"""
Baseline version: the old Aletheia pipeline
(single Junior + single Chief, no verification step).
Used only for evaluation comparison — not imported by app.py.
"""

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from llm_factory import get_llm, invoke_text
from vector_store_factory import get_vector_store
import config

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
# Honour VECTOR_DB_BACKEND so the baseline reads from the same store
# as the new pipeline — required for apples-to-apples evaluation.
vector_store = get_vector_store(embeddings)
# LLM client resolved lazily so this baseline honours the same LLM_BACKEND
# switch as agent_audit.py — keeps evaluation comparisons apples-to-apples.


def run_baseline_audit(contract_segment):
    """
    Old Aletheia pipeline:
    - Single Junior Auditor (finds issues)
    - Single Chief Officer (only synthesises, no verification)
    - No multi-agent, no self-verification
    """

    # 1. Retrieve top-k (the old parameters — intentionally smaller than
    # the new pipeline so the ablation is non-trivial).
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":        config.BASELINE_RETRIEVAL_K,
            "fetch_k":  config.BASELINE_RETRIEVAL_FETCH_K,
        },
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

    llm = get_llm()
    initial_opinion = invoke_text(llm, prompt_1)

    # 3. Single Chief (old version: just a simple review, no re-retrieval)
    prompt_2 = f"""You are the Chief Compliance Officer. Review the 
following audit and provide a final report:

Initial Audit: {initial_opinion}
Regulatory Basis: {context}

Tasks:
1. STRICT SOURCE GROUNDING: Only use facts in the Regulatory Basis above.
2. Verify critical numbers and citations.
3. Provide final findings and recommendations."""

    final_audit = invoke_text(llm, prompt_2)
    return final_audit