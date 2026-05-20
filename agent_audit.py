import os
import re
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

from agent_audit_baseline import vector_store

# Load environment variables
load_dotenv()

# ========================================================
# 1. Initialize Pinecone + LLM
# ========================================================
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store=PineconeVectorStore(
    index_name="alethia-legal-db",
    embedding=embeddings,
    pinecone_api_key="PINECONE_API_KEY",
)

llm = OllamaLLM(model="llama3.2", temperature=0)
#openAI= OllamaLLM(model="OpenAI", temperature=0)

JUNIOR_A_PROMPT = """You are a Risk Hunter Auditor specializing in 
identifying potential VIOLATIONS in financial contracts.

Your mindset: Assume the contract has hidden risks. Your job is to find them.

═══════════════════════════════════
Regulatory Basis:
{context}

Contract to Audit:
{contract_segment}
═══════════════════════════════════

For EACH potential risk you find, output in this exact format:

[Risk #N]
- Severity: HIGH / MEDIUM / LOW
- Issue: <one sentence describing the violation>
- Contract Clause: <quote the relevant contract text>
- Regulatory Source: <quote the exact regulation text from above>
- Reasoning: <why this is a risk, 1-2 sentences>

**IMPORTANT:** Only report findings where you have identified an actual issue.
DO NOT include items that appear compliant — silently skip them.

If you cannot find any risks, output exactly: "No risks identified."
"""

JUNIOR_B_PROMPT = """You are a Compliance Maven Auditor specializing in 
verifying that financial contracts MEET ALL REQUIRED OBLIGATIONS.

Your mindset: Assume the contract may be MISSING required clauses.

═══════════════════════════════════
Regulatory Basis:
{context}

Contract to Audit:
{contract_segment}
═══════════════════════════════════

For EACH potential missing obligation, output in this exact format:

[Gap #N]
- Severity: HIGH / MEDIUM / LOW
- Missing Provision: <one sentence describing what's missing>
- Required by: <quote the exact regulation text that mandates it>
- Consequence: <what happens if this stays missing, 1-2 sentences>

**IMPORTANT:** Only report findings where you have identified an actual gap.
DO NOT include items that appear satisfied — silently skip them.

If all obligations appear satisfied, output exactly: 
"All compliance obligations appear satisfied."
"""

# ========================================================
# 3. Chief Officer prompt (with verification context)
# ========================================================

CHIEF_OFFICER_VERIFICATION_PROMPT = """You are the Chief Compliance Officer 
making the FINAL audit decision. You have INDEPENDENT verification capability — 
beyond the Juniors' findings, you have been given ADDITIONAL regulatory text 
re-retrieved from the database specifically to verify each citation.

═══════════════════════════════════════════════════════════════════
USER'S DECISION: {user_decision}
═══════════════════════════════════════════════════════════════════

🔴 RISK HUNTER (Junior A) findings:
{junior_a_output}

🔵 COMPLIANCE MAVEN (Junior B) findings:
{junior_b_output}

═══════════════════════════════════════════════════════════════════
📚 ORIGINAL CONTEXT (what Juniors saw):
═══════════════════════════════════════════════════════════════════
{original_context}

═══════════════════════════════════════════════════════════════════
🔍 VERIFICATION CONTEXT (independently retrieved by you for each citation):
═══════════════════════════════════════════════════════════════════
{verification_context}

═══════════════════════════════════════════════════════════════════
YOUR TASKS:
═══════════════════════════════════════════════════════════════════

**TASK 1: VERIFY EACH CITATION** (this is your CORE responsibility)
For every regulatory citation in the Junior(s) findings:

1. Check the VERIFICATION CONTEXT above (which I retrieved independently 
   for that specific citation).

2. Classify the citation:
   ✅ VERIFIED: Citation text matches the verification context.
   ⚠️ PARTIAL: Citation references a real source but interpretation may be off.
   ❌ UNVERIFIED: Citation cannot be confirmed in the verification context.

3. For ❌ UNVERIFIED findings: Mark prominently with "❌ [UNVERIFIED CITATION]" 
   in the final report — but STILL include the finding so user knows AI tried to flag it.

**TASK 2: PROCESS BASED ON USER DECISION**
- "ONLY_A" → use only Junior A's verified findings
- "ONLY_B" → use only Junior B's verified findings  
- "BOTH" → merge findings from both, deduplicate similar ones, 
           flag contradictions as "⚠️ AUDITOR DISAGREEMENT"

**TASK 3: GROUP BY SEVERITY**
Order: HIGH → MEDIUM → LOW. 
DO NOT change severity assigned by Juniors.

**TASK 4: RECOMMENDATIONS**
For each VERIFIED finding only:
- Specific contract modification
- Priority: Must-fix / Should-fix / Consider
- Difficulty: Low / Medium / High

═══════════════════════════════════════════════════════════════════
STRICT CONSTRAINTS:
═══════════════════════════════════════════════════════════════════
1. NEVER cite anything outside ORIGINAL CONTEXT or VERIFICATION CONTEXT.
2. DO NOT invent new findings beyond what Juniors reported.
3. Be transparent about UNVERIFIED citations — flag, don't hide.

═══════════════════════════════════════════════════════════════════
OUTPUT FORMAT (follow exactly):
═══════════════════════════════════════════════════════════════════

## Final Audit Report

### 🔴 Critical Issues (HIGH)
[verified findings here]

### 🟡 Important Issues (MEDIUM)
[verified findings here]

### 🟢 Minor Issues (LOW)
[verified findings here]

### ❌ Unverifiable Claims (flagged for user review)
[findings whose citations could NOT be verified]

### 📋 Recommendations
[recommendations for VERIFIED findings only]
"""


# ========================================================
# 4. Helper: extract citations from a Junior's output
# ========================================================
def extract_citations(text):
    """
    Use regex to pull every regulatory citation out of a Junior's output.

    Matches two formats:
    - "Regulatory Source: ..." (used by Junior A)
    - "Required by: ..."       (used by Junior B)
    """
    citations = []

    # Junior A format
    pattern_a = r'Regulatory Source:\s*(.+?)(?=\n-|\n\[|\Z)'
    matches_a = re.findall(pattern_a, text, re.DOTALL)
    citations.extend([m.strip() for m in matches_a])

    # Junior B format
    pattern_b = r'Required by:\s*(.+?)(?=\n-|\n\[|\Z)'
    matches_b = re.findall(pattern_b, text, re.DOTALL)
    citations.extend([m.strip() for m in matches_b])

    # Deduplicate (preserve order)
    seen = set()
    unique = []
    for c in citations:
        # Use first 200 chars as fingerprint to avoid near-duplicates
        fingerprint = c[:200]
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(c)

    return unique


# ========================================================
# 5. Helper: re-retrieve each citation independently
# ========================================================
def re_retrieve_citations(citations, k=3):
    """
    For each citation, query Pinecone independently and return the
    merged verification context.
    """
    all_verification_chunks = []

    print(f"🔍 Re-retrieving {len(citations)} citations for verification...")

    for i, citation in enumerate(citations, 1):
        # Truncate the query (Pinecone discourages very long queries)
        query = citation[:500]

        # Retrieve top-k most relevant chunks
        docs = vector_store.similarity_search(query, k=k)

        # Format each retrieval result
        chunk_text = f"\n--- Verification for Citation #{i} ---\n"
        chunk_text += f"Original citation: {citation[:200]}...\n\n"
        chunk_text += "Retrieved evidence:\n"
        for j, doc in enumerate(docs, 1):
            chunk_text += f"[Evidence {j}] {doc.page_content[:400]}\n\n"

        all_verification_chunks.append(chunk_text)
        print(f"   ✓ Citation {i}/{len(citations)} re-retrieved")

    return "\n".join(all_verification_chunks)


# ========================================================
# 6. Entry point 1: run both Juniors
# ========================================================
def run_dual_juniors(contract_segment):
    """
    Run both Junior agents and return their outputs plus the
    initial retrieval context.
    """
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20}
    )
    docs = retriever.invoke(contract_segment)
    context = "\n\n---\n\n".join([d.page_content for d in docs])

    print(f"✅ Initial retrieval: {len(docs)} chunks")

    print("🔴 Running Risk Hunter (Junior A)...")
    junior_a_output = llm.invoke(
        JUNIOR_A_PROMPT.format(context=context, contract_segment=contract_segment)
    )
    print("   ✓ Done")

    print("🔵 Running Compliance Maven (Junior B)...")
    junior_b_output = llm.invoke(
        JUNIOR_B_PROMPT.format(context=context, contract_segment=contract_segment)
    )
    print("   ✓ Done")

    return {
        "junior_a_output": junior_a_output,
        "junior_b_output": junior_b_output,
        "context": context,
    }


# ========================================================
# 7. Entry point 2: run the Chief Officer (with verification)
# ========================================================
def run_chief_officer(junior_a_output, junior_b_output, context, user_decision):
    """
    The Chief Officer now performs 3 steps:
    1. Extract citations from the Junior(s) output
    2. Re-query Pinecone independently for each citation
    3. Combine verification context + Junior output → final report
    """

    # --- Handle NEITHER ---
    if user_decision == "NEITHER":
        return """## ⚠️ Audit Escalated to Human Review

The user has determined that neither AI auditor's analysis is sufficient.
Recommend immediate review by a senior compliance officer.

### 📜 Audit Trail
- User decision: NEITHER (escalated)
"""

    # --- Decide which Junior(s) to verify based on user decision ---
    texts_to_verify = []
    if user_decision in ("ONLY_A", "BOTH"):
        texts_to_verify.append(junior_a_output)
    if user_decision in ("ONLY_B", "BOTH"):
        texts_to_verify.append(junior_b_output)

    # --- Step 1: extract every citation ---
    print("\n🔍 PHASE: Citation Verification")
    print("-" * 50)
    all_citations = []
    for text in texts_to_verify:
        all_citations.extend(extract_citations(text))

    print(f"📌 Extracted {len(all_citations)} unique citations")

    # --- Step 2: re-retrieve each citation ---
    if all_citations:
        verification_context = re_retrieve_citations(all_citations)
    else:
        verification_context = "(No citations to verify — Junior(s) found no issues.)"

    # --- Step 3: Chief Officer synthesises everything ---
    print("\n👨‍⚖️ Running Chief Officer (with verification context)...")

    # Mask out Junior outputs the user did not select
    final_a = junior_a_output if user_decision in ("ONLY_A", "BOTH") else "[NOT SELECTED]"
    final_b = junior_b_output if user_decision in ("ONLY_B", "BOTH") else "[NOT SELECTED]"

    prompt = CHIEF_OFFICER_VERIFICATION_PROMPT.format(
        user_decision=user_decision,
        junior_a_output=final_a,
        junior_b_output=final_b,
        original_context=context,
        verification_context=verification_context,
    )

    final_report = llm.invoke(prompt)
    print("   ✓ Chief Officer complete")

    return final_report


# ========================================================
# 8. Backward-compatible single-call interface
# ========================================================
def run_audit(contract_segment):
    """One-shot wrapper kept for backward compatibility."""
    juniors_result = run_dual_juniors(contract_segment)
    return run_chief_officer(
        junior_a_output=juniors_result["junior_a_output"],
        junior_b_output=juniors_result["junior_b_output"],
        context=juniors_result["context"],
        user_decision="BOTH",
    )


# ========================================================
# 9. Quick Q&A function (single-turn mode)
# ========================================================

QUICK_QA_PROMPT = """You are a knowledgeable financial compliance 
advisor specializing in NZ-US cross-border tax regulations.

A user has asked the following question:
═══════════════════════════════════
Q: {question}
═══════════════════════════════════

You have been given the following regulatory context retrieved from 
the database:
═══════════════════════════════════
{context}
═══════════════════════════════════

**Your task:** Answer the user's question clearly and concisely.

**Strict rules:**
1. ✅ Base your answer ONLY on the regulatory context above.
2. ✅ If the answer is YES or NO, state it clearly first, then explain.
3. ✅ Cite specific regulations when possible (e.g., "Article 12 of the treaty states...").
4. ❌ DO NOT invent regulations, articles, or numbers not in the context.
5. ❌ If the context does NOT contain enough information to answer, 
   say: "I cannot find sufficient information in the available 
   regulations to answer this confidently. Please consult a tax 
   professional or refer to the official source documents."

**Output format:**

**Short Answer:** [Yes / No / It depends — one sentence]

**Explanation:** [2-4 sentences with reasoning]

**Source(s):** [Quote or cite the specific regulation(s) you used]

**Confidence:** [HIGH / MEDIUM / LOW — based on how well the context covers this question]
"""


def run_quick_qa(question):
    """
    Single-turn quick Q&A — a stripped-down RAG.
    No verification, no multi-agent — just retrieve + one LLM call.

    Args:
        question: the user's natural-language question

    Returns:
        The LLM answer (markdown formatted).
    """
    # 1. MMR retrieval (the user's question is the query)
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20}
    )
    docs = retriever.invoke(question)
    context = "\n\n---\n\n".join([d.page_content for d in docs])

    # 2. One LLM call
    prompt = QUICK_QA_PROMPT.format(question=question, context=context)
    answer = llm.invoke(prompt)

    return answer
# ========================================================
# 10. CLI test entry point
# ========================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🛡️  Aletheia 2.0 — Multi-Agent + Self-Verifying Chief")
    print("=" * 70)

    test_contract = """
    Service Agreement between NZ Company Ltd (New Zealand) and 
    US Corporation Inc (California, USA).

    Clause 1: Payment. NZ Company shall pay US Corporation USD 50,000 
    quarterly for software development services.

    Clause 2: Withholding Tax. NZ Company shall withhold 5% tax on all 
    payments to US Corporation.

    Clause 3: Dispute Resolution. Any disputes shall be settled in 
    California courts under California law.

    Clause 4: Term. This agreement is effective immediately and shall 
    continue indefinitely.
    """

    # Phase 1
    print("\n📊 PHASE 1: Dual Junior Auditors")
    print("-" * 70)
    result = run_dual_juniors(test_contract)

    print("\n" + "=" * 70)
    print("🔴 JUNIOR A:")
    print("=" * 70)
    print(result["junior_a_output"])

    print("\n" + "=" * 70)
    print("🔵 JUNIOR B:")
    print("=" * 70)
    print(result["junior_b_output"])

    # Phase 2
    print("\n" + "=" * 70)
    print("👨‍⚖️ PHASE 2: Self-Verifying Chief Officer (user: BOTH)")
    print("=" * 70)
    final = run_chief_officer(
        junior_a_output=result["junior_a_output"],
        junior_b_output=result["junior_b_output"],
        context=result["context"],
        user_decision="BOTH",
    )

    print("\n" + "=" * 70)
    print("📜 FINAL AUDIT REPORT:")
    print("=" * 70)
    print(final)

