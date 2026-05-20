import streamlit as st
from pypdf import PdfReader
from docx import Document
from agent_audit import run_dual_juniors, run_chief_officer
# Quick Q&A helper (single-turn mode)
from agent_audit import run_quick_qa

# ═══════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════
st.set_page_config(
    page_title="Aletheia | HITL Compliance Auditor",
    page_icon="🛡️",
    layout="wide"
)

# ═══════════════════════════════════════════════════════
# Session State init
# ═══════════════════════════════════════════════════════
if "stage" not in st.session_state:
    st.session_state.stage = "upload"
if "junior_results" not in st.session_state:
    st.session_state.junior_results = None
if "final_report" not in st.session_state:
    st.session_state.final_report = None
if "contract_text" not in st.session_state:
    st.session_state.contract_text = None
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []  # Q&A history list (single-turn, but kept for display)


# ═══════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════
def extract_text(uploaded_file):
    """Extract text from the uploaded file."""
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext == "pdf":
        reader = PdfReader(uploaded_file)
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    elif ext == "docx":
        doc = Document(uploaded_file)
        return "\n".join(p.text for p in doc.paragraphs)
    elif ext == "txt":
        return uploaded_file.read().decode("utf-8")
    return None


def reset_audit_session():
    """Reset the audit-flow session state."""
    st.session_state.stage = "upload"
    st.session_state.junior_results = None
    st.session_state.final_report = None
    st.session_state.contract_text = None


# ═══════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════
st.title("🛡️ Aletheia: AI Compliance Assistant")
st.markdown("*Multi-Agent RAG with Human-in-the-Loop verification*")

# ═══════════════════════════════════════════════════════
# Sidebar: system status
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ System")
    st.success("🧠 Llama 3.2")
    st.success("📚 Pinecone")
    st.divider()
    st.caption("**Audit Mode:** Multi-agent + HITL for contract review")
    st.caption("**Q&A Mode:** Quick regulatory questions")

# ═══════════════════════════════════════════════════════
# Two tabs: Audit | Quick Q&A
# ═══════════════════════════════════════════════════════
tab_audit, tab_qa = st.tabs(["📤 Contract Audit", "💬 Quick Q&A"])

# ───────────────────────────────────────────────────────
# Tab 1: Contract Audit (existing flow, unchanged)
# ───────────────────────────────────────────────────────
with tab_audit:
    # Stage 1: Upload
    if st.session_state.stage == "upload":
        st.subheader("📤 Upload a Financial Contract")

        uploaded_file = st.file_uploader(
            "Supported: PDF, DOCX, TXT",
            type=["pdf", "docx", "txt"],
            key="audit_uploader"
        )

        if uploaded_file:
            st.success(f"✅ Loaded: **{uploaded_file.name}**")
            contract_text = extract_text(uploaded_file)

            if contract_text:
                with st.expander("📄 Preview"):
                    st.text(contract_text[:500] + ("..." if len(contract_text) > 500 else ""))

                if st.button("🚀 Run Multi-Agent Audit", type="primary"):
                    st.session_state.contract_text = contract_text
                    with st.spinner("⏳ Dual auditors analyzing... (~60s)"):
                        result = run_dual_juniors(contract_text[:3000])
                        st.session_state.junior_results = result
                        st.session_state.stage = "review"
                        st.rerun()

    # Stage 2: Review (HITL)
    elif st.session_state.stage == "review":
        st.subheader("🧑‍⚖️ Adjudicate the Auditors")
        st.info("Two AI auditors analyzed your contract from different angles. Choose which to accept.")

        juniors = st.session_state.junior_results
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("### 🔴 Risk Hunter")
            st.caption("*Finds violations*")
            with st.container(border=True):
                st.markdown(juniors["junior_a_output"])

        with col_b:
            st.markdown("### 🔵 Compliance Maven")
            st.caption("*Finds gaps*")
            with st.container(border=True):
                st.markdown(juniors["junior_b_output"])

        st.divider()

        decision = st.radio(
            "Your decision:",
            options=["BOTH", "ONLY_A", "ONLY_B", "NEITHER"],
            format_func=lambda x: {
                "BOTH": "✅ Accept BOTH (recommended)",
                "ONLY_A": "🔴 Only Risk Hunter",
                "ONLY_B": "🔵 Only Compliance Maven",
                "NEITHER": "⚠️ Escalate to human review",
            }[x],
            index=0
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("👨‍⚖️ Continue to Chief Officer", type="primary"):
                with st.spinner("⏳ Chief verifying citations... (~80s)"):
                    final = run_chief_officer(
                        junior_a_output=juniors["junior_a_output"],
                        junior_b_output=juniors["junior_b_output"],
                        context=juniors["context"],
                        user_decision=decision,
                    )
                    st.session_state.final_report = final
                    st.session_state.user_decision = decision
                    st.session_state.stage = "final"
                    st.rerun()
        with col2:
            if st.button("⬅️ Back"):
                reset_audit_session()
                st.rerun()

    # Stage 3: Final Report
    elif st.session_state.stage == "final":
        st.subheader("📜 Final Audit Report")
        st.success(f"✅ User decision: **{st.session_state.user_decision}**")

        with st.container(border=True):
            st.markdown(st.session_state.final_report)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="💾 Download Report",
                data=st.session_state.final_report,
                file_name="Aletheia_Audit_Report.txt",
                mime="text/plain",
                type="primary"
            )
        with col2:
            if st.button("🔄 New Audit"):
                reset_audit_session()
                st.rerun()

# ───────────────────────────────────────────────────────
# Tab 2: Quick Q&A
# ───────────────────────────────────────────────────────
with tab_qa:
    st.subheader("💬 Ask a Quick Regulatory Question")
    st.caption("Get a fast answer grounded in NZ-US tax regulations. No file upload needed.")

    # Example prompts so the user knows what kinds of questions work
    with st.expander("💡 Example questions"):
        st.markdown("""
        - Is the seafood import tariff from NZ to California 30%?
        - What is the withholding tax rate for royalties under NZ-US treaty?
        - Does NZ-US tax treaty cover digital services?
        - What constitutes a permanent establishment under the treaty?
        """)

    # Input box
    question = st.text_area(
        "Your question:",
        placeholder="e.g., What is the withholding tax rate for cross-border services?",
        height=80,
        key="qa_input"
    )

    # Submit button
    if st.button("🔍 Get Answer", type="primary", key="qa_submit"):
        if not question.strip():
            st.warning("⚠️ Please enter a question.")
        else:
            with st.spinner("⏳ Searching regulations and reasoning..."):
                answer = run_quick_qa(question)
                # Store this Q&A in history (single-turn mode only shows latest)
                st.session_state.qa_history.append({
                    "question": question,
                    "answer": answer
                })

    # Display the most recent answer (single-turn mode: no accumulated context)
    if st.session_state.qa_history:
        st.divider()
        latest = st.session_state.qa_history[-1]

        st.markdown("### 📋 Answer")
        with st.container(border=True):
            st.markdown(f"**Q:** {latest['question']}")
            st.markdown("---")
            st.markdown(latest["answer"])

        # If there are earlier questions, show them collapsed
        if len(st.session_state.qa_history) > 1:
            with st.expander(f"📜 Previous questions ({len(st.session_state.qa_history) - 1})"):
                for i, qa in enumerate(reversed(st.session_state.qa_history[:-1]), 1):
                    st.markdown(f"**Q{len(st.session_state.qa_history) - i}:** {qa['question']}")
                    st.markdown(qa["answer"])
                    st.divider()

        # Clear-history button
        if st.button("🗑️ Clear Q&A history", key="qa_clear"):
            st.session_state.qa_history = []
            st.rerun()