import os
import re

import streamlit as st
from pypdf import PdfReader
from docx import Document
from agent_audit import run_dual_juniors, run_chief_officer
# Quick Q&A helper (single-turn mode)
from agent_audit import run_quick_qa
from llm_factory import (
    OLLAMA,
    DEEPSEEK,
    describe_active_backend,
    get_backend_name,
)
from vector_store_factory import (
    describe_active_backend as describe_active_vector_backend,
)
import db


# ═══════════════════════════════════════════════════════
# Structured renderer for Chief Officer reports.
#
# Old behaviour: `st.markdown(report)` rendered the raw report blob,
# which meant every finding looked the same regardless of severity
# and an "UNVERIFIED" tag jumped out as a scary red ❌. Peer-review
# group 15 flagged that this caused non-technical compliance officers
# to over-react to low-priority findings.
#
# New behaviour: parse `[N] [SEVERITY] [TRUST] "..."` blocks out of
# the report. Map *severity* (not trust) to colour:
#     HIGH → st.error (red)
#     MED  → st.warning (yellow)
#     LOW  → st.info (blue)
# Show trust as a small inline badge, with softened labels:
#     Verified | Needs Review | Source Not Found
# Falls back to plain markdown if the report doesn't match the
# structured format (e.g. the "Contract looks compliant" single-line
# case, or escalation messages).
# ═══════════════════════════════════════════════════════

_FINDING_BOUNDARY_RE = re.compile(r'^\s*\[(\d+)\]\s*', re.MULTILINE)
_FIRST_LINE_RE = re.compile(
    r'\[(HIGH|MED|LOW)\]\s*\[(Verified|Needs Review|Source Not Found)\]\s*"([^"]+)"',
    re.IGNORECASE,
)
_FIX_RE = re.compile(r'Fix:\s*([^\n]+)', re.IGNORECASE)
_SOURCE_RE = re.compile(r'Source:\s*([^\n]+)', re.IGNORECASE)

_SEVERITY_RENDER = {
    "HIGH": (lambda body: st.error(body), "🔴"),
    "MED":  (lambda body: st.warning(body), "🟡"),
    "LOW":  (lambda body: st.info(body), "🔵"),
}

_TRUST_BADGE = {
    "Verified":         "✅ Verified",
    "Needs Review":     "🟠 Needs Review",
    "Source Not Found": "❔ Source Not Found",
}


def render_audit_report(report):
    """Render a Chief Officer audit report with severity-based styling.

    Args:
        report: the raw markdown-style report string from
            ``run_chief_officer`` (or from saved history).
    """
    if not report or not report.strip():
        st.info("No report available.")
        return

    blocks = list(_FINDING_BOUNDARY_RE.finditer(report))
    if not blocks:
        # Compliant case / escalation / anything without [N] structure.
        st.markdown(report)
        return

    rendered_any = False
    for i, m in enumerate(blocks):
        num = m.group(1)
        body_start = m.end()
        body_end = blocks[i + 1].start() if i + 1 < len(blocks) else len(report)
        body = report[body_start:body_end].strip()

        first = _FIRST_LINE_RE.search(body)
        if not first:
            # Block doesn't match the new structured format; let it
            # fall through to plain markdown rendering at the bottom.
            continue

        severity, trust, clause = first.groups()
        severity = severity.upper()
        # Title-case trust to match _TRUST_BADGE keys regardless of LLM casing.
        trust = " ".join(w.capitalize() for w in trust.split())
        # 'Source not Found' → 'Source Not Found' (match dict key)
        if trust.lower() == "source not found":
            trust = "Source Not Found"
        elif trust.lower() == "needs review":
            trust = "Needs Review"

        fix = _FIX_RE.search(body)
        source = _SOURCE_RE.search(body)

        render_fn, prefix = _SEVERITY_RENDER.get(
            severity, (lambda body: st.info(body), "🔵")
        )
        lines = [
            f"**{prefix} Finding #{num}** · {severity} severity · "
            f"{_TRUST_BADGE.get(trust, trust)}",
            "",
            f"> {clause.strip()}",
        ]
        if fix:
            lines += ["", f"**Fix:** {fix.group(1).strip()}"]
        if source:
            lines += ["", f"**Source:** {source.group(1).strip()}"]
        render_fn("\n".join(lines))
        rendered_any = True

    if not rendered_any:
        # All blocks were unparseable — show the original verbatim.
        st.markdown(report)
        return

    # Tail material after the last finding (e.g. a Recommendations note).
    tail = report[blocks[-1].end():]
    tail = re.sub(r'^\s*"[^"]*"\s*\n?', '', tail)  # drop the closing clause we already showed
    tail = re.sub(r'^.*?(Fix:|Source:).*$', '', tail, flags=re.MULTILINE)
    tail = tail.strip()
    if tail:
        st.markdown(tail)

    # Trust-label explainer — addresses Group 15's "trust paradox".
    with st.expander("ℹ️ What do the trust labels mean?"):
        st.markdown(
            "- **✅ Verified** — The cited regulation was independently "
            "confirmed by re-querying the regulatory database.\n"
            "- **🟠 Needs Review** — The citation partially matches a "
            "related regulation. Worth double-checking, but not necessarily wrong.\n"
            "- **❔ Source Not Found** — The AI could not locate the cited "
            "regulation in its database. **This does NOT mean the contract "
            "is non-compliant** — it only means the AI couldn't verify "
            "this specific citation. Confirm with a compliance professional."
        )

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
if "user" not in st.session_state:
    st.session_state.user = None       # dict from Supabase auth, or None
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "sign_in"   # "sign_in" or "sign_up"
if "stage" not in st.session_state:
    st.session_state.stage = "upload"
if "junior_results" not in st.session_state:
    st.session_state.junior_results = None
if "final_report" not in st.session_state:
    st.session_state.final_report = None
if "contract_text" not in st.session_state:
    st.session_state.contract_text = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "audit_saved" not in st.session_state:
    st.session_state.audit_saved = False
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []   # single-turn Q&A history (in-memory only)


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
    st.session_state.uploaded_file_name = None
    st.session_state.audit_saved = False


# ═══════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════
st.title("🛡️ Aletheia: AI Compliance Assistant")
st.markdown("*Multi-Agent RAG with Human-in-the-Loop verification*")
st.caption("Aletheia is AI and can make mistakes. Please double-check responses.")


# ═══════════════════════════════════════════════════════
# Auth gate — must sign in before using the app
# ═══════════════════════════════════════════════════════
def render_auth_form():
    """Show sign-in / sign-up form. Returns early if not authenticated."""
    st.subheader("🔐 Welcome to Aletheia")

    mode = st.radio(
        "Choose:",
        options=["sign_in", "sign_up"],
        format_func=lambda x: "Sign in" if x == "sign_in" else "Create account",
        horizontal=True,
        key="auth_mode_radio",
        index=0 if st.session_state.auth_mode == "sign_in" else 1,
    )
    st.session_state.auth_mode = mode

    with st.form("auth_form", clear_on_submit=False):
        email = st.text_input("Email", placeholder="you@example.com")
        password = st.text_input("Password", type="password",
                                 placeholder="At least 6 characters")
        display_name = None
        if mode == "sign_up":
            display_name = st.text_input("Display name",
                                         placeholder="What should we call you?")
        submitted = st.form_submit_button(
            "Sign in" if mode == "sign_in" else "Create account",
            type="primary",
            use_container_width=True,
        )

        if submitted:
            if not email or not password:
                st.error("Email and password are required.")
                return
            if mode == "sign_up" and not (display_name and display_name.strip()):
                st.error("Display name is required to sign up.")
                return
            try:
                if mode == "sign_up":
                    user = db.sign_up(email, password, display_name.strip())
                else:
                    user = db.sign_in(email, password)
                st.session_state.user = user
                st.rerun()
            except Exception as e:
                st.error(f"❌ {e}")


if st.session_state.user is None:
    render_auth_form()
    st.stop()   # everything below runs only when signed in


# ═══════════════════════════════════════════════════════
# Sidebar: signed-in user + system status
# ═══════════════════════════════════════════════════════
with st.sidebar:
    user = st.session_state.user
    name = db.display_name_of(user)
    st.markdown(f"### 👋 Hi, **{name}**")
    st.caption(user.get("email", ""))
    if st.button("Sign out", use_container_width=True):
        db.sign_out()
        st.session_state.user = None
        reset_audit_session()
        st.rerun()

    st.divider()
    st.header("⚙️ System")

    # ── LLM backend switcher ──────────────────────────────
    # Writing to os.environ here is enough: agent_audit.py resolves the LLM
    # via get_llm() on every call, and llm_factory caches per backend so a
    # switch is essentially free after the first invoke.
    backend_labels = {
        OLLAMA: "🦙 Llama (Ollama, local)",
        DEEPSEEK: "🐋 DeepSeek (cloud API)",
    }
    backend_options = [OLLAMA, DEEPSEEK]
    current_backend = get_backend_name()
    chosen = st.radio(
        "LLM backend",
        options=backend_options,
        format_func=lambda b: backend_labels[b],
        index=backend_options.index(current_backend),
        key="llm_backend_radio",
    )
    if chosen != current_backend:
        os.environ["LLM_BACKEND"] = chosen
        st.rerun()
    else:
        # Persist the chosen value so describe_active_backend() reflects it
        # even before the first audit runs.
        os.environ["LLM_BACKEND"] = chosen

    if chosen == DEEPSEEK and not os.getenv("DEEPSEEK_API_KEY"):
        st.warning(
            "DeepSeek selected but `DEEPSEEK_API_KEY` is not set in `.env` — "
            "audits will fail until you add it."
        )

    st.success(describe_active_backend())
    st.success(describe_active_vector_backend())
    st.success("🗄️ Supabase")
    st.divider()
    st.caption("**Audit Mode:** Multi-agent + HITL for contract review")
    st.caption("**Q&A Mode:** Quick regulatory questions")
    st.caption("**My History:** your past audits")


# ═══════════════════════════════════════════════════════
# Three tabs: Audit | Quick Q&A | My History
# ═══════════════════════════════════════════════════════
tab_audit, tab_qa, tab_history = st.tabs(
    ["📤 Contract Audit", "💬 Quick Q&A", "📜 My History"]
)

# ───────────────────────────────────────────────────────
# Tab 1: Contract Audit
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
                    st.session_state.uploaded_file_name = uploaded_file.name
                    st.session_state.audit_saved = False
                    with st.spinner("⏳ Dual auditors analyzing... (~90s)"):
                        result = run_dual_juniors(contract_text[:1500])
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
            options=["BOTH", "ONLY_A", "ONLY_B"],
            format_func=lambda x: {
                "BOTH": "✅ Accept BOTH (recommended)",
                "ONLY_A": "🔴 Only Risk Hunter",
                "ONLY_B": "🔵 Only Compliance Maven",

            }[x],
            index=0
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("👨‍⚖️ Continue to Chief Officer", type="primary"):
                with st.spinner("⏳ Chief verifying citations... (~60s)"):
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

        # Auto-save to history (once per audit run)
        if not st.session_state.audit_saved and st.session_state.final_report:
            try:
                db.save_audit(
                    file_name=st.session_state.uploaded_file_name or "untitled",
                    final_report=st.session_state.final_report,
                )
                st.session_state.audit_saved = True
                st.toast("💾 Saved to your audit history", icon="✅")
            except Exception as e:
                st.warning(f"⚠️ Could not save to history: {e}")

        with st.container(border=True):
            render_audit_report(st.session_state.final_report)

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

# ───────────────────────────────────────────────────────
# Tab 3: My History (past audits for the signed-in user)
# ───────────────────────────────────────────────────────
with tab_history:
    st.subheader("📜 My Audit History")
    st.caption("Every audit you've run shows up here. Only you can see your audits.")

    try:
        audits = db.list_my_audits()
    except Exception as e:
        st.error(f"Could not load history: {e}")
        audits = []

    if not audits:
        st.info("No audits yet. Run one from the **Contract Audit** tab and it will appear here.")
    else:
        for row in audits:
            ts = row.get("created_at", "")[:19].replace("T", " ")
            label = f"📄 {row['file_name']}  ·  {ts}"
            with st.expander(label):
                render_audit_report(row["final_report"])
                st.download_button(
                    label="💾 Download",
                    data=row["final_report"],
                    file_name=f"Aletheia_{row['file_name']}_{ts}.txt",
                    mime="text/plain",
                    key=f"dl_{row['id']}",
                )
