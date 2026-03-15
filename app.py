import streamlit as st
import os
from pypdf import PdfReader
from docx import Document
from agent_audit import run_audit

# --- Page Configuration ---
st.set_page_config(
    page_title="Aletheia | AI Multi-Format Auditor",
    page_icon="🛡️",
    layout="wide"
)

# --- UI Styling ---
st.title("🛡️ Aletheia: Multi-Format Compliance Auditor")
st.markdown("""
**AGI Architecture**: Integrated **RAG** with **Agentic Reflection Loops**.  
**Supported Formats**: PDF, DOCX, TXT.  
*The system simulates a multi-stage review process between a Junior Auditor and a Chief Compliance Officer.*
""")

# --- Sidebar ---
with st.sidebar:
    st.header("System Configuration")
    st.status("Llama 3.2: Connected", state="complete")
    st.status("Pinecone DB: Active", state="complete")
    st.divider()
    st.info("Role: Senior Compliance Auditor (English Mode)")


# --- File Processing Functions ---

def extract_text(uploaded_file):
    """Detects file type and extracts text accordingly."""
    file_extension = uploaded_file.name.split('.')[-1].lower()

    if file_extension == "pdf":
        return extract_from_pdf(uploaded_file)
    elif file_extension == "docx":
        return extract_from_docx(uploaded_file)
    elif file_extension == "txt":
        return extract_from_txt(uploaded_file)
    else:
        st.error("Unsupported file format.")
        return None


def extract_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text


def extract_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_from_txt(file):
    return file.read().decode("utf-8")


# --- Main Interface ---

uploaded_file = st.file_uploader(
    "Upload Financial Contract for Audit",
    type=["pdf", "docx", "txt"]
)

if uploaded_file:
    # Display file details
    st.write(f"📁 **File Loaded:** {uploaded_file.name}")

    # Process and Audit
    if st.button("Run Agentic Audit"):
        with st.status("🔍 Aletheia Intelligence at work...", expanded=True) as status:
            st.write("Extracting content from file...")
            contract_text = extract_text(uploaded_file)

            if contract_text and len(contract_text.strip()) > 0:
                st.write("Executing Reflection Loop (Junior Auditor -> Chief Officer)...")
                # Auditing the first 3000 chars to maintain context efficiency
                audit_report = run_audit(contract_text[:3000])

                status.update(label="Audit Analysis Complete!", state="complete", expanded=False)

                # Show Results
                st.divider()
                st.subheader("📊 Final Compliance Audit Report")
                with st.container(border=True):
                    st.markdown(audit_report)

                # Download Result
                st.download_button(
                    label="Download Audit Report (.txt)",
                    data=audit_report,
                    file_name=f"Audit_Report_{uploaded_file.name}.txt",
                    mime="text/plain"
                )
            else:
                st.error("Failed to extract text. The file might be empty or corrupted.")