# app.py

# --- SQLITE FIX FOR STREAMLIT CLOUD (harmless even if unused) ---
__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3", sys.modules.get("sqlite3"))
# ----------------------------------------------------------------

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

import indexer
from agent_graph import app_graph

# Load env vars (for local runs; on Streamlit Cloud use Secrets)
load_dotenv()

st.set_page_config(page_title="EuroUrol Checker", layout="wide")
st.title("🇪🇺 European Urology: Statistical Compliance Widget")

GUIDELINES_DIR = "./guidelines"


def _guidelines_present() -> bool:
    """Return True if at least one PDF is present in ./guidelines."""
    return (
        os.path.exists(GUIDELINES_DIR)
        and any(f.lower().endswith(".pdf") for f in os.listdir(GUIDELINES_DIR))
    )


# --- ENV CHECK ---
if not os.getenv("AZURE_OPENAI_API_KEY"):
    st.error(
        "⚠️ Azure OpenAI keys missing. "
        "If running locally, check your .env. If on Streamlit Cloud, check Secrets."
    )
    st.stop()


# --- SIDEBAR: SETUP ---
with st.sidebar:
    st.header("1. System Setup")

    if _guidelines_present():
        st.success("✅ Guidelines present in ./guidelines")
    else:
        st.warning("⚠️ Guidelines missing. Upload the 4 EU guideline PDFs below.")

    st.divider()

    st.subheader("Upload / Update Guidelines")
    st.info(
        "Upload the four guideline PDFs:\n"
        "- Causality\n"
        "- Figures and Tables\n"
        "- Stat Reporting Guidelines\n"
        "- Systematic review and MA guidelines"
    )

    uploaded_guidelines = st.file_uploader(
        "Upload Guideline PDFs",
        type="pdf",
        accept_multiple_files=True,
    )

    if st.button("Build / Validate Knowledge Base"):
        if not uploaded_guidelines:
            st.error("Please upload the guideline PDFs first.")
        else:
            with st.spinner("Saving guideline PDFs and validating knowledge base..."):
                os.makedirs(GUIDELINES_DIR, exist_ok=True)

                # Save uploaded PDFs into ./guidelines
                for pdf in uploaded_guidelines:
                    save_path = os.path.join(GUIDELINES_DIR, pdf.name)
                    with open(save_path, "wb") as f:
                        f.write(pdf.getbuffer())

                try:
                    indexer.build_knowledge_base()
                    st.success("✅ Knowledge base validation complete.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error while building/validating knowledge base: {e}")


# --- MAIN: MANUSCRIPT CHECKER ---

st.header("2. Run Compliance Check")

uploaded_paper = st.file_uploader("Upload Manuscript (PDF)", type="pdf")

if uploaded_paper:
    if st.button("Analyze Manuscript"):
        if not _guidelines_present():
            st.error(
                "Guidelines are not present. Please upload the four EU guideline PDFs "
                "and click 'Build / Validate Knowledge Base' in the sidebar first."
            )
        else:
            with st.spinner("Agent graph is analyzing the manuscript..."):
                # Save uploaded paper to a temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_paper.read())
                    tmp_path = tmp.name

                try:
                    loader = PyPDFLoader(tmp_path)
                    pages = loader.load()
                    full_text = "\n".join(p.page_content for p in pages)

                    initial_state = {
                        "paper_content": full_text,
                        "paper_type": "",
                        "audit_logs": [],
                        "final_report": "",
                    }

                    result = app_graph.invoke(initial_state)

                    st.success("Analysis complete.")
                    st.markdown(result["final_report"])

                except Exception as e:
                    st.error(f"Error during analysis: {e}")
                finally:
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
