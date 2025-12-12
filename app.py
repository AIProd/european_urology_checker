# app.py

# --- SQLITE FIX FOR STREAMLIT CLOUD ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --------------------------------------

import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
import indexer

# Load .env (for local use)
load_dotenv()

st.set_page_config(page_title="EuroUrol Checker", layout="wide")

# Check environment config
if not os.getenv("AZURE_OPENAI_API_KEY"):
    st.error("⚠️ API Keys missing! If running locally, check .env. If on Cloud, check Secrets.")
    st.stop()

# Import graph AFTER ensuring env vars are loaded
from agent_graph import app_graph

st.title("🇪🇺 European Urology: Statistical Compliance Widget")

GUIDELINES_DIR = "./guidelines"


def _guidelines_present() -> bool:
    return (
        os.path.exists(GUIDELINES_DIR)
        and any(f.lower().endswith(".pdf") for f in os.listdir(GUIDELINES_DIR))
    )


# --- SIDEBAR: SETUP ---
with st.sidebar:
    st.header("1. System Setup")

    if _guidelines_present():
        st.success("✅ Guidelines present")
    else:
        st.warning("⚠️ Guidelines missing")

    st.divider()

    st.subheader("Update Knowledge Base")
    st.info("Upload the 4 European Urology statistical guideline PDFs here.")
    uploaded_guidelines = st.file_uploader(
        "Upload Guidelines",
        type="pdf",
        accept_multiple_files=True,
    )

    if st.button("Build Database"):
        if uploaded_guidelines:
            with st.spinner("Saving guidelines and validating knowledge base..."):
                if not os.path.exists(GUIDELINES_DIR):
                    os.makedirs(GUIDELINES_DIR)
                # Save uploaded PDFs into ./guidelines
                for pdf in uploaded_guidelines:
                    with open(os.path.join(GUIDELINES_DIR, pdf.name), "wb") as f:
                        f.write(pdf.getbuffer())

                # Run a dry-run build to verify everything works
                try:
                    indexer.build_knowledge_base()
                    st.success("Knowledge base validated!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error while building knowledge base: {e}")
        else:
            st.error("Upload files first.")


# --- MAIN: CHECKER ---
st.header("2. Run Compliance Check")
uploaded_paper = st.file_uploader("Upload Manuscript (PDF)", type="pdf")

if uploaded_paper:
    if st.button("Analyze Manuscript"):
        if not _guidelines_present():
            st.error("Upload and build the guideline knowledge base first (left sidebar).")
        else:
            with st.spinner("Agent is analyzing..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_paper.read())
                    tmp_path = tmp.name

                try:
                    loader = PyPDFLoader(tmp_path)
                    pages = loader.load()
                    full_text = "\n".join([p.page_content for p in pages])

                    initial_state = {
                        "paper_content": full_text,
                        "paper_type": "",
                        "audit_logs": [],
                        "final_report": "",
                    }

                    result = app_graph.invoke(initial_state)

                    st.success("Analysis Complete")
                    st.markdown(result["final_report"])

                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    os.remove(tmp_path)
