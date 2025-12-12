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
import indexer  # our updated indexer

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

# --- SIDEBAR: SETUP ---
with st.sidebar:
    st.header("1. System Setup")

    if os.path.exists("./chroma_db"):
        st.success("✅ Knowledge Base Loaded")
    else:
        st.warning("⚠️ Knowledge Base Missing")

    st.divider()

    st.subheader("Update Knowledge Base")
    st.info("Upload 4 guideline PDFs here.")
    uploaded_guidelines = st.file_uploader(
        "Upload Guidelines",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("Build Database"):
        if uploaded_guidelines:
            with st.spinner("Indexing guidelines into Chroma..."):
                if not os.path.exists("./guidelines"):
                    os.makedirs("./guidelines")
                # Save uploaded PDFs into ./guidelines
                for pdf in uploaded_guidelines:
                    with open(os.path.join("./guidelines", pdf.name), "wb") as f:
                        f.write(pdf.getbuffer())

                # Build KB (will delete old DB and rebuild)
                indexer.build_knowledge_base()
                st.success("Database updated!")
                st.rerun()
        else:
            st.error("Upload files first.")

# --- MAIN: CHECKER ---
st.header("2. Run Compliance Check")
uploaded_paper = st.file_uploader("Upload Manuscript (PDF)", type="pdf")

if uploaded_paper:
    if st.button("Analyze Manuscript"):
        if not os.path.exists("./chroma_db"):
            st.error("Build knowledge base first!")
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
