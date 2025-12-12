# indexer.py

import os
import shutil
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

GUIDELINES_DIR = "./guidelines"
DB_DIR = "./chroma_db"


def _get_embedding_function() -> AzureOpenAIEmbeddings:
    """
    Return a *single, consistent* Azure embedding instance.

    IMPORTANT:
      - AZURE_EMBEDDING_DEPLOYMENT should be your embedding deployment name.
      - Do NOT change this without rebuilding the DB (we already do rmtree before build).
    """
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    embedding_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

    if not all([azure_endpoint, azure_api_key, azure_api_version, embedding_deployment]):
        raise ValueError(
            "Missing Azure embedding env vars. "
            "Expected AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, "
            "AZURE_OPENAI_API_VERSION, AZURE_EMBEDDING_DEPLOYMENT."
        )

    return AzureOpenAIEmbeddings(
        azure_deployment=embedding_deployment,
        azure_endpoint=azure_endpoint,
        openai_api_key=azure_api_key,
        openai_api_version=azure_api_version,
    )


def build_knowledge_base() -> None:
    # Verify keys exist
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        print("ERROR: AZURE_OPENAI_API_KEY not found. Check your Streamlit secrets or .env.")
        return

    # 1. Clean up old DB – avoids embedding dimension mismatch
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

    # 2. Ensure guidelines dir exists and has PDFs
    if not os.path.exists(GUIDELINES_DIR):
        os.makedirs(GUIDELINES_DIR)
        print(f"Directory '{GUIDELINES_DIR}' created. Please add your PDF files there.")
        return

    pdf_files = [f for f in os.listdir(GUIDELINES_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in {GUIDELINES_DIR}. Upload the four guideline PDFs first.")
        return

    # 3. Load PDFs
    documents = []
    for file in pdf_files:
        path = os.path.join(GUIDELINES_DIR, file)
        print(f"Loading {file}...")
        loader = PyPDFLoader(path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_doc"] = file
        documents.extend(docs)

    # 4. Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = splitter.split_documents(documents)

    print(f"Indexing {len(splits)} chunks into Chroma...")

    embedding_function = _get_embedding_function()

    try:
        Chroma.from_documents(
            documents=splits,
            embedding=embedding_function,
            persist_directory=DB_DIR,
        )
        print(f"✅ Success! Database built at {DB_DIR}")
    except Exception as e:
        # This will show the *real* error in the Streamlit logs
        print("[indexer] ERROR while building Chroma DB:", repr(e))
        raise


if __name__ == "__main__":
    build_knowledge_base()
