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


def infer_guideline_type(filename: str) -> str:
    """Map a guideline PDF to a high-level type for filtered retrieval."""
    fn = filename.lower()
    if "causality" in fn:
        return "causality"
    if "figure" in fn or "table" in fn:
        return "figures_tables"
    if "systematic" in fn or "meta" in fn:
        return "systematic_meta"
    if "stat" in fn:
        return "statistics"
    return "other"


def build_knowledge_base():
    # 0. Key check
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        print("ERROR: AZURE_OPENAI_API_KEY not found in .env file.")
        return

    # 1. Clean up old DB
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

    os.makedirs(GUIDELINES_DIR, exist_ok=True)

    # 2. Find PDFs
    files = [f for f in os.listdir(GUIDELINES_DIR) if f.lower().endswith(".pdf")]
    if not files:
        print(f"No PDF files found in {GUIDELINES_DIR}.")
        return

    documents = []

    # 3. Load PDFs + set metadata
    for file in files:
        path = os.path.join(GUIDELINES_DIR, file)
        print(f"Loading {path}...")
        loader = PyPDFLoader(path)
        docs = loader.load()
        gtype = infer_guideline_type(file)
        for doc in docs:
            doc.metadata["source_doc"] = file
            doc.metadata["guideline_type"] = gtype
        documents.extend(docs)

    if not documents:
        print("No documents loaded from guideline PDFs.")
        return

    # 4. Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    splits = text_splitter.split_documents(documents)
    print(f"Indexing {len(splits)} chunks...")

    # 5. Embeddings + Vector store
    embedding_function = AzureOpenAIEmbeddings(
        model=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )

    Chroma.from_documents(
        documents=splits,
        embedding=embedding_function,
        persist_directory=DB_DIR,
        # collection_name optional; default is fine
    )

    print(f"✅ Success! Database built in {DB_DIR}")


if __name__ == "__main__":
    build_knowledge_base()
