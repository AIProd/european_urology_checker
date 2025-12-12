# indexer.py

import os
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

load_dotenv()

GUIDELINES_DIR = "./guidelines"


def infer_guideline_type(filename: str) -> str:
    """Map guideline filename to a high-level type."""
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


def _get_embedding_function() -> AzureOpenAIEmbeddings:
    """Create a single, consistent Azure embedding client."""
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    embedding_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

    if not all([azure_endpoint, azure_api_key, azure_api_version, embedding_deployment]):
        raise ValueError(
            "Missing Azure embedding env vars. Expected: "
            "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, "
            "AZURE_OPENAI_API_VERSION, AZURE_EMBEDDING_DEPLOYMENT."
        )

    return AzureOpenAIEmbeddings(
        azure_deployment=embedding_deployment,
        azure_endpoint=azure_endpoint,
        openai_api_key=azure_api_key,
        openai_api_version=azure_api_version,
    )


def _load_guideline_docs() -> List[Document]:
    """Load all guideline PDFs as LangChain Documents with guideline_type metadata."""
    if not os.path.exists(GUIDELINES_DIR):
        raise FileNotFoundError(
            f"Guidelines folder '{GUIDELINES_DIR}' not found. "
            "Upload the EU guideline PDFs via the Streamlit sidebar first."
        )

    pdf_files = [
        f for f in os.listdir(GUIDELINES_DIR)
        if f.lower().endswith(".pdf")
    ]
    if not pdf_files:
        raise RuntimeError(
            f"No PDF files found in {GUIDELINES_DIR}. "
            "Upload the 4 EU guideline PDFs there."
        )

    documents: List[Document] = []
    for file in pdf_files:
        path = os.path.join(GUIDELINES_DIR, file)
        loader = PyPDFLoader(path)
        docs = loader.load()
        gtype = infer_guideline_type(file)
        for d in docs:
            d.metadata["source_doc"] = file
            d.metadata["guideline_type"] = gtype
        documents.extend(docs)

    return documents


def _split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    return splitter.split_documents(docs)


def retrieve_guidelines_by_type(
    guideline_type: str,
    query: str,
    k: int = 5,
) -> List[Document]:
    """
    Load PDFs, build an in-memory vector store for the requested guideline type,
    and return the top-k relevant chunks.
    """
    all_docs = _load_guideline_docs()
    typed_docs = [
        d for d in all_docs
        if d.metadata.get("guideline_type") == guideline_type
    ]
    if not typed_docs:
        raise RuntimeError(
            f"No documents found for guideline_type='{guideline_type}'. "
            "Check that filenames contain e.g. 'Stat', 'Causality', "
            "'Figures and Tables', 'Systematic review'."
        )

    chunks = _split_docs(typed_docs)
    embedding = _get_embedding_function()

    vs = InMemoryVectorStore(embedding=embedding)
    vs.add_documents(chunks)
    retriever = vs.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)


def build_knowledge_base() -> None:
    """
    Called by the 'Build Database' button.

    It doesn't persist anything; it just:
    - ensures the 4 PDFs exist
    - does a tiny test retrieval for each guideline type
    so you get early feedback if something is misconfigured.
    """
    docs = _load_guideline_docs()
    print(f"Loaded {len(docs)} guideline pages from {GUIDELINES_DIR}.")

    tested_types = ["statistics", "figures_tables", "causality", "systematic_meta"]
    for gtype in tested_types:
        try:
            _ = retrieve_guidelines_by_type(gtype, "test", k=1)
            print(f"Guideline type '{gtype}': OK")
        except Exception as e:
            print(f"Guideline type '{gtype}': ERROR – {e}")
            raise

    print("✅ Knowledge base validation complete.")
