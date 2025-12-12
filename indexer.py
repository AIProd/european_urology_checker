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


def _get_embedding_function() -> AzureOpenAIEmbeddings:
    """Create a *single, consistent* Azure embedding client."""
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


def _load_guideline_docs() -> List[Document]:
    """Load all guideline PDFs from ./guidelines as LangChain Documents."""
    if not os.path.exists(GUIDELINES_DIR):
        raise FileNotFoundError(
            f"Guidelines folder '{GUIDELINES_DIR}' not found. "
            "Upload the EU guideline PDFs via the Streamlit UI first."
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
        for d in docs:
            d.metadata["source_doc"] = file
        documents.extend(docs)

    return documents


def _build_inmemory_vectorstore() -> InMemoryVectorStore:
    """Build an in-memory vector store from the guideline PDFs."""
    embedding = _get_embedding_function()
    docs = _load_guideline_docs()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(docs)

    vector_store = InMemoryVectorStore(embedding=embedding)
    vector_store.add_documents(chunks)
    return vector_store


def retrieve_guidelines(query: str, k: int = 5) -> List[Document]:
    """
    End-to-end helper: load PDFs, build an in-memory vector store,
    and return top-k relevant guideline chunks for the given query.
    """
    vector_store = _build_inmemory_vectorstore()
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)


def build_knowledge_base() -> None:
    """
    Used by the 'Build Database' button.

    It doesn't persist anything; it just does a dry-run build + query
    so you get early visibility if Azure / guidelines are misconfigured.
    """
    docs = _load_guideline_docs()
    _ = retrieve_guidelines("statistical reporting guidelines", k=1)
    print(f"✅ Knowledge base OK – loaded {len(docs)} guideline pages.")
