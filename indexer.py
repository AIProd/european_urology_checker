import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings

# Load environment variables
load_dotenv()

# CONFIG
GUIDELINES_DIR = "./guidelines"
DB_DIR = "./chroma_db"

def build_knowledge_base():
    # Verify Keys exist
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        print("ERROR: AZURE_OPENAI_API_KEY not found in .env file.")
        return

    # 1. Clean up old DB
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

    documents = []
    
    # 2. Check for PDFs
    if not os.path.exists(GUIDELINES_DIR):
        os.makedirs(GUIDELINES_DIR)
        print(f"Directory '{GUIDELINES_DIR}' created. Please add your PDF files there.")
        return

    files = [f for f in os.listdir(GUIDELINES_DIR) if f.endswith(".pdf")]
    if not files:
        print(f"No PDF files found in {GUIDELINES_DIR}.")
        return

    # 3. Load PDFs
    for file in files:
        print(f"Loading {file}...")
        loader = PyPDFLoader(os.path.join(GUIDELINES_DIR, file))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_doc"] = file
        documents.extend(docs)

    # 4. Split Text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)

    # 5. Create Vector Store
    print(f"Indexing {len(splits)} chunks...")
    
    embedding_function = AzureOpenAIEmbeddings(
        model=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )

    Chroma.from_documents(
        documents=splits, 
        embedding=embedding_function, 
        persist_directory=DB_DIR
    )
    print(f"Success! Database built in {DB_DIR}")

if __name__ == "__main__":
    build_knowledge_base()
