"""
Pathfinder RAG System

Loads Colin's three career documents from the rag_docs/ folder, splits them
into chunks, embeds them, and stores them in a FAISS vector database. The
agent queries this vector store on every turn to inject personalized context
into the system prompt.

Documents loaded:
  - rag_docs/Resume.pdf            -> Colin's resume
  - rag_docs/job_preferences.txt   -> What Colin is looking for in a role
  - rag_docs/job_search_guide.txt  -> General job search strategy guidance

Pattern follows the class lesson: PyPDFLoader for PDFs, TextLoader for .txt,
RecursiveCharacterTextSplitter for chunking, OpenAIEmbeddings for embeddings,
FAISS for the vector store.
"""

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Load secrets locally from .env, then overlay any Streamlit Cloud secrets.
load_dotenv()
try:
    import streamlit as st
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────
# DOCUMENT PATHS: pulled from the rag_docs/ folder in the project
# ─────────────────────────────────────────────────────────────────

RAG_DOCS_FOLDER = "rag_docs"

RESUME_PATH      = os.path.join(RAG_DOCS_FOLDER, "Resume.pdf")
PREFERENCES_PATH = os.path.join(RAG_DOCS_FOLDER, "job_preferences.txt")
STRATEGY_PATH    = os.path.join(RAG_DOCS_FOLDER, "job_search_guide.txt")


# ─────────────────────────────────────────────────────────────────
# LOAD THE DOCUMENTS
# ─────────────────────────────────────────────────────────────────

def load_documents():
    """
    Loads the three RAG documents from disk using LangChain document loaders.
    PDFs are loaded with PyPDFLoader, text files with TextLoader.
    Each document is tagged with a 'source' in metadata so we can show the
    agent where retrieved context came from.
    """
    all_docs = []

    # 1. Resume (PDF)
    print(f"Loading resume from {RESUME_PATH}...")
    resume_loader = PyPDFLoader(RESUME_PATH)
    resume_docs = resume_loader.load()
    for doc in resume_docs:
        doc.metadata["source"] = "resume"
    all_docs.extend(resume_docs)

    # 2. Job preferences (text)
    print(f"Loading preferences from {PREFERENCES_PATH}...")
    prefs_loader = TextLoader(PREFERENCES_PATH, encoding="utf-8")
    prefs_docs = prefs_loader.load()
    for doc in prefs_docs:
        doc.metadata["source"] = "preferences"
    all_docs.extend(prefs_docs)

    # 3. Job search strategy guide (text)
    print(f"Loading strategy guide from {STRATEGY_PATH}...")
    strategy_loader = TextLoader(STRATEGY_PATH, encoding="utf-8")
    strategy_docs = strategy_loader.load()
    for doc in strategy_docs:
        doc.metadata["source"] = "strategy_guide"
    all_docs.extend(strategy_docs)

    print(f"Loaded {len(all_docs)} document pages total.")
    return all_docs


# ─────────────────────────────────────────────────────────────────
# BUILD THE VECTOR STORE (FAISS vector db)
# ─────────────────────────────────────────────────────────────────

def build_vectorstore():
    """
    Splits the loaded documents into chunks, embeds them with OpenAI
    embeddings, and stores them in a FAISS vector database.
    """
    documents = load_documents()

    # Split into chunks for embedding
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    # Embed and store in FAISS
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore


def get_retriever():
    """
    Builds the vector store and returns a retriever.
    Called once at agent startup. Returns the top 4 most relevant chunks
    per query.
    """
    vectorstore = build_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": 4})


def retrieve_context(query: str, retriever) -> str:
    """
    Given a user query and a retriever, returns the most relevant chunks
    from the vector store, formatted with their source labels for the
    system prompt.
    """
    docs = retriever.invoke(query)
    context_parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        context_parts.append(f"[{source}]\n{doc.page_content}")
    return "\n\n".join(context_parts)