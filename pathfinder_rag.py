"""
Pathfinder RAG

Builds the FAISS vector store the agent uses for advice retrieval.

A note on what belongs in retrieval
-----------------------------------
The first version embedded three documents: the resume, the preferences sheet,
and the strategy guide, then pulled the four nearest chunks on every turn. That
had a subtle failure mode. Facts the app needs on every single turn, like the
salary floor or the list of target cities, were only present when the user's
phrasing happened to retrieve the chunk containing them. Ask about pay and the
model saw the salary range; ask about a job in Denver and it might not.

Retrieval is the right tool for a corpus too large to fit in a prompt and where
different questions need different passages. That describes the strategy guide,
which is genuinely a reference document. It does not describe a one-page
preference sheet, which is now structured data in profile.yaml and goes into
every prompt in full.

So the split is deliberate:
  profile.yaml  -> structured, complete, injected every turn, drives scoring
  resume.txt    -> embedded, retrieved for experience-specific questions
  strategy guide-> embedded, retrieved for advice questions

The index is cached on disk so a restart does not re-pay for embeddings.
"""

from __future__ import annotations

import hashlib
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
try:
    import streamlit as st
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DOCS_FOLDER = os.path.join(BASE_DIR, "rag_docs")
INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")

# (filename, source label). The first file that exists for each entry wins,
# which lets the resume live as .txt or .pdf without a code change.
SOURCES: list[tuple[list[str], str]] = [
    (["resume.txt", "Resume.pdf", "resume.pdf"], "resume"),
    (["job_search_guide.txt"], "strategy_guide"),
]


def _load_one(candidates: list[str], label: str):
    for filename in candidates:
        path = os.path.join(RAG_DOCS_FOLDER, filename)
        if not os.path.exists(path):
            continue
        loader = (PyPDFLoader(path) if path.lower().endswith(".pdf")
                  else TextLoader(path, encoding="utf-8"))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = label
        print(f"[rag] loaded {filename} as '{label}' ({len(docs)} pages)")
        return docs
    print(f"[rag] no file found for '{label}', skipping")
    return []


def load_documents():
    docs = []
    for candidates, label in SOURCES:
        docs.extend(_load_one(candidates, label))
    return docs


def _corpus_fingerprint() -> str:
    """Hash of the source files, so a stale cached index is never reused."""
    h = hashlib.sha256()
    for candidates, _ in SOURCES:
        for filename in candidates:
            path = os.path.join(RAG_DOCS_FOLDER, filename)
            if os.path.exists(path):
                h.update(filename.encode())
                h.update(str(os.path.getmtime(path)).encode())
                break
    return h.hexdigest()[:16]


def build_vectorstore() -> FAISS:
    """Load, chunk, embed, and index. Reuses a cached index when the docs match."""
    fingerprint = _corpus_fingerprint()
    stamp_path = os.path.join(INDEX_DIR, "fingerprint.txt")

    if os.path.isdir(INDEX_DIR) and os.path.exists(stamp_path):
        try:
            if open(stamp_path).read().strip() == fingerprint:
                store = FAISS.load_local(
                    INDEX_DIR, OpenAIEmbeddings(),
                    allow_dangerous_deserialization=True,
                )
                print("[rag] reused cached index")
                return store
        except Exception as exc:
            print(f"[rag] cached index unusable, rebuilding: {exc}")

    documents = load_documents()
    if not documents:
        raise FileNotFoundError(
            f"No RAG source documents found in {RAG_DOCS_FOLDER}."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,        # larger than before: bullet points were being
        chunk_overlap=120,     # split mid-sentence at 500/50
    )
    chunks = splitter.split_documents(documents)
    print(f"[rag] {len(documents)} documents -> {len(chunks)} chunks")

    store = FAISS.from_documents(chunks, OpenAIEmbeddings())
    try:
        os.makedirs(INDEX_DIR, exist_ok=True)
        store.save_local(INDEX_DIR)
        with open(stamp_path, "w") as fh:
            fh.write(fingerprint)
        print("[rag] index cached to disk")
    except Exception as exc:
        print(f"[rag] could not cache index: {exc}")
    return store


def get_retriever(k: int = 4):
    return build_vectorstore().as_retriever(search_kwargs={"k": k})


def retrieve_context(query: str, retriever) -> str:
    """Nearest chunks, labelled by source, ready to drop into a prompt."""
    try:
        docs = retriever.invoke(query)
    except Exception as exc:
        print(f"[rag] retrieval failed: {exc}")
        return ""
    return "\n\n".join(
        f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )
