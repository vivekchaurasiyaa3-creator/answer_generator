# Agents/retriever_agent.py
"""Lightweight retriever helper with safe imports and CLI.

Usage:
  python Agents/retriever_agent.py --query "your question" --kb ./kb --k 4

This module tries to import LangChain pieces from the main package and
falls back to `langchain_community` locations if available. If the
required packages are missing it prints a helpful install hint.
"""
import argparse
import json
import sys
from typing import Any, Dict


_IMPORT_ERROR = None

def _import_langchain_components():
    global _IMPORT_ERROR
    try:
        # preferred locations
        from langchain.document_loaders import DirectoryLoader  # type: ignore
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
        from langchain.vectorstores import FAISS  # type: ignore
        from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore
        return DirectoryLoader, RecursiveCharacterTextSplitter, FAISS, HuggingFaceEmbeddings
    except Exception:
        try:
            # community package fallback (some setups use this)
            from langchain_community.document_loaders import DirectoryLoader  # type: ignore
            from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
            from langchain_community.vectorstores import FAISS  # type: ignore
            from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
            return DirectoryLoader, RecursiveCharacterTextSplitter, FAISS, HuggingFaceEmbeddings
        except Exception as e:
            _IMPORT_ERROR = e
            return None, None, None, None


def _ensure_dependencies():
    DirectoryLoader, RecursiveCharacterTextSplitter, FAISS, HuggingFaceEmbeddings = _import_langchain_components()
    if DirectoryLoader is None:
        msg = (
            "Missing required LangChain packages or their community equivalents.\n"
            "Install dependencies with:\n\n"
            "  pip install -r requirements.txt\n\n"
            "Or install minimal set:\n"
            "  pip install langchain faiss-cpu sentence-transformers transformers\n"
        )
        print(msg, file=sys.stderr)
        if _IMPORT_ERROR is not None:
            print("Underlying import error:", repr(_IMPORT_ERROR), file=sys.stderr)
        sys.exit(1)
    return DirectoryLoader, RecursiveCharacterTextSplitter, FAISS, HuggingFaceEmbeddings


def retriever(state: Dict[str, Any], kb_dir: str = "./kb", k: int = 4, model_name: str = "all-MiniLM-L6-v2") -> Dict[str, Any]:
    """Load documents from `kb_dir`, split, embed and run a similarity search.

    Returns a JSON-serializable dictionary with `retrieved` list of dicts.
    """
    DirectoryLoader, RecursiveCharacterTextSplitter, FAISS, HuggingFaceEmbeddings = _ensure_dependencies()

    loader = DirectoryLoader(kb_dir)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db = FAISS.from_documents(chunks, embeddings)

    results = db.similarity_search(state.get("query", ""), k=k)

    serialized = []
    for r in results:
        # r is usually a langchain.Document
        content = getattr(r, "page_content", str(r))
        metadata = getattr(r, "metadata", {}) or {}
        serialized.append({"content": content, "metadata": metadata})

    return {"retrieved": serialized}


def _cli():
    parser = argparse.ArgumentParser(description="Run simple retriever against local KB")
    parser.add_argument("--query", required=True, help="Query text")
    parser.add_argument("--kb", default="./kb", help="Path to knowledge base directory")
    parser.add_argument("--k", type=int, default=4, help="Number of results to retrieve")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="HF embedding model name")
    args = parser.parse_args()

    out = retriever({"query": args.query}, kb_dir=args.kb, k=args.k, model_name=args.model)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _cli()

