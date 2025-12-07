
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

KB_PATH = "./kb"

def retriever(state):
    if not os.path.isdir(KB_PATH):
        raise RuntimeError(f"Knowledge base path {KB_PATH!r} is not a directory. Create it and put your docs inside.")

    # This loader will use PyPDFLoader for PDFs and avoid unstructured
    loader = DirectoryLoader(
        KB_PATH,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)

    results = db.similarity_search(state["query"], k=4)
    return {"retrieved": results}