from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def retriever(state):
    loader = DirectoryLoader("./kb")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    emb = HuggingFaceEmbeddings()
    db = FAISS.from_documents(chunks, emb)

    results = db.similarity_search(state["query"], k=4)
    return {"retrieved": results}
