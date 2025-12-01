from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import json

def retriever(query, kb_dir="./kb", k=4, model_name="all-MiniLM-L6-v2"):
    loader = DirectoryLoader(kb_dir, glob="**/*.txt")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db = FAISS.from_documents(chunks, embeddings)

    results = db.similarity_search(query, k=k)

    output = []
    for r in results:
        output.append({
            "content": r.page_content,
            "metadata": r.metadata
        })

    return {"retrieved": output}

if __name__ == "__main__":
    print(json.dumps(
        retriever("what is ai?"),
        indent=2,
        ensure_ascii=False
    ))
