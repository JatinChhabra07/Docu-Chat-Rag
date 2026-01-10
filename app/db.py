from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings


def create_vectorDB(chunks):
    embeddings = OllamaEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

def load_vector_db():
    embeddings = OllamaEmbeddings()
    return FAISS.load_local("faiss_index", embeddings)