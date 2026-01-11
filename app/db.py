from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

EMBEDDING_MODEL = "nomic-embed-text"

def create_vectorDB(chunks):
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

def load_vector_db():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)