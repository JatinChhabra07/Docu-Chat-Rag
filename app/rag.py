from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.messages import HumanMessage
from dotenv import load_dotenv
import os



def load_and_chunk_pdf(pdf_path:str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 200
    )

    chunks = splitter.split_documents(documents)
    return chunks

def answer_question(vectorstore, question:str):
    docs = vectorstore.similarity_search(question, k=3)

    context = ""
    pages = set()

    for doc in docs:
        page = doc.metadata.get("page")
        pages.add(page)
        context+=f"(Page {page}): {doc.page_content}\n\n"

        llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"),
                       model="llama3-70b-8192",
                       temperature=0)

        prompt = f"""
        Answer the question using ONLY the context below.
        Always mention the page number in the answer.

        Context:
        {context}

        Question:
        {question}
        """

        response = llm([HumanMessage(content=prompt)])

        return response.content, sorted(pages)