from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import Document
from typing import List
import tempfile
import os


def load_and_split_pdf(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[Document]:
    """
    Đọc PDF và chia thành các chunk bằng LangChain.
    PyMuPDFLoader: đọc từng trang PDF
    RecursiveCharacterTextSplitter: chia thông minh theo đoạn văn, câu, từ
    """
    # Load PDF
    loader = PyMuPDFLoader(pdf_path)
    pages = loader.load()  # List[Document], mỗi Document = 1 trang

    # Split thành chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],  # ưu tiên tách theo đoạn văn
    )
    chunks = splitter.split_documents(pages)
    return chunks


def build_retriever(chunks: List[Document], top_k: int = 5) -> BM25Retriever:
    """
    Xây dựng BM25Retriever từ danh sách chunks.
    LangChain BM25Retriever wrap rank_bm25 bên dưới.
    """
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = top_k
    return retriever


def build_rag_chain(retriever: BM25Retriever, api_key: str) -> ConversationalRetrievalChain:
    """
    Xây dựng ConversationalRetrievalChain:
    - LLM: Gemini 1.5 Flash
    - Memory: lưu 4 lượt hội thoại gần nhất
    - Retriever: BM25
    """
    # LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.3,
    )

    # Memory: giữ 4 lượt chat gần nhất
    memory = ConversationBufferWindowMemory(
        k=4,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    # Chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,  # trả về đoạn văn tham khảo
        verbose=False,
    )
    return chain


def ask(chain: ConversationalRetrievalChain, question: str) -> dict:
    """
    Gửi câu hỏi vào chain, trả về answer + source documents.
    """
    result = chain.invoke({"question": question})
    return {
        "answer": result["answer"],
        "sources": result.get("source_documents", []),
    }


def save_uploaded_pdf(uploaded_file) -> str:
    """Lưu file upload từ Streamlit vào temp file, trả về path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name
