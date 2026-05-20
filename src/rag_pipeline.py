from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import List
import tempfile


def load_and_split_pdf(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[Document]:
    """Đọc PDF và chia thành các chunk."""
    loader = PyMuPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    return chunks


def build_retriever(chunks: List[Document], top_k: int = 5) -> BM25Retriever:
    """Xây dựng BM25Retriever từ danh sách chunks."""
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = top_k
    return retriever


def build_rag_chain(retriever: BM25Retriever, api_key: str):
    """
    Xây dựng RAG chain theo chuẩn mới LCEL (LangChain Expression Language).
    Không dùng ConversationalRetrievalChain deprecated nữa.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=api_key,
        temperature=0.3,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Bạn là trợ lý AI thông minh, chuyên trả lời câu hỏi dựa trên nội dung tài liệu PDF.

NGUYÊN TẮC:
- Chỉ trả lời dựa trên ngữ cảnh bên dưới. Không bịa đặt.
- Nếu không tìm thấy trong tài liệu, nói thẳng: "Tôi không tìm thấy thông tin này trong tài liệu."
- Trích dẫn số trang khi có thể.
- Trả lời bằng ngôn ngữ của người dùng (tiếng Việt hoặc tiếng Anh).

NGỮ CẢNH TỪ TÀI LIỆU:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    def format_docs(docs: List[Document]) -> str:
        parts = []
        for doc in docs:
            page = doc.metadata.get("page", "?")
            parts.append(f"[Trang {page + 1}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"])),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def ask(chain, retriever: BM25Retriever, question: str, chat_history: list) -> dict:
    """Gửi câu hỏi, trả về answer + source documents."""
    messages = []
    for turn in chat_history[-4:]:  # chỉ lấy 4 lượt gần nhất
        messages.append(HumanMessage(content=turn["user"]))
        messages.append(AIMessage(content=turn["answer"]))

    answer = chain.invoke({
        "question": question,
        "chat_history": messages,
    })

    sources = retriever.invoke(question)

    return {
        "answer": answer,
        "sources": sources,
    }


def save_uploaded_pdf(uploaded_file) -> str:
    """Lưu file upload từ Streamlit vào temp file, trả về path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name
