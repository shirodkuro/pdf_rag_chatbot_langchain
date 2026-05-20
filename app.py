import streamlit as st
import os
from src.rag_pipeline import (
    load_and_split_pdf,
    build_retriever,
    build_rag_chain,
    ask,
    save_uploaded_pdf,
)

# ─── Cấu hình trang ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF RAG Chatbot (LangChain)",
    page_icon="🦜",
    layout="wide",
)

st.title("🦜 PDF RAG Chatbot")
st.caption("Powered by LangChain + BM25 + Gemini")

# ─── Session state ────────────────────────────────────────────────────────────
if "chain" not in st.session_state:
    st.session_state.chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Cấu hình")

    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Lấy tại https://aistudio.google.com/app/apikey",
    )

    st.divider()
    st.header("📂 Upload PDF")

    uploaded_file = st.file_uploader("Chọn file PDF", type=["pdf"])

    with st.expander("🔧 Cài đặt nâng cao"):
        chunk_size = st.slider("Chunk size (ký tự)", 300, 2000, 500, 100)
        chunk_overlap = st.slider("Chunk overlap (ký tự)", 0, 300, 100, 50)
        top_k = st.slider("Top-k retrieval", 1, 10, 5)

    if uploaded_file:
        if st.button("🚀 Xử lý PDF", type="primary", use_container_width=True):
            if not api_key:
                st.error("⚠️ Vui lòng nhập Gemini API Key!")
            else:
                with st.spinner("Đang xử lý PDF..."):
                    try:
                        # Lưu file tạm
                        tmp_path = save_uploaded_pdf(uploaded_file)

                        # LangChain pipeline
                        chunks = load_and_split_pdf(tmp_path, chunk_size, chunk_overlap)
                        os.unlink(tmp_path)

                        retriever = build_retriever(chunks, top_k=top_k)
                        chain, retriever = build_rag_chain(retriever, api_key)

                        # Lưu vào session
                        st.session_state.chain = chain
                        st.session_state.retriever = retriever
                        st.session_state.pdf_name = uploaded_file.name
                        st.session_state.num_chunks = len(chunks)
                        st.session_state.chat_history = []

                        st.success(f"✅ Xong! **{len(chunks)}** chunks từ **{uploaded_file.name}**")

                    except Exception as e:
                        st.error(f"❌ Lỗi: {str(e)}")

    if st.session_state.pdf_name:
        st.divider()
        st.success(f"📄 **{st.session_state.pdf_name}**")
        st.info(f"🔢 {st.session_state.num_chunks} chunks")

        if st.button("🗑️ Reset", use_container_width=True):
            st.session_state.chain = None
            st.session_state.chat_history = []
            st.session_state.pdf_name = None
            st.rerun()

# ─── Main chat area ───────────────────────────────────────────────────────────
if not st.session_state.chain:
    st.info("👈 Nhập API Key và upload PDF để bắt đầu.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1️⃣ Nhập API Key")
    with col2:
        st.markdown("### 2️⃣ Upload PDF")
    with col3:
        st.markdown("### 3️⃣ Đặt câu hỏi")
else:
    # Hiển thị lịch sử
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["user"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            if turn.get("sources"):
                with st.expander("📎 Đoạn văn tham khảo"):
                    for i, doc in enumerate(turn["sources"], 1):
                        page = doc.metadata.get("page", "?")
                        st.markdown(f"**[Trang {page + 1}]**\n\n{doc.page_content}")
                        if i < len(turn["sources"]):
                            st.divider()

    # Input
    question = st.chat_input("Nhập câu hỏi về nội dung PDF...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Đang suy nghĩ..."):
                result = ask(
                    st.session_state.chain,
                    st.session_state.retriever,
                    question,
                    st.session_state.chat_history,
                )

            st.write(result["answer"])

            if result["sources"]:
                with st.expander("📎 Đoạn văn tham khảo"):
                    for i, doc in enumerate(result["sources"], 1):
                        page = doc.metadata.get("page", "?")
                        st.markdown(f"**[Trang {page + 1}]**\n\n{doc.page_content}")
                        if i < len(result["sources"]):
                            st.divider()

        st.session_state.chat_history.append({
            "user": question,
            "answer": result["answer"],
            "sources": result["sources"],
        })
