# 🦜 PDF RAG Chatbot — LangChain Version

Chatbot hỏi đáp PDF sử dụng **LangChain** + BM25 + Gemini.

## 🚀 Cách chạy

```bash
# 1. Tạo và kích hoạt môi trường
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 2. Cài thư viện
pip install -r requirements.txt

# 3. Chạy app
streamlit run app.py
```

## 🏗️ Cấu trúc

```
pdf_rag_chatbot_langchain/
├── app.py                  # Streamlit UI
├── src/
│   └── rag_pipeline.py     # Toàn bộ RAG logic (LangChain)
├── requirements.txt
└── README.md
```

## 🔄 Pipeline LangChain

```
PDF
 ↓ PyMuPDFLoader          ← đọc PDF theo trang
 ↓ RecursiveCharacterTextSplitter  ← chunk thông minh
 ↓ BM25Retriever          ← tìm đoạn văn liên quan
 ↓ ConversationBufferWindowMemory  ← nhớ 4 lượt chat
 ↓ ChatGoogleGenerativeAI ← Gemini sinh câu trả lời
 ↓ ConversationalRetrievalChain   ← kết nối tất cả
```

## So sánh với version From Scratch

| | From Scratch | LangChain (bản này) |
|---|---|---|
| Số dòng code | ~200 | ~100 |
| Dễ debug | ✅ | ❌ khó hơn |
| Dễ mở rộng | ❌ tốn công | ✅ swap component dễ |
| Phù hợp | Học hiểu sâu | Production nhanh |
