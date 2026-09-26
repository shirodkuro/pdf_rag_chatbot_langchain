# evaluation/run_all.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from ragas_eval import run_ragas_eval
from llm_judge import llm_judge
from src.rag_pipeline import load_and_split_pdf, build_retriever, build_rag_chain, ask

def full_evaluation(pdf_path, api_key, test_set_path):
    with open(test_set_path) as f:
        test_set = json.load(f)

    chunks = load_and_split_pdf(pdf_path)
    retriever = build_retriever(chunks)
    chain, retriever = build_rag_chain(retriever, api_key)

    chat_history = []

    print("=" * 50)
    print("KẾT QUẢ ĐÁNH GIÁ TỪNG CÂU")
    print("=" * 50)

    for item in test_set:
        result = ask(chain, retriever, item["question"], chat_history)
        score = llm_judge(api_key, item["question"], result["answer"], item["ground_truth"])

        print(f"\n❓ {item['question']}")
        print(f"🤖 {result['answer'][:100]}...")
        print(f"📊 Điểm: Chính xác={score['accuracy']}/5 | Đầy đủ={score['completeness']}/5 | Rõ ràng={score['clarity']}/5")
        print(f"💬 {score['comment']}")

        chat_history.append({
            "user": item["question"],
            "answer": result["answer"],
        })

    print("\n" + "=" * 50)
    print("RAGAS METRICS TỔNG THỂ")
    print("=" * 50)
    run_ragas_eval(pdf_path, api_key, test_set_path)

if __name__ == "__main__":
    full_evaluation(
        pdf_path="C:/Users/FPT/Downloads/pdf_rag_chatbot_langchain/bao-cao-btl-daktmt 1.pdf",
        api_key=<api_key>,
        test_set_path="evaluation/test_set.json"
    )