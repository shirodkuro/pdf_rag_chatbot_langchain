# evaluation/ragas_eval.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragas import evaluate
from ragas.metrics import (
    faithfulness,        # Bot có "chém gió" không?
    answer_relevancy,   # Câu trả lời có đúng trọng tâm không?
    context_precision,  # Chunks tìm được có đúng không?
)
from datasets import Dataset
import json
from src.rag_pipeline import (
    load_and_split_pdf,
    build_retriever,
    build_rag_chain,
    ask,
)

def run_ragas_eval(pdf_path: str, api_key: str, test_set_path: str):
    # Load test set
    with open(test_set_path) as f:
        test_set = json.load(f)

    # Build pipeline
    chunks = load_and_split_pdf(pdf_path)
    retriever = build_retriever(chunks)
    chain, retriever = build_rag_chain(retriever, api_key)

    chat_history = []

    # Chạy từng câu hỏi
    questions, answers, contexts, ground_truths = [], [], [], []

    for item in test_set:
        result = ask(chain, retriever, item["question"], chat_history)

        questions.append(item["question"])
        answers.append(result["answer"])
        contexts.append([doc.page_content for doc in result["sources"]])
        ground_truths.append(item["ground_truth"])

        chat_history.append({                    
            "user": item["question"],
            "answer": result["answer"],
        })

    # Đóng gói dataset
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    # Chấm điểm
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )

    print(result)
    # Output ví dụ:
    # faithfulness       : 0.85
    # answer_relevancy   : 0.91
    # context_precision  : 0.78