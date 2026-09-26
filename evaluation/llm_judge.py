# evaluation/llm_judge.py

from langchain_google_genai import ChatGoogleGenerativeAI
import json

JUDGE_PROMPT = """
Bạn là giám khảo đánh giá chất lượng câu trả lời của chatbot.

Câu hỏi: {question}
Câu trả lời của bot: {answer}
Câu trả lời chuẩn: {ground_truth}

Hãy chấm điểm từ 1-5 cho các tiêu chí sau và giải thích ngắn gọn:
1. Độ chính xác (có đúng với tài liệu không?)
2. Độ đầy đủ (có trả lời đủ ý không?)
3. Độ mạch lạc (câu trả lời có rõ ràng không?)

Trả lời theo format JSON:
{{
  "accuracy": <1-5>,
  "completeness": <1-5>,
  "clarity": <1-5>,
  "comment": "<nhận xét ngắn>"
}}
"""

def llm_judge(api_key: str, question: str, answer: str, ground_truth: str):
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=api_key,
        temperature=0,
    )

    prompt = JUDGE_PROMPT.format(
        question=question,
        answer=answer,
        ground_truth=ground_truth,
    )

    response = llm.invoke(prompt)

    if isinstance(response.content, list):
        text = response.content[0]["text"]
    else:
        text = response.content

    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    return json.loads(text)

# Chạy thử
if __name__ == "__main__":
    result = llm_judge(
        api_key="AIzaSyBrI985Vp5uYyI9LrDZgClx8DY13wDrQ3s",
        question="RAG là gì?",
        answer="RAG là kỹ thuật tìm kiếm và sinh văn bản",
        ground_truth="RAG kết hợp Retrieval và Generation",
    )
    print(result)
    # {
    #   "accuracy": 4,
    #   "completeness": 3,
    #   "clarity": 5,
    #   "comment": "Câu trả lời đúng nhưng chưa đầy đủ"
    # }