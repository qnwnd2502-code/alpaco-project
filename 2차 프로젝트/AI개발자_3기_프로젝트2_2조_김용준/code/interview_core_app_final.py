# app.py

from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from interview_core import (
    search_questions_semantic,
    truncate_summary,
    build_intro_prompt_for_question,
    build_evaluation_prompt_for_answer,
    call_llm,
    get_category_ko,
)

app = FastAPI()


# ---------- 요청/응답 스키마 ----------

class SearchRequest(BaseModel):
    category: Optional[str] = None   # 예: "ICT"
    keywords: List[str]              # 예: ["기술보유", "협업"]


class IntroRequest(BaseModel):
    category: Optional[str] = None   # 선택된 카테고리 코드
    keywords: List[str]
    question: str
    summary: str


class EvalRequest(BaseModel):
    category: Optional[str] = None
    keywords: List[str]
    question: str
    summary: str
    user_answer: str


# ---------- 엔드포인트 ----------

@app.post("/search")
def search(req: SearchRequest):
    """
    카테고리 + 키워드 → 의미기반 유사도 Top-N 질문 추천
    """
    query_sentence, hits = search_questions_semantic(
        raw_keywords=req.keywords,
        topk_questions=5,
        selected_category=req.category,
        category_boost=1.3,
    )

    # 프론트에서 쓰기 좋은 형태로 그대로 반환
    return {
        "query": query_sentence,
        "hits": hits,   # [{idx,question,summary,category,category_ko,score}, ...]
    }


@app.post("/intro")
def intro(req: IntroRequest):
    """
    선택된 질문에 대해:
    - 요약답변 일부 + 카테고리 + 키워드 기반으로
    - LLM이 '평가 포인트/체크리스트' 설명을 생성
    """
    # 요약 길이 너무 길면 자르기
    summary_short = truncate_summary(req.summary, max_chars=300)

    prompt = build_intro_prompt_for_question(
        question_text=req.question,
        category_code=req.category,
        keywords=req.keywords,
        summary_short=summary_short,
    )
    llm_output = call_llm(prompt)

    return {
        "question": req.question,
        "category": req.category,
        "category_ko": get_category_ko(req.category) if req.category else None,
        "summary_short": summary_short,
        "llm_intro": llm_output,
    }


@app.post("/evaluate")
def evaluate(req: EvalRequest):
    """
    사용자가 작성한 실제 답변에 대해:
    - 강점 / 보완점 / 논리성 평가 / 점수 / 수정 제안 / 모범답변 생성
    """
    summary_short = truncate_summary(req.summary, max_chars=300)

    prompt = build_evaluation_prompt_for_answer(
        question_text=req.question,
        category_code=req.category,
        keywords=req.keywords,
        summary_short=summary_short,
        user_answer=req.user_answer,
    )
    llm_output = call_llm(prompt)

    return {
        "question": req.question,
        "category": req.category,
        "category_ko": get_category_ko(req.category) if req.category else None,
        "summary_short": summary_short,
        "user_answer": req.user_answer,
        "llm_evaluation": llm_output,
    }
