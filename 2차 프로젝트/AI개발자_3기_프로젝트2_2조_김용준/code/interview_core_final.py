# 수정사항 1. 질문 top5 만들떄 유사한게 질문이 너무 많아서 TOP-K 방식으로 필터링 코드 추가했습니다
# 수정사항 2. LLM쪽 프롬프트 수정좀 했습니다

import os
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================
# 0. 환경설정 & OpenAI 클라이언트
# =========================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# 🔐 API 키는 환경변수로 관리 (서버에서 설정)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("[경고] OPENAI_API_KEY 환경변수가 설정되지 않았습니다. LLM 호출 시 에러가 날 수 있습니다.")

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================================
# 1. CSV 로드
# =========================================

# ⚠ 여기 경로는 서버 경로에 맞게 바꿔줄 것
CSV_PATH = "/home/alpaco/kyj/Rnn/training_labeling_full.csv"

print("CSV 로드 중...")
df = pd.read_csv(CSV_PATH)

# --- 질문 컬럼 자동 탐색 ---
q_candidates = [c for c in df.columns if "question" in c.lower()]
if not q_candidates:
    raise ValueError("질문 컬럼(이름에 'question' 포함)이 없습니다.")
QUESTION_COL = q_candidates[0]
print("QUESTION_COL:", QUESTION_COL)

# --- 요약(summary/clean) 컬럼 자동 탐색 ---
s_candidates = [c for c in df.columns if ("summary" in c.lower()) or ("clean" in c.lower())]
if not s_candidates:
    raise ValueError("요약 컬럼(이름에 'summary' 또는 'clean' 포함)이 없습니다.")
SUMMARY_COL = s_candidates[0]
print("SUMMARY_COL:", SUMMARY_COL)

# --- 카테고리 컬럼 ---
CATEGORY_COL = "dataSet_info_occupation"
if CATEGORY_COL not in df.columns:
    raise ValueError(f"{CATEGORY_COL} 컬럼이 CSV에 없습니다.")
print("CATEGORY_COL:", CATEGORY_COL)

# 결측치/문자 처리
df[QUESTION_COL] = df[QUESTION_COL].fillna("").astype(str)
df[SUMMARY_COL]  = df[SUMMARY_COL].fillna("").astype(str)
df[CATEGORY_COL] = df[CATEGORY_COL].fillna("").astype(str)

# 질문 정규화 (공백 정리)
df["_q_norm"] = (
    df[QUESTION_COL]
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

questions  = df["_q_norm"].tolist()
categories = df[CATEGORY_COL].tolist()
print(f"총 질문 개수: {len(questions)}")

CATEGORY_KO_NAME = {
    "ARD": "예술·디자인·창의",
    "BM":  "비즈니스·경영",
    "ICT": "ICT·정보통신",
    "MM":  "제조·기계·생산",
    "PS":  "공공·서비스·교육",
    "RND": "연구개발(R&D)",
    "SM":  "세일즈·마케팅",
}

def get_category_ko(code: str) -> str:
    code = str(code).strip()
    return CATEGORY_KO_NAME.get(code, code)


# =========================================
# 2. SBERT 임베딩 로드 / 계산
# =========================================

SBERT_MODEL_NAME = "jhgan/ko-sroberta-multitask"
print("문장 임베딩 모델 로드:", SBERT_MODEL_NAME)
sbert = SentenceTransformer(SBERT_MODEL_NAME, device=device)

# ⚠ 여기도 서버 경로에 맞게
EMB_PATH = "/home/alpaco/kyj/Rnn/question_embs.npy"

if os.path.exists(EMB_PATH):
    print(f"저장된 임베딩 로드 중... ({EMB_PATH})")
    q_embs = np.load(EMB_PATH)
    print("질문 임베딩 shape:", q_embs.shape)
else:
    print("질문 임베딩 계산 중...(최초 1회)")
    q_embs = sbert.encode(
        questions,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print("질문 임베딩 shape:", q_embs.shape)

    print(f"임베딩 npy로 저장: {EMB_PATH}")
    np.save(EMB_PATH, q_embs)


# =========================================
# 3. 키워드 확장 + 쿼리 문장 생성
# =========================================

TAG_PRESETS = {
    "기술": ["기술", "기술 보유", "보유 기술", "스킬", "기술 역량"],
    "기술보유": ["기술 보유", "보유 기술", "기술", "스킬"],
    "이직": ["이직", "퇴사", "퇴사 사유", "이직 사유"],
    "이직사유": ["이직 사유", "퇴사 사유", "이직"],
    "갈등": ["갈등", "충돌", "팀 갈등", "대인 갈등"],
    "협업": ["협업", "팀워크", "팀 프로젝트", "협력"],
    "성격": ["성격", "성향", "가치관"],
    "프로젝트": ["프로젝트", "성과"],
}

def expand_keywords(raw_keywords: List[str]) -> List[str]:
    """
    사용자가 넣은 키워드를, 사전(TAG_PRESETS)에 따라 확장.
    예) '이직사유' -> ['이직 사유','퇴사 사유','이직']
    """
    out = []
    for kw in raw_keywords:
        kw = kw.strip()
        if kw in TAG_PRESETS:
            out.extend(TAG_PRESETS[kw])
        else:
            out.append(kw)
    # 중복 제거 (순서 유지)
    return list(dict.fromkeys(out))

def build_query_sentence(raw_keywords: List[str]) -> str:
    expanded = expand_keywords(raw_keywords)
    if not expanded:
        return ""
    return f"{', '.join(expanded)} 에 대한 면접 질문"


# =========================================
# 4. 질문 검색 + 카테고리 가중치 + 중복 제거
# =========================================

def search_questions_semantic(
    raw_keywords: List[str],
    topk_questions: int = 5,            # 최종적으로 보여줄 개수
    selected_category: Optional[str] = None,
    category_boost: float = 1.3,        # 카테고리 가중치
    pool_size: int = 30,                # 1차 후보 개수
    dup_threshold: float = 0.96         # 질문끼리 너무 비슷하면 제거
) -> Dict:
    """
    웹 서버에서 직접 호출할 추천 질문 함수.

    반환:
    {
        "query_sentence": str,
        "hits": [
            {
              "idx": int,
              "question": str,
              "summary": str,
              "category": str,
              "category_ko": str,
              "score": float,
            }, ...
        ]
    }
    """

    # 1) 키워드 → 쿼리 문장
    query_sent = build_query_sentence(raw_keywords)
    if not query_sent:
        return {"query_sentence": "", "hits": []}

    # 2) 쿼리 임베딩
    q_vec = sbert.encode(
        [query_sent],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )  # shape: (1, dim)

    # 3) 쿼리 vs 전체 질문 유사도
    sims = cosine_similarity(q_vec, q_embs)[0]  # shape: (num_questions,)

    # 4) 카테고리 가중치
    if selected_category is not None:
        cat_series = df[CATEGORY_COL].astype(str)
        mask = (cat_series == selected_category).to_numpy()
        sims = np.where(mask, sims * category_boost, sims)

    # 5) 1차로 pool_size개 뽑기
    pool_size = min(pool_size, len(sims))
    pool_idx = np.argsort(-sims)[:pool_size]

    # 6) 2차: 중복 제거하며 최종 topk 선택 (질문 임베딩 dot product로 체크)
    selected_indices: List[int] = []
    for idx in pool_idx:
        if len(selected_indices) >= topk_questions:
            break

        if not selected_indices:
            selected_indices.append(idx)
            continue

        too_similar = False
        for sel_idx in selected_indices:
            sim_qq = float(np.dot(q_embs[idx], q_embs[sel_idx]))  # 코사인 유사도
            if sim_qq >= dup_threshold:
                too_similar = True
                break

        if too_similar:
            continue

        selected_indices.append(idx)

    # 부족하면 pool에서 채우기
    if len(selected_indices) < topk_questions:
        for idx in pool_idx:
            if idx not in selected_indices:
                selected_indices.append(idx)
            if len(selected_indices) >= topk_questions:
                break

    hits: List[Dict] = []
    for i in selected_indices:
        row = df.iloc[i]
        cat_code = str(row[CATEGORY_COL])
        hits.append({
            "idx": int(i),
            "question": row[QUESTION_COL],
            "summary": row[SUMMARY_COL],
            "category": cat_code,
            "category_ko": get_category_ko(cat_code),
            "score": float(sims[i]),
        })

    return {
        "query_sentence": query_sent,
        "hits": hits,
    }


# =========================================
# 5. 요약답변 자르기 helper (LLM 프롬프트용)
# =========================================

def truncate_summary(text: str, max_chars: int = 300) -> str:
    """요약답변이 너무 길면 ... 처리."""
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


# =========================================
# 6. LLM 호출 공통 함수
# =========================================

def call_llm(prompt: str) -> str:
    """gpt-4o-mini 호출 (한 번에 프롬프트 → 응답 텍스트)"""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content


# =========================================
# 7. LLM 프롬프트 1: 질문에 대한 "평가 포인트 안내"
# =========================================

def build_intro_prompt_for_question(
    question_text: str,
    category_code: Optional[str],
    keywords: List[str],
    summary_short: str
) -> str:
    cat_code = category_code if category_code is not None else ""
    cat_ko = get_category_ko(cat_code)
    kw_str = ", ".join(keywords) if keywords else "(키워드 없음)"

    prompt = f"""
당신은 한국어로 답변하는 면접 코치 AI입니다.

[직무 카테고리]
코드: {cat_code}
설명: {cat_ko}

[키워드]
{kw_str}

[면접 질문]
{question_text}

[이 질문에 대해 기존 지원자들이 자주 한 요약 예시 답변]
{summary_short}

요청사항:
0) 인사말은 생략하고 바로 본론부터 작성해 주세요.
1) 위 직무 카테고리와 질문을 기준으로, 면접관이 이 질문을 통해 확인하고 싶어하는
   핵심 평가 포인트를 1~5개 정도로 정리해 주세요. (개수는 자유롭게, 억지로 늘릴 필요는 없습니다.)
2) 각 평가 포인트는 불릿포인트(•)로 작성하고,
   "무엇을 보는지"와 "왜 중요한지"가 드러나도록 한두 문장씩 설명해 주세요.
3) 마지막에는 지원자가 답변을 준비할 때 체크해야 할 체크리스트를
   다시 한 번 불릿포인트(~하였는가? 형식)로 요약해 주세요.
4) 실제 면접 준비용 설명이므로, 전체적으로 친절하지만 단정한 톤의 한국어로 작성해 주세요.
""".strip()

    return prompt


def get_intro_feedback_for_hit(
    hit: Dict,
    raw_keywords: List[str],
    selected_category: Optional[str] = None,
    max_summary_chars: int = 300,
) -> Dict:
    """
    - search_questions_semantic() 의 hits 중 하나를 받아서
    - LLM에 '평가 포인트 안내'를 요청하고 응답을 돌려줌.

    반환:
    {
      "question": str,
      "summary_short": str,
      "category": str,
      "category_ko": str,
      "intro_feedback": str,
    }
    """
    question_text = hit["question"]
    cat_code = selected_category if selected_category is not None else hit["category"]
    summary_short = truncate_summary(hit["summary"], max_chars=max_summary_chars)

    prompt = build_intro_prompt_for_question(
        question_text=question_text,
        category_code=cat_code,
        keywords=raw_keywords,
        summary_short=summary_short,
    )
    intro_output = call_llm(prompt)

    return {
        "question": question_text,
        "summary_short": summary_short,
        "category": cat_code,
        "category_ko": get_category_ko(cat_code),
        "intro_feedback": intro_output,
    }


# =========================================
# 8. LLM 프롬프트 2: 사용자 답변 평가 + 점수 + 수정 제안 + 모범답변
# =========================================

def build_evaluation_prompt_for_answer(
    question_text: str,
    category_code: Optional[str],
    keywords: List[str],
    summary_short: str,
    user_answer: str
) -> str:
    cat_code = category_code if category_code is not None else ""
    cat_ko = get_category_ko(cat_code)
    kw_str = ", ".join(keywords) if keywords else "(키워드 없음)"

    prompt = f"""
당신은 한국어 면접 코칭을 담당하는 AI입니다.

[직무 카테고리]
코드: {cat_code}
설명: {cat_ko}

[키워드]
{kw_str}

[면접 질문]
{question_text}

[참고용 요약 예시 답변]
{summary_short}

[지원자 실제 답변]
{user_answer}

아래 세 가지 기준을 항상 함께 고려해서 평가해 주세요.
- 내용 적합성: 질문, 직무 카테고리, 키워드와 얼마나 관련성이 높은지
- 구조·논리성: 도입–전개–결론의 흐름, 인과관계, 말의 전개가 자연스러운지
- 구체성·사례 제시: 실제 경험, 상황–행동–결과(SAR)가 얼마나 구체적으로 드러나는지

요청사항:
1) 위 질문과 직무 관점에서, 지원자의 답변을 평가해 주세요.
   - 강점(좋았던 부분)을 1~5개 정도로 정리 (개수는 자유롭게, 억지로 늘릴 필요는 없습니다.)
   - 보완점(아쉬운 부분)을 1~5개 정도로 정리 (개수는 자유롭게, 억지로 늘릴 필요는 없습니다.)
   - 답변의 논리성(구조, 흐름, 인과관계 등)에 대한 평가를 한 단락으로 작성해 주세요.
2) 위 세 가지 기준(내용 적합성, 구조·논리성, 구체성·사례)을 종합하여,
   이 답변의 완성도를 10점 만점 기준으로 점수화해 주세요.
   - 예: "총점: 7.5 / 10"
3) 위 평가를 바탕으로, 지원자가 실제 면접에서 사용할 수 있도록
   지원자의 답변을 토대로 한 "수정 제안"을 구체적으로 작성해 주세요.
   - 어떤 내용을 추가/삭제/순서 변경하면 좋을지 제안해 주세요.
4) 마지막으로, 사용자가 처음에 선택했던 질문에 대해
   {cat_ko} 직무에 어울리고, {kw_str} 키워드가 자연스럽게 녹아 있는
   '모범 답변'을 5~10문장 분량으로 작성해 주세요.
   - 기존 지원자의 답변과 완전히 동일하지 않도록, 새로운 표현을 사용해 주세요.

5) 전체 출력은 아래 구조를 따라 주세요:

[강점]
- ...

[보완점]
- ...

[논리성 평가]
... (한두 단락)

[점수]
총점: X.X / 10

[수정 제안]
- ...

[모범 답변]
...

6) 모든 내용은 자연스러운 한국어 존댓말로 작성해 주세요.
""".strip()

    return prompt


def evaluate_answer_for_hit(
    hit: Dict,
    raw_keywords: List[str],
    user_answer: str,
    selected_category: Optional[str] = None,
    max_summary_chars: int = 300,
) -> Dict:
    """
    - 사용자가 선택한 질문(hit) + 실제 답변을 받아
    - LLM으로 평가 + 점수 + 수정 제안 + 모범답변 생성.

    반환:
    {
      "question": str,
      "summary_short": str,
      "category": str,
      "category_ko": str,
      "evaluation": str,
    }
    """
    question_text = hit["question"]
    cat_code = selected_category if selected_category is not None else hit["category"]
    summary_short = truncate_summary(hit["summary"], max_chars=max_summary_chars)

    prompt = build_evaluation_prompt_for_answer(
        question_text=question_text,
        category_code=cat_code,
        keywords=raw_keywords,
        summary_short=summary_short,
        user_answer=user_answer,
    )
    eval_output = call_llm(prompt)

    return {
        "question": question_text,
        "summary_short": summary_short,
        "category": cat_code,
        "category_ko": get_category_ko(cat_code),
        "evaluation": eval_output,
    }


# =========================================
# 9. (선택) 카테고리 목록 제공 helper
# =========================================

def get_category_list() -> List[Dict]:
    """
    프론트에서 '카테고리 드롭다운' 만들 때 쓰기 좋은 helper.
    """
    unique_cats = sorted(df[CATEGORY_COL].astype(str).unique())
    return [
        {"code": code, "name": get_category_ko(code)}
        for code in unique_cats
    ]
