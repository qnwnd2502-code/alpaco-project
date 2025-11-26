import os
import json
from typing import List, Dict, Any
import re

from dotenv import load_dotenv

from disfluency import (
    count_stutter_from_wav,
    transcribe_and_count_filler_from_wav,
)

from interview_core import (
    search_questions_semantic,
    expand_keywords,
    get_intro_feedback_for_hit,
    build_intro_prompt_for_question,
    build_evaluation_prompt_for_answer,
    call_llm,
    get_category_list,
)

# .env 로드 (없으면 조용히 무시)
load_dotenv()

# ---------------- LangSmith 설정 ----------------
langsmith_key = os.getenv("LANGCHAIN_API_KEY")
if langsmith_key:
    os.environ["LANGCHAIN_API_KEY"] = langsmith_key

os.environ["LANGCHAIN_ENDPOINT"] = os.getenv(
    "LANGCHAIN_ENDPOINT",
    "https://api.smith.langchain.com"
)
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "Alphaco")
# ------------------------------------------------

from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ------------------------------------------------
# 면접 질문/평가 모듈(interview_core)와 연결용 전역 상태
# ------------------------------------------------

# "1.IT(정보,통신)" 같은 숫자 카테고리 → CSV 직무 코드 매핑
# CATEGORY_ID_TO_CODE = {
#     1: "ICT",  # IT(정보,통신)
#     2: "BM",   # 비즈니스·경영
#     3: "PS",   # 공공·서비스·교육
#     4: "ARD",  # 예술·디자인·창의
#     5: "RND",  # 연구개발(R&D)
#     6: "SM",   # 세일즈·마케팅
#     7: "MM",   # 제조·기계·생산
#     # 필요하면 수정/추가
# }
CATEGORY_ID_TO_CODE = {
    1: "ARD",  # 예술·디자인·창의
    2: "BM",   # 비즈니스·경영
    3: "ICT",  # ICT·정보통신
    4: "MM",   # 제조·기계·생산
    5: "PS",   # 공공·서비스·교육
    6: "RND",  # 연구개발(R&D)
    7: "SM",   # 세일즈·마케팅
}


# tool_1 검색 결과 / tool_2 선택 질문 / tool_5·6 평가 결과를
# 한번의 연습 세션 동안 캐시하기 위한 전역 변수
LAST_SEARCH_RESULT: Dict[str, Any] | None = None
LAST_SELECTED_QUESTION: Dict[str, Any] | None = None
LAST_EVAL_RESULT: Dict[str, Any] | None = None

# ============================================================
# 1. 실제 모델들을 감싸는 Python 함수
# ============================================================

def run_tool_1_select_questions(category_id: int, keyword: str) -> List[str]:
    """
    tool_1:
      - interview_core.search_questions_semantic 을 사용해서
        (category_id, keyword)에 맞는 질문 5개를 SBERT로 검색한다.
      - 반환: 질문 텍스트 5개 리스트
      - 내부적으로는 hits/키워드 등을 LAST_SEARCH_RESULT에 캐시해
        이후 tool_2, tool_5, tool_6에서 재사용한다.
    """
    global LAST_SEARCH_RESULT, LAST_SELECTED_QUESTION, LAST_EVAL_RESULT

    category_code = CATEGORY_ID_TO_CODE.get(category_id)
    # 키워드 한 개만 받을 수도 있고, 쉼표로 여러 개 받을 수도 있다고 가정
    raw_keywords = [kw.strip() for kw in keyword.split(",") if kw.strip()]
    if not raw_keywords and keyword.strip():
        raw_keywords = [keyword.strip()]

    result = search_questions_semantic(
        raw_keywords=raw_keywords,
        topk_questions=5,
        selected_category=category_code,
        category_boost=1.5,
    )
    hits = result.get("hits", [])

    # 전역 캐시 업데이트
    LAST_SEARCH_RESULT = {
        "category_id": category_id,
        "category_code": category_code,
        "keyword": keyword,
        "raw_keywords": raw_keywords,
        "result": result,   # search_questions_semantic 의 전체 dict
    }
    LAST_SELECTED_QUESTION = None
    LAST_EVAL_RESULT = None

    # 우리 파이프라인 상단에서는 "질문 텍스트 리스트"만 필요
    return [h["question"] for h in hits]


def run_tool_2_example_and_intent(question_index: int) -> Dict[str, str]:
    """
    tool_2:
      - tool_1에서 뽑았던 5개 질문 중 하나(번호 1~5)를 선택하면,
        해당 질문에 대한:
          * 예시 답변(example_answer)  → CSV summary 사용
          * 질문 의도(question_intent) → build_intro_prompt_for_question + call_llm 사용
      - tool_1 호출 이후에만 의미가 있다. (LAST_SEARCH_RESULT 의존)
    """
    global LAST_SELECTED_QUESTION

    if LAST_SEARCH_RESULT is None:
        raise RuntimeError("질문 목록이 없습니다. 먼저 tool_1을 호출해서 질문을 검색하세요.")

    hits = LAST_SEARCH_RESULT["result"].get("hits", [])
    if not (1 <= question_index <= len(hits)):
        raise ValueError(f"question_index={question_index} 는 유효하지 않습니다. (1~{len(hits)})")

    hit = hits[question_index - 1]
    question_text = hit["question"]
    summary_short = hit.get("summary_short") or hit.get("summary") or ""
    category_code = str(hit.get("category", "")).strip() or LAST_SEARCH_RESULT.get("category_code")
    keywords = LAST_SEARCH_RESULT.get("raw_keywords", [])

    # 1) 예시 답변: CSV summary 컬럼을 그대로 사용
    example_answer = hit.get("summary") or summary_short

    # 2) 질문 의도/평가 포인트: LLM 프롬프트 생성 후 call_llm
    prompt = build_intro_prompt_for_question(
        question_text=question_text,
        category_code=category_code,
        keywords=keywords,
        summary_short=summary_short,
    )
    question_intent = call_llm(prompt)

    # 이후 tool_5, tool_6에서 쓸 수 있도록 캐시
    LAST_SELECTED_QUESTION = {
        "index": question_index,
        "question_text": question_text,
        "summary_short": summary_short,
        "category_code": category_code,
        "keywords": keywords,
    }

    return {
        "example_answer": example_answer,
        "question_intent": question_intent,
    }


# def run_tool_3_count_disfluency(
#     wav_path: str,
#     sample_rate: int,
#     duration_sec: float,
#     transcript: str,
#     text_hint: str,
# ) -> Dict[str, int]:
#     """
#     tool_3:

#     - 입력:
#         - wav_path: 16kHz mono WAV 파일 경로
#         - sample_rate: 샘플레이트 (현재는 16,000 으로 고정 사용)
#         - duration_sec: 전체 길이 (sec)
#         - transcript: Whisper 로부터 얻은 한국어 발화 텍스트
#         - text_hint: 질문/모범답변 등 맥락 텍스트 (현재 로직에서는 사용하지 않음)

#     - 처리:
#         1) wav_path → 오디오 기반 말더듬(wav2vec2) 모델로 stutter_count 계산
#         2) wav_path → Whisper+filler head 로 filler 토큰 개수 계산
#            → 이 값을 disfluency_tag_count 로 사용

#     - 출력:
#         {
#             "stutter_count": <오디오 기반 말더듬 segment 개수>,
#             "disfluency_tag_count": <filler 토큰 개수>,
#         }
#     """

#     # 1) wav2vec2 말더듬 모델로 stutter count
#     stutter_count_audio = count_stutter_from_wav(wav_path)

#     # 2) Whisper+filler head 로 filler 토큰 카운트 → 디플루언시 카운트로 사용
#     disfluency_tag_count = 0
#     if wav_path:
#         try:
#             filler_result = transcribe_and_count_filler_from_wav(wav_path)
#             disfluency_tag_count = int(filler_result.get("filler_token_count", 0))
#         except Exception as e:
#             # filler 모델에 문제가 있으면 일단 0으로 두고 로그만 남김
#             print("[run_tool_3_count_disfluency] filler count error:", e)

#     return {
#         "stutter_count": int(stutter_count_audio),
#         "disfluency_tag_count": int(disfluency_tag_count),
#     }
# def run_tool_3_count_disfluency(
#     audio_info: Dict[str, Any],
#     transcript: str,
#     question: str,
# ) -> Dict[str, Any]:
#     """
#     tool_3: WAV + transcript 로 말더듬/디플루언시 카운트
#     - 말더듬(stutter): wav2vec2 오디오 모델
#     - filler(간투사/불필요한 말): Whisper filler head
#     """
#     wav_path = audio_info.get("path")
#     if not wav_path:
#         raise ValueError("audio_info['path'] 가 비어 있습니다.")

#     # 1) 말더듬 카운트 (wav2vec2)
#     stutter_count_audio = count_stutter_from_wav(wav_path)

#     # 2) filler 카운트 (Whisper + filler head)
#     filler_result = transcribe_and_count_filler_from_wav(wav_path)
#     filler_count = int(filler_result.get("filler_token_count", 0))

#     return {
#         "stutter_count": int(stutter_count_audio),
#         "filler_count": int(filler_count),
#     }
# agent.py

def run_tool_3_count_disfluency(
    wav_path: str,
    transcript: str,
    question: str,
) -> Dict[str, Any]:
    """
    tool_3: WAV + transcript 로 말더듬/디플루언시 카운트
    - 말더듬(stutter): wav2vec2 오디오 모델
    - filler(간투사/불필요한 말): Whisper filler head
    """
    if not wav_path:
        raise ValueError("wav_path 가 비어 있습니다.")

    # 1) 오디오 기반 말더듬 카운트
    stutter_count_audio = count_stutter_from_wav(wav_path)

    # 2) Whisper+filler head 기반 filler 카운트
    filler_result = transcribe_and_count_filler_from_wav(wav_path)
    filler_count = int(filler_result.get("filler_token_count", 0))

    return {
        "stutter_count": int(stutter_count_audio),
        "filler_count": int(filler_count),
    }




def run_tool_4_score_fluency(stutter_count: int, filler_count: int) -> float:
    """
    LLM(에이전트에 사용한 동일 모델)을 사용해
    말더듬/디플루언시 카운트로 0.0~1.0 유창성 점수를 계산한다.
    """
    # LLM에게 줄 프롬프트
    prompt = f"""
    당신은 한국어 발화의 유창성을 수치로 평가하는 전문가입니다.

    다음 정보를 바탕으로, 발화의 '유창성 점수'를 0.0~1.0 사이의 숫자로만 평가하세요.
    소수 셋째 자리까지 제시하세요.

    입력 정보:
    - 말더듬(stutter) 횟수: {stutter_count}
    - 주저어 단어(filler) 횟수: {filler_count}

    판단 가이드(참고):
    - 말더듬/주저어가 거의 없으면: 0.80 ~ 1.00
    - 약간 있지만 크게 방해되진 않으면: 0.60 ~ 0.80
    - 꽤 자주 나타나 발화가 다소 끊기면: 0.40 ~ 0.60
    - 매우 자주 나타나 발화를 이해하기 어렵다면: 0.00 ~ 0.40

    중요:
    - 최종 출력은 숫자만 써주세요 (예: 0.842)
    - 설명 문장, 단위(점) 등은 절대 쓰지 마세요.
    """.strip()

    try:
        # LangChain LLM 호출 (파일 하단에서 만든 llm 사용)
        resp = llm.invoke(prompt)
        text = getattr(resp, "content", str(resp)).strip()

        match = re.search(r"\d+\.\d+|\d+", text)
        if not match:
            raise ValueError(f"LLM output parse 실패: {text}")

        score = float(match.group(0))

        # 0.0~1.0 범위로 클리핑
        if score < 0.0:
            score = 0.0
        if score > 1.0:
            score = 1.0

        return float(round(score, 3))

    except Exception as e:
        # LLM이 이상한 값을 내거나 파싱 실패 시, 안전한 백업 규칙
        print("[run_tool_4_score_fluency] LLM error, fallback rule 사용:", e)

        base = 1.0
        penalty = 0.02 * stutter_count + 0.01 * filler_count
        score = max(0.0, base - penalty)
        return float(round(score, 3))


def _extract_score_from_eval_text(text: str) -> float:
    """
    평가 텍스트 안의 [점수] 블록에서
    '총점: X.X / 10' 형태를 찾아 0.0~1.0로 정규화해서 반환.
    """
    m = re.search(r"총점\s*[:：]\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*10", text)
    if not m:
        # 못 찾으면 0.75로 적당히 설정 (필요하면 조정)
        return 0.75
    score_10 = float(m.group(1))
    score_10 = max(0.0, min(score_10, 10.0))
    return float(round(score_10 / 10.0, 3))


def _parse_eval_sections(text: str) -> Dict[str, str]:
    """
    평가 텍스트에서
    [강점], [보완점], [논리성 평가], [점수], [수정 제안], [모범 답변]
    블록을 잘라서 dict로 반환.
    """
    def _block(t: str, start_tag: str, next_tags: List[str]) -> str:
        start = t.find(start_tag)
        if start == -1:
            return ""
        start += len(start_tag)
        # 다음 태그들 중 가장 먼저 나오는 위치까지 자르기
        end_candidates = [t.find(tag) for tag in next_tags if t.find(tag) != -1]
        end = min(end_candidates) if end_candidates else len(t)
        return t[start:end].strip()

    strengths = _block(
        text, "[강점]",
        ["[보완점]", "[논리성 평가]", "[점수]", "[수정 제안]", "[모범 답변]"],
    )
    weaknesses = _block(
        text, "[보완점]",
        ["[논리성 평가]", "[점수]", "[수정 제안]", "[모범 답변]"],
    )
    logic_eval = _block(
        text, "[논리성 평가]",
        ["[점수]", "[수정 제안]", "[모범 답변]"],
    )
    score_text = _block(
        text, "[점수]",
        ["[수정 제안]", "[모범 답변]"],
    )
    suggestions = _block(
        text, "[수정 제안]",
        ["[모범 답변]"],
    )
    best_answer = _block(
        text, "[모범 답변]",
        [],
    )

    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "logic_eval": logic_eval,
        "score_text": score_text,
        "suggestions": suggestions,
        "best_answer": best_answer,
    }


def run_tool_5_logic_score(
    wav_path: str,
    sample_rate: int,
    duration_sec: float,
    transcript: str,
    question: str,
) -> float:
    """
    tool_5:
      - build_evaluation_prompt_for_answer + call_llm 을 사용해서
        질문+답변에 대한 전체 평가 텍스트를 얻고,
      - 그 안의 [점수] 블록에서 '총점: X.X / 10'을 파싱해
        0.0~1.0 논리성 점수로 변환한다.
      - 전체 평가는 LAST_EVAL_RESULT에 캐시해서 tool_6에서 재사용.
    """
    global LAST_EVAL_RESULT

    # 1) 카테고리/키워드/요약 찾기
    if LAST_SELECTED_QUESTION and LAST_SELECTED_QUESTION.get("question_text") == question:
        category_code = LAST_SELECTED_QUESTION.get("category_code")
        keywords = LAST_SELECTED_QUESTION.get("keywords", [])
        summary_short = LAST_SELECTED_QUESTION.get("summary_short", "")
    elif LAST_SEARCH_RESULT:
        hits = LAST_SEARCH_RESULT["result"].get("hits", [])
        hit = next((h for h in hits if h.get("question") == question), None)
        if hit:
            category_code = str(hit.get("category", "")).strip()
            summary_short = hit.get("summary_short") or hit.get("summary") or ""
            keywords = LAST_SEARCH_RESULT.get("raw_keywords", [])
        else:
            category_code = None
            summary_short = ""
            keywords = []
    else:
        category_code = None
        summary_short = ""
        keywords = []

    # 2) LLM 프롬프트 생성 + 호출
    prompt = build_evaluation_prompt_for_answer(
        question_text=question,
        category_code=category_code,
        keywords=keywords,
        summary_short=summary_short,
        user_answer=transcript,
    )
    eval_text = call_llm(prompt)

    # 3) 섹션 파싱 + 점수 추출
    sections = _parse_eval_sections(eval_text)
    logic_score = _extract_score_from_eval_text(sections.get("score_text") or eval_text)

    # 4) 캐시 저장 (tool_6에서 재사용)
    LAST_EVAL_RESULT = {
        "eval_text": eval_text,
        "sections": sections,
        "logic_score": logic_score,
        "question": question,
        "transcript": transcript,
    }

    return logic_score


def run_tool_6_summary_and_best(
    wav_path: str,
    sample_rate: int,
    duration_sec: float,
    transcript: str,
    question: str,
) -> Dict[str, str]:
    """
    tool_6:
      - 논리성 평가에서 사용한 평가 텍스트(LAST_EVAL_RESULT)를 바탕으로
        [강점, 보완점, 논리성 평가, 수정 제안] + 모범 답변을 반환한다.
      - 기존 '발화 내용 요약' 대신 이 4가지 평가 요소를 제공.
    """
    global LAST_EVAL_RESULT

    # 아직 평가가 안 되었거나, 질문/답변이 안 맞으면 tool_5를 먼저 돌려서 평가를 만든다.
    if (
        LAST_EVAL_RESULT is None or
        LAST_EVAL_RESULT.get("question") != question or
        LAST_EVAL_RESULT.get("transcript") != transcript
    ):
        _ = run_tool_5_logic_score(
            wav_path=wav_path,
            sample_rate=sample_rate,
            duration_sec=duration_sec,
            transcript=transcript,
            question=question,
        )

    sections = LAST_EVAL_RESULT.get("sections", {})

    return {
        # 요약 대신 제공할 평가 요소들
        "strengths": sections.get("strengths", ""),
        "weaknesses": sections.get("weaknesses", ""),
        "logic_eval": sections.get("logic_eval", ""),
        "suggestions": sections.get("suggestions", ""),
        # 점수 문장 자체를 보여주고 싶으면 같이 보낼 수 있음
        "logic_score_text": sections.get("score_text", ""),
        # 두 번째 발화용 모범 답변
        "best_answer": sections.get("best_answer", ""),
    }


# ============================================================
# 2. LangChain Tool 래핑 (Agent용)
# ============================================================

@tool
def select_questions(category_id: int, keyword: str) -> str:
    """
    카테고리 번호(category_id)와 키워드를 사용해 연습 질문 5개를 선택한다.
    반환값: JSON 문자열, 예) ["질문1", "질문2", ...]
    """
    qs = run_tool_1_select_questions(category_id, keyword)
    return json.dumps(qs, ensure_ascii=False)


@tool
def generate_example_and_intent(question_index: int) -> str:
    """
    선택된 질문 번호(1~5)를 입력으로 받아,
    - 예시 답변(example_answer)
    - 질문 의도(question_intent)
    를 생성하여 JSON 문자열로 반환한다.
    """
    result = run_tool_2_example_and_intent(question_index)
    return json.dumps(result, ensure_ascii=False)


# @tool
# def count_disfluency_from_transcript(transcript: str, text_hint: str) -> str:
#     """
#     transcript와 text_hint를 사용해
#     말더듬/디플루언시 태그를 카운트하는 예시 Tool.
#     (웹 API에서는 get_counts_for_web를 직접 사용)
#     """
#     counts = run_tool_3_count_disfluency(
#         wav_path="",   # 이 Tool은 실제로는 사용하지 않는 예시용
#         sample_rate=16000,
#         duration_sec=0.0,
#         transcript=transcript,
#         text_hint=text_hint,
#     )
#     return json.dumps(counts, ensure_ascii=False)
@tool
def count_disfluency_from_transcript(transcript: str, text_hint: str) -> str:
    """
    transcript만 가지고 대략적인 filler/stutter를 추정하는 간단한 예시 Tool.
    (실제 웹 API에서는 get_counts_for_web를 사용해 오디오 기반으로 측정)
    """
    # 아주 단순한 휴리스틱 예시: 자주 등장하는 한국어 간투사들
    filler_candidates = ["음", "어", "그", "저", "그러니까", "뭐랄까"]
    filler_count = 0
    for w in filler_candidates:
        filler_count += transcript.count(w)

    # 텍스트만으로는 stutter(말더듬) 추정이 어려우므로 0으로 둠
    stutter_count = 0

    counts = {
        "stutter_count": int(stutter_count),
        "filler_count": int(filler_count),
    }
    return json.dumps(counts, ensure_ascii=False)



@tool
def score_fluency_from_counts(stutter_count: int, filler_count: int) -> float:
    """
    카운트 값으로 유창성 점수를 계산하는 Tool 예시.
    """
    return run_tool_4_score_fluency(stutter_count, filler_count)


@tool
def evaluate_logic_from_transcript(transcript: str, question: str) -> float:
    """
    transcript와 question으로 논리성 점수를 계산하는 Tool 예시.
    """
    return run_tool_5_logic_score(
        wav_path="",
        sample_rate=16000,
        duration_sec=0.0,
        transcript=transcript,
        question=question,
    )


@tool
def summarize_and_best_from_transcript(transcript: str, question: str) -> str:
    """
    transcript와 question으로 요약+모범답변을 생성하는 Tool 예시.
    """
    result = run_tool_6_summary_and_best(
        wav_path="",
        sample_rate=16000,
        duration_sec=0.0,
        transcript=transcript,
        question=question,
    )
    return json.dumps(result, ensure_ascii=False)


TOOLS = [
    select_questions,
    generate_example_and_intent,
    count_disfluency_from_transcript,
    score_fluency_from_counts,
    evaluate_logic_from_transcript,
    summarize_and_best_from_transcript,
]

SYSTEM_PROMPT = """
너는 발화 유창성·논리성 연습을 도와주는 AI Agent야.

사용 가능한 도구:
- select_questions(category_id, keyword): 카테고리 번호와 키워드로 연습 질문 5개 선택
- generate_example_and_intent(question_index): 질문 번호(1~5)로 예시 답변과 질문 의도 생성
- count_disfluency_from_transcript(transcript, text_hint): 발화 텍스트에서 말더듬/디플루언시 태그 카운트
- score_fluency_from_counts(stutter_count, filler_count): 카운트로 유창성 점수 계산
- evaluate_logic_from_transcript(transcript, question): 발화 텍스트의 논리성 점수 계산
- summarize_and_best_from_transcript(transcript, question): 발화를 평가하고 모범답변 생성

최종 답변은 한국어로 간단하고 명확하게 작성한다.
"""

llm = init_chat_model(
    model="gpt-4.1-nano",
    api_key=OPENAI_API_KEY,
    temperature=0.2,
)

agent = create_agent(
    model=llm,
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
)

# ============================================================
# 4. FastAPI에서 직접 호출하는 헬퍼 (웹 API용)
# ============================================================

def get_questions_for_web(category_id: int, keyword: str) -> List[str]:
    """
    카테고리 번호 + 키워드로 질문 5개 텍스트 리스트 반환.
    (tool_1)
    """
    return run_tool_1_select_questions(category_id, keyword)


def get_example_and_intent_for_web(question_index: int) -> Dict[str, str]:
    """
    질문 번호(1~5)를 입력으로 예시답변 + 질문의도 생성.
    (tool_2)
    """
    return run_tool_2_example_and_intent(question_index)


# def get_counts_for_web(
#     audio_info: Dict[str, Any],
#     transcript: str,
#     text_hint: str,
# ) -> Dict[str, int]:
#     """
#     audio_info: process_audio_upload() 가 반환한 dict
#         - wav_path
#         - sample_rate
#         - duration_sec
#     transcript: Whisper로 얻은 텍스트
#     text_hint: 질문/모범답변 등 맥락 텍스트
#     → tool_3 호출
#     """
#     return run_tool_3_count_disfluency(
#         wav_path=audio_info["wav_path"],
#         sample_rate=audio_info["sample_rate"],
#         duration_sec=audio_info["duration_sec"],
#         transcript=transcript,
#         text_hint=text_hint,
#     )
def get_counts_for_web(
    audio_info: Dict[str, Any],
    transcript: str,
    text_hint: str,
) -> Dict[str, int]:
    """
    audio_info: process_audio_upload() 가 반환한 dict
        - wav_path
        - sample_rate
        - duration_sec
    transcript: Whisper로 얻은 텍스트
    text_hint: 질문/모범답변 등 맥락 텍스트 (현재 run_tool_3에서는 질문으로 전달)
    """
    wav_path = audio_info["wav_path"]
    return run_tool_3_count_disfluency(
        wav_path=wav_path,
        transcript=transcript,
        question=text_hint,
    )



def get_fluency_for_web(stutter_count: int, filler_count: int) -> float:
    """
    tool_4: 카운트 → 유창성 점수
    """
    return run_tool_4_score_fluency(stutter_count, filler_count)


def get_logic_score_for_web(
    audio_info: Dict[str, Any],
    transcript: str,
    question: str,
) -> float:
    """
    tool_5: WAV + transcript → 논리성 점수
    """
    return run_tool_5_logic_score(
        wav_path=audio_info["wav_path"],
        sample_rate=audio_info["sample_rate"],
        duration_sec=audio_info["duration_sec"],
        transcript=transcript,
        question=question,
    )


def get_summary_and_best_for_web(
    audio_info: Dict[str, Any],
    transcript: str,
    question: str,
) -> Dict[str, str]:
    """
    tool_6: WAV + transcript → 평가 요약 + 모범답변
    """
    return run_tool_6_summary_and_best(
        wav_path=audio_info["wav_path"],
        sample_rate=audio_info["sample_rate"],
        duration_sec=audio_info["duration_sec"],
        transcript=transcript,
        question=question,
    )
