# app.py
import os
import io
import uuid
from pathlib import Path
import sqlite3
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pydub import AudioSegment
from openai import OpenAI

from agent import (
    get_questions_for_web,
    get_example_and_intent_for_web,
    get_counts_for_web,
    get_fluency_for_web,
    get_logic_score_for_web,
    get_summary_and_best_for_web,
)

# ------------------------------------------------
# 기본 설정
# ------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "fluency.db"

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
WAV_DIR = DATA_DIR / "wav"

for d in (DATA_DIR, UPLOAD_DIR, WAV_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI()

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# 1. ARD (예술·디자인·창의)
# 2. BM (비즈니스·경영)
# 3. ICT (ICT·정보통신)
# 4. MM (제조·기계·생산)
# 5. PS (공공·서비스·교육)
# 6. RND (연구개발(R&D))
# 7. SM (세일즈·마케팅)
# 카테고리: id(정수) + label
CATEGORIES = [
    {"id": 1, "label": "ARD (예술·디자인·창의)"},
    {"id": 2, "label": "BM (비즈니스·경영)"},
    {"id": 3, "label": "ICT (ICT·정보통신)"},
    {"id": 4, "label": "MM (제조·기계·생산)"},
    {"id": 5, "label": "PS (공공·서비스·교육)"},
    {"id": 6, "label": "RND (연구개발 R&D)"},
    {"id": 7, "label": "SM (세일즈·마케팅)"},
]

# OpenAI Whisper
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WHISPER_MODEL = os.getenv("WHISPER_MODEL_NAME", "whisper-1")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ====================================================
# 1. 오디오 처리 유틸
# ====================================================

async def process_audio_upload(upload_file: UploadFile, prefix: str) -> Dict[str, Any]:
    """
    1) 업로드된 파일 내용을 메모리에 읽고
    2) data/uploads/ 에 원본 파일 그대로 저장
    3) pydub + ffmpeg 로 16kHz mono WAV 로 변환 → data/wav/
    4) wav_path, sample_rate(16000), duration_sec 반환
    """
    contents = await upload_file.read()
    ext = Path(upload_file.filename).suffix.lower() or ".bin"
    uid = uuid.uuid4().hex

    # 원본 저장
    orig_name = f"{prefix}_{uid}{ext}"
    orig_path = UPLOAD_DIR / orig_name
    with open(orig_path, "wb") as f:
        f.write(contents)

    # pydub 로 16kHz mono WAV 변환
    audio_seg = AudioSegment.from_file(io.BytesIO(contents))
    audio_seg = audio_seg.set_frame_rate(16000).set_channels(1)

    wav_name = f"{prefix}_{uid}.wav"
    wav_path = WAV_DIR / wav_name
    audio_seg.export(wav_path, format="wav")

    duration_sec = len(audio_seg) / 1000.0  # ms → sec
    sample_rate = 16000

    return {
        "original_path": str(orig_path),
        "wav_path": str(wav_path),
        "sample_rate": sample_rate,
        "duration_sec": duration_sec,
    }


def transcribe_wav_with_whisper(wav_path: str, language: str = "ko") -> str:
    """
    16kHz mono WAV 파일을 Whisper에 보내 텍스트를 얻는다.
    지원 포맷: mp3, mp4, mpeg, mpga, m4a, wav, webm 등
    (여기서는 변환된 wav_path 사용)
    """
    if openai_client is None:
        print("Whisper: OPENAI_API_KEY 가 설정되지 않았습니다.")
        return ""

    try:
        with open(wav_path, "rb") as f:
            resp = openai_client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=f,
                response_format="text",
                language=language,  # 한국어면 "ko", 자동 인식 원하면 생략 가능
            )
        # response_format="text" → resp 는 문자열
        return resp
    except Exception as e:
        print("Whisper transcription error:", e)
        return ""


# ====================================================
# 2. DB 초기화 및 헬퍼
# ====================================================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            category TEXT,
            keyword TEXT,
            question TEXT,
            example_answer TEXT,
            question_intent TEXT,
            -- 1차 발화 오디오/텍스트
            first_audio_path TEXT,
            first_audio_duration REAL,
            first_transcript TEXT,
            first_stutter_count INTEGER,
            first_filler_count INTEGER,
            first_fluency_score REAL,
            logic_score REAL,
            summary TEXT,
            model_answer TEXT,
            -- 2차 발화 오디오/텍스트
            second_audio_path TEXT,
            second_audio_duration REAL,
            second_transcript TEXT,
            second_stutter_count INTEGER,
            second_filler_count INTEGER,
            second_fluency_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.commit()
    conn.close()


def insert_first_attempt(
    username: str,
    category: str,
    keyword: str,
    question: str,
    example_answer: str,
    question_intent: str,
    first_audio_path: str,
    first_audio_duration: float,
    first_transcript: str,
    first_stutter_count: int,
    first_filler_count: int,
    first_fluency_score: float,
    logic_score: float,
    summary: str,
    model_answer: str,
) -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO attempts (
            username, category, keyword, question,
            example_answer, question_intent,
            first_audio_path, first_audio_duration, first_transcript,
            first_stutter_count, first_filler_count, first_fluency_score,
            logic_score, summary, model_answer
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            username, category, keyword, question,
            example_answer, question_intent,
            first_audio_path, first_audio_duration, first_transcript,
            first_stutter_count, first_filler_count, first_fluency_score,
            logic_score, summary, model_answer,
        ),
    )
    conn.commit()
    attempt_id = cur.lastrowid
    conn.close()
    return attempt_id


def update_second_attempt(
    attempt_id: int,
    second_audio_path: str,
    second_audio_duration: float,
    second_transcript: str,
    second_stutter_count: int,
    second_filler_count: int,
    second_fluency_score: float,
):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE attempts
        SET second_audio_path = ?,
            second_audio_duration = ?,
            second_transcript = ?,
            second_stutter_count = ?,
            second_filler_count = ?,
            second_fluency_score = ?
        WHERE id = ?
        """,
        (
            second_audio_path,
            second_audio_duration,
            second_transcript,
            second_stutter_count,
            second_filler_count,
            second_fluency_score,
            attempt_id,
        ),
    )
    conn.commit()
    conn.close()


def get_all_first_scores() -> List[float]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT first_fluency_score FROM attempts WHERE first_fluency_score IS NOT NULL")
    rows = cur.fetchall()
    conn.close()
    return [float(r[0]) for r in rows]


def get_all_second_scores() -> List[float]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT second_fluency_score FROM attempts WHERE second_fluency_score IS NOT NULL")
    rows = cur.fetchall()
    conn.close()
    return [float(r[0]) for r in rows]


def get_top_first_results(limit: int = 10) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT username, first_fluency_score, category, keyword, created_at
        FROM attempts
        WHERE first_fluency_score IS NOT NULL
        ORDER BY first_fluency_score DESC, created_at ASC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    return [
        {
            "username": r[0],
            "score": float(r[1]),
            "category": r[2],
            "keyword": r[3],
            "created_at": r[4],
        }
        for r in rows
    ]


def get_top_second_results(limit: int = 10) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT username, second_fluency_score, category, keyword, created_at
        FROM attempts
        WHERE second_fluency_score IS NOT NULL
        ORDER BY second_fluency_score DESC, created_at ASC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    return [
        {
            "username": r[0],
            "score": float(r[1]),
            "category": r[2],
            "keyword": r[3],
            "created_at": r[4],
        }
        for r in rows
    ]


def get_first_score_by_id(attempt_id: int) -> Optional[float]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT first_fluency_score FROM attempts WHERE id = ?",
        (attempt_id,),
    )
    row = cur.fetchone()
    conn.close()
    if row is None or row[0] is None:
        return None
    return float(row[0])


def compute_rank_and_distribution(score: float, scores: List[float]):
    """
    점수 리스트에서 주어진 score의 등수/백분위/구간 분포 계산.
    """
    if not scores:
        return 1, 1, 100.0, []

    scores_sorted = sorted(scores, reverse=True)
    total = len(scores_sorted)

    rank = scores_sorted.index(score) + 1
    percentile = 100.0 * (1 - (rank - 0.5) / total)

    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    labels = ["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
    counts = [0] * (len(bins) - 1)

    for s in scores:
        for i in range(len(bins) - 1):
            if bins[i] <= s < bins[i + 1]:
                counts[i] += 1
                break

    distribution = [
        {"range": labels[i], "count": counts[i]}
        for i in range(len(labels))
    ]
    return rank, total, percentile, distribution


init_db()


# ====================================================
# 3. 라우트
# ====================================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "categories": CATEGORIES,
        },
    )


@app.post("/api/generate_questions")
async def generate_questions(
    category_id: int = Form(...),
    category_label: str = Form(...),
    keyword: str = Form(...),
):
    """
    카테고리 번호(category_id) + 키워드로 tool_1 호출.
    화면에는 category_label(예: '1.IT(정보,통신)')을 계속 사용.
    """
    questions = get_questions_for_web(category_id, keyword)
    return {"questions": questions}


@app.post("/api/generate_example_intent")
async def generate_example_intent(
    question_index: int = Form(...),
):
    """
    선택된 질문 번호(1~5)를 tool_2에 넘겨 예시답변 + 질문의도 생성.
    """
    result = get_example_and_intent_for_web(question_index)
    return result  # {example_answer, question_intent}


@app.post("/api/first_evaluate")
async def first_evaluate(
    username: str = Form(...),
    category: str = Form(...),        # 예: '1.IT(정보,통신)'
    keyword: str = Form(...),
    question: str = Form(...),        # 실제 질문 텍스트
    example_answer: str = Form(...),
    question_intent: str = Form(...),
    audio: UploadFile = File(...),
):
    # 1) 오디오 저장 + 16kHz mono WAV 변환
    audio_info = await process_audio_upload(audio, prefix="first")
    wav_path = audio_info["wav_path"]

    # 2) Whisper로 transcript 획득
    transcript = transcribe_wav_with_whisper(wav_path, language="ko")

    # 3) tool_3: WAV + transcript + 질문 텍스트 → 말더듬/디플루언시 카운트
    counts = get_counts_for_web(audio_info, transcript, question)
    first_stutter_count = int(counts.get("stutter_count", 0))
    first_filler_count = int(counts.get("filler_count", 0))

    # 4) tool_4: 카운트 → 유창성 점수
    first_fluency_score = get_fluency_for_web(first_stutter_count, first_filler_count)

    # 5) tool_5: WAV + transcript + 질문 텍스트 → 논리성 점수
    logic_score = get_logic_score_for_web(audio_info, transcript, question)

    # 6) tool_6: WAV + transcript + 질문 텍스트 → [강점/보완점/논리성 평가/수정 제안] + 모범답변
    summary_best = get_summary_and_best_for_web(audio_info, transcript, question)

    strengths = summary_best.get("strengths", "")
    weaknesses = summary_best.get("weaknesses", "")
    logic_eval = summary_best.get("logic_eval", "")
    suggestions = summary_best.get("suggestions", "")
    logic_score_text = summary_best.get("logic_score_text", "")
    best_answer = summary_best.get("best_answer", "")

    # DB에 저장할 summary 문자열 (강점/보완점/논리성/수정 제안/점수 묶어서 저장)
    summary_parts = []
    if strengths:
        summary_parts.append("[강점]\n" + strengths)
    if weaknesses:
        summary_parts.append("[보완점]\n" + weaknesses)
    if logic_eval:
        summary_parts.append("[논리성 평가]\n" + logic_eval)
    if suggestions:
        summary_parts.append("[수정 제안]\n" + suggestions)
    if logic_score_text:
        summary_parts.append("[점수]\n" + logic_score_text)
    summary_text = "\n\n".join(summary_parts)
    model_answer_text = best_answer

    # 7) DB 저장
    attempt_id = insert_first_attempt(
        username=username,
        category=category,
        keyword=keyword,
        question=question,
        example_answer=example_answer,
        question_intent=question_intent,
        first_audio_path=wav_path,
        first_audio_duration=audio_info["duration_sec"],
        first_transcript=transcript,
        first_stutter_count=first_stutter_count,
        first_filler_count=first_filler_count,
        first_fluency_score=first_fluency_score,
        logic_score=logic_score,
        summary=summary_text,
        model_answer=model_answer_text,
    )

    # 8) 1차 유창성 점수 기준 순위/분포/리더보드
    all_first_scores = get_all_first_scores()
    rank, total, percentile, distribution = compute_rank_and_distribution(
        first_fluency_score,
        all_first_scores,
    )
    leaderboard = get_top_first_results(limit=10)

    # 9) 프론트로 전달
    return {
        "attempt_id": attempt_id,
        "first_score": first_fluency_score,
        "first_rank": rank,
        "first_total": total,
        "first_percentile": percentile,
        "first_distribution": distribution,
        "first_leaderboard": leaderboard,
        "stutter_count": first_stutter_count,
        "filler_count": first_filler_count,
        "logic_score": logic_score,
        # 기존 요약/모범답변 필드 (DB/2차 발화용)
        "summary": summary_text,
        "model_answer": model_answer_text,
        "transcript": transcript,
        # 새 구조: 강점/보완점/논리성평가/수정제안/점수문장
        "strengths": strengths,
        "weaknesses": weaknesses,
        "logic_eval": logic_eval,
        "suggestions": suggestions,
        "logic_score_text": logic_score_text,
    }


@app.post("/api/second_evaluate")
async def second_evaluate(
    attempt_id: int = Form(...),
    username: str = Form(...),
    category: str = Form(...),        # 예: '1.IT(정보,통신)'
    keyword: str = Form(...),
    question: str = Form(...),
    model_answer: str = Form(...),
    audio: UploadFile = File(...),
):
    # 1) 오디오 저장 + 16kHz mono WAV 변환
    audio_info = await process_audio_upload(audio, prefix="second")
    wav_path = audio_info["wav_path"]

    # 2) Whisper transcript
    transcript = transcribe_wav_with_whisper(wav_path, language="ko")

    # 3) 두 번째 발화: 모범답변 읽기 → text_hint = model_answer
    counts = get_counts_for_web(audio_info, transcript, model_answer)
    second_stutter_count = int(counts.get("stutter_count", 0))
    second_filler_count = int(counts.get("filler_count", 0))

    # 4) tool_4: 카운트 → 2차 유창성 점수
    second_fluency_score = get_fluency_for_web(second_stutter_count, second_filler_count)

    # 5) DB 업데이트
    update_second_attempt(
        attempt_id=attempt_id,
        second_audio_path=wav_path,
        second_audio_duration=audio_info["duration_sec"],
        second_transcript=transcript,
        second_stutter_count=second_stutter_count,
        second_filler_count=second_filler_count,
        second_fluency_score=second_fluency_score,
    )

    # 6) 1차 점수 불러오기 + 향상 정도
    first_score = get_first_score_by_id(attempt_id) or 0.0
    improvement = second_fluency_score - first_score

    # 7) 2차 유창성 점수 기준 순위/분포/리더보드
    all_second_scores = get_all_second_scores()
    rank, total, percentile, distribution = compute_rank_and_distribution(
        second_fluency_score,
        all_second_scores,
    )
    leaderboard = get_top_second_results(limit=10)

    return {
        "second_score": second_fluency_score,
        "second_rank": rank,
        "second_total": total,
        "second_percentile": percentile,
        "second_distribution": distribution,
        "second_leaderboard": leaderboard,
        "stutter_count": second_stutter_count,
        "filler_count": second_filler_count,
        "first_score": first_score,
        "improvement": improvement,
        "transcript": transcript,
    }
