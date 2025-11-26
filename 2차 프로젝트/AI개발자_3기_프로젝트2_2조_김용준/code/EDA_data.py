from datasets import load_dataset, load_from_disk, Audio
from datasets import DatasetDict
from pathlib import Path

# ---------------------
# 경로 설정
# ---------------------
PROJECT_ROOT = Path("/home/alpaco/jws/whisper_only")  # 너 환경에 맞게 수정
DATA_ROOT    = PROJECT_ROOT / "data"

RAW_DATA_DIR  = DATA_ROOT / "ksponspeech_raw_hf"   # HF에서 받은 원본 저장
PROC_DATA_DIR = DATA_ROOT / "ksponspeech_proc"     # 전처리 결과 저장

DATA_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------------
# 1) HF에서 KsponSpeech 로드 (스크립트 없는 parquet 버전)
# ---------------------
# Murple/ksponspeech  -> 스크립트 방식 (에러 발생)
# DragonLine/ksponspeech -> parquet 자동 변환 (에러 없음)
raw_kspon = load_dataset("DragonLine/ksponspeech")
print(raw_kspon)
# 예상: DatasetDict({
#   train: Dataset(...),
#   test:  Dataset(...),
#   valid: Dataset(...)
# })

# ---------------------
# 2) 컬럼 이름 정리 + 오디오 16kHz로 맞추기
# ---------------------
# DragonLine/ksponspeech는 text 컬럼 이름이 "transcripts"
if "transcripts" in raw_kspon["train"].column_names:
    raw_kspon = raw_kspon.rename_column("transcripts", "text")

# Whisper는 기본 16kHz 가정이므로 casting (이미 16k여도 안전)
raw_kspon = raw_kspon.cast_column("audio", Audio(sampling_rate=16_000))

print(raw_kspon["train"][0])
# 예:
# {
#   'audio': {'path': ..., 'array': np.array([...]), 'sampling_rate': 16000},
#   'text': '...문장...',
#   'id': ... (존재할 수도 있고 없을 수도 있음)
# }

# ---------------------
# 3) "원본(raw)" DatasetDict 통째로 디스크에 저장
# ---------------------
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
raw_kspon.save_to_disk(str(RAW_DATA_DIR))

# 나중에 다시 불러올 때:
# raw_kspon = load_from_disk(str(RAW_DATA_DIR))
# train / eval DatasetDict로 정리
kspon = DatasetDict({
    "train": raw_kspon["train"],
    "eval":  raw_kspon["valid"],
})
print(kspon)
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizerFast,
    WhisperProcessor,
)

MODEL_NAME = "openai/whisper-medium"

feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)

tokenizer = WhisperTokenizerFast.from_pretrained(
    MODEL_NAME,
    language="Korean",
    task="transcribe",
    add_prefix_space=True,   # ← 이 옵션 추가
)


processor = WhisperProcessor(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
)


print("sampling rate:", processor.feature_extractor.sampling_rate)
print("pad token id:", tokenizer.pad_token_id)
import re

FILLER_WORDS = ["음", "어", "아", "그러니까", "그게", "저", "뭐랄까", "에"]
FILLER_SET = set(FILLER_WORDS)

def normalize_korean_token(tok: str) -> str:
    tok = tok.strip()
    # 기본적인 문장부호 제거
    tok = re.sub(r"[.,?!…~]+", "", tok)
    return tok

def make_word_level_filler_labels(text: str):
    """
    text를 띄어쓰기 기준 단어로 나누고,
    각 단어가 filler인지 0/1로 라벨링.
    """
    words = text.strip().split()
    labels = []
    for w in words:
        nw = normalize_korean_token(w)
        labels.append(1 if nw in FILLER_SET else 0)
    return words, labels
import numpy as np

def prepare_example(batch):
    """
    batch: {"audio": ..., "text": ...}
    → "input_features", "labels", "filler_labels" 를 추가
    """

    # 1) 오디오 → log-Mel spectrogram
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]  # (80, T)

    # 2) 텍스트 → word 단위 분리 + word-level filler 라벨
    text = batch["text"].strip()
    words, word_filler = make_word_level_filler_labels(text)

    # 3) tokenizer: word 단위로 넣어서 word_ids() 사용
    enc = tokenizer(
        words,
        is_split_into_words=True,
        add_special_tokens=True,
        return_attention_mask=True,
    )

    input_ids = enc["input_ids"]
    word_ids = enc.word_ids()  # len == len(input_ids)

    # 4) token-level filler_labels 생성
    filler_labels = []
    for wid in word_ids:
        if wid is None:
            filler_labels.append(-100)      # special token
        else:
            filler_labels.append(int(word_filler[wid]))

    batch["labels"] = input_ids
    batch["filler_labels"] = filler_labels

    return batch
num_proc = 4  # CPU 상황에 맞게 조절

kspon_proc = kspon.map(
    prepare_example,
    remove_columns=kspon["train"].column_names,  # "audio", "text", "id" 등 제거
    num_proc=num_proc,
)

print(kspon_proc)
print(kspon_proc["train"][0].keys())
# → dict_keys(['input_features', 'labels', 'filler_labels'])

# 전처리된 DatasetDict 저장
PROC_DATA_DIR.mkdir(parents=True, exist_ok=True)
kspon_proc.save_to_disk(str(PROC_DATA_DIR))

# 나중에 다시 쓸 때:
# from datasets import load_from_disk
# kspon_proc = load_from_disk(str(PROC_DATA_DIR))
