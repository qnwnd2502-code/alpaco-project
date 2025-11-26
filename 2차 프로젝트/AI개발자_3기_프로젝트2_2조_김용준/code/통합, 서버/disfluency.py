# # disfluency.py
# from __future__ import annotations

# import os
# from pathlib import Path
# from typing import Dict, List, Tuple

# import numpy as np
# import torch
# import torch.nn.functional as F
# import torchaudio
# import librosa  # ← 추가

# from transformers import (
#     AutoModelForTokenClassification,
#     AutoTokenizer,
#     Wav2Vec2ForSequenceClassification,
#     Wav2Vec2Processor,
#     # ↓ Whisper filler head용 추가
#     WhisperFeatureExtractor,
#     WhisperTokenizerFast,
#     WhisperProcessor,
#     WhisperForConditionalGeneration,
# )


# # =========================
# # 환경 / 경로 설정
# # =========================

# PROJECT_ROOT = Path(__file__).resolve().parent

# # ---- 텍스트 기반 디플루언시(BIO 태깅) 모델 ----
# MODEL_NAME = "klue/roberta-large"
# CKPT_PATH = PROJECT_ROOT / "checkpoints_kspon_disfluency" / "best_model.pt"

# # ---- 오디오 기반 말더듬(wav2vec2) 모델 ----
# #   → 여기에 wav2vec2 말더듬 모델 디렉토리를 두고,
# #     config.json, preprocessor_config.json, tokenizer 관련 파일, model.safetensors 를 넣어두면 됩니다.
# #   예: fluency_site/checkpoints_stutter_wav2vec2/ 아래에 6개 파일 복사
# STUTTER_MODEL_DIR = os.getenv(
#     "STUTTER_MODEL_DIR",
#     str(PROJECT_ROOT / "checkpoints_stutter_wav2vec2"),
# )

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # =========================
# # 텍스트 라벨 / 토크나이저 / 모델 로드
# # =========================

# LABEL_LIST = [
#     "O",
#     "B-F",
#     "I-F",
#     "B-R",
#     "I-R",
# ]
# LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
# ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# # F: disfluency, R: repetition (stutter)
# DISFLUENCY_LABELS = {"B-F", "I-F"}
# STUTTER_LABELS = {"B-R", "I-R"}

# # 토크나이저/모델 로드 (기본은 klue/roberta-large)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# base_model = AutoModelForTokenClassification.from_pretrained(
#     MODEL_NAME,
#     num_labels=len(LABEL_LIST),
#     id2label=ID2LABEL,
#     label2id=LABEL2ID,
# )

# if CKPT_PATH.exists():
#     state = torch.load(CKPT_PATH, map_location="cpu")
#     # state_dict만 저장된 형태라고 가정
#     if "state_dict" in state:
#         base_model.load_state_dict(state["state_dict"])
#     else:
#         base_model.load_state_dict(state)
#     print(f"[disfluency] Loaded fine-tuned weights from {CKPT_PATH}")
# else:
#     print(f"[disfluency] WARNING: CKPT not found: {CKPT_PATH}  → base RoBERTa will be used.")

# text_model = base_model.to(device)
# text_model.eval()

# # =========================
# # 오디오 기반 말더듬(wav2vec2) 모델 로드
# # =========================

# STUTTER_SAMPLE_RATE = 16000
# STUTTER_SEGMENT_SEC = 3.0   # 3초 단위 segment (inference_code 기준)
# STUTTER_HOP_SEC = 3.0       # 겹치지 않게 3초씩 잘라서 예측

# try:
#     stutter_processor = Wav2Vec2Processor.from_pretrained(STUTTER_MODEL_DIR)
#     stutter_model = Wav2Vec2ForSequenceClassification.from_pretrained(
#         STUTTER_MODEL_DIR
#     ).to(device)
#     stutter_model.eval()
#     print(f"[stutter] Loaded wav2vec2 stutter model from {STUTTER_MODEL_DIR}")
# except Exception as e:
#     stutter_processor = None
#     stutter_model = None
#     print(f"[stutter] WARNING: failed to load stutter model from {STUTTER_MODEL_DIR}: {e}")


# # =========================
# # 유틸 함수들 (텍스트)
# # =========================

# def _tokenize_with_offsets(text: str):
#     """
#     단순 예시: whitespace 기준으로 토큰 분할 후,
#     BERT 입력 토큰과 매핑을 만들 수 있다.
#     실제론 더 정교한 전처리가 필요할 수 있음.
#     """
#     words = text.split()
#     encoding = tokenizer(
#         words,
#         is_split_into_words=True,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#     )
#     return encoding, words


# def _decode_labels(pred_ids: List[int]) -> List[str]:
#     return [ID2LABEL.get(i, "O") for i in pred_ids]


# # =========================
# # 유틸 함수들 (오디오: wav2vec2용)
# # =========================

# def _load_wav_mono_16k(path: str, target_sr: int = 16000):
#     """
#     WAV 파일을 mono, 16kHz로 로드해서 [1, T] torch.Tensor(float32)로 반환.
#     torchaudio.load 대신 librosa.load를 사용해서 torchcodec 의존성을 피한다.
#     """
#     # librosa: mono=True → (T,) float32, sr=None → 원본 샘플레이트 유지
#     y, sr = librosa.load(path, sr=None, mono=True)

#     # 샘플레이트가 다르면 16k로 리샘플
#     if sr != target_sr:
#         y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
#         sr = target_sr

#     # [T] → [1, T] torch.Tensor(float32)
#     wav = torch.from_numpy(y).float().unsqueeze(0)  # [1, T]

#     return wav, sr



# def _segment_audio(
#     wav: torch.Tensor,
#     sample_rate: int = STUTTER_SAMPLE_RATE,
#     segment_sec: float = STUTTER_SEGMENT_SEC,
#     hop_sec: float = STUTTER_HOP_SEC,
# ) -> List[torch.Tensor]:
#     """
#     waveform 을 segment_sec 단위로 잘라서 리스트로 반환.
#     hop_sec 만큼 stride.
#     """
#     seg_len = int(sample_rate * segment_sec)
#     hop_len = int(sample_rate * hop_sec)

#     if len(wav) <= seg_len:
#         return [wav]

#     segments: List[torch.Tensor] = []
#     for start in range(0, len(wav) - seg_len + 1, hop_len):
#         end = start + seg_len
#         segments.append(wav[start:end])
#     return segments


# # =========================
# # 메인 1: transcript에서 말더듬/디플루언시 카운트 (텍스트 모델)
# # =========================

# def count_disfluency_from_transcript(text: str) -> Dict[str, int]:
#     """
#     text(발화 스크립트)를 받아서
#     - 말더듬(stutter) 토큰 수
#     - 디플루언시(간투사 등) 토큰 수
#     를 반환.
#     (여기서는 RoBERTa BIO 태깅 결과만 사용)
#     """
#     text = text.strip()
#     if not text:
#         return {"stutter_count": 0, "disfluency_tag_count": 0}

#     encoding, words = _tokenize_with_offsets(text)
#     encoding = {k: v.to(device) for k, v in encoding.items()}

#     with torch.no_grad():
#         outputs = text_model(**encoding)
#         logits = outputs.logits  # [batch, seq_len, num_labels]
#         preds = torch.argmax(logits, dim=-1)[0].cpu().numpy().tolist()

#     # 첫 토큰은 [CLS], 마지막은 [SEP]라고 가정해서 제외
#     token_labels = _decode_labels(preds)[1 : len(words) + 1]

#     stutter_count = 0
#     disfluency_count = 0

#     for lab in token_labels:
#         if lab in STUTTER_LABELS:
#             stutter_count += 1
#         if lab in DISFLUENCY_LABELS:
#             disfluency_count += 1

#     return {
#         "stutter_count": int(stutter_count),
#         "disfluency_tag_count": int(disfluency_count),
#     }


# # =========================
# # 메인 2: WAV에서 말더듬(stutter) 카운트 (오디오 모델)
# # =========================

# def count_stutter_from_wav(wav_path: str) -> int:
#     """
#     wav 파일 경로를 받아 wav2vec2 기반 말더듬 모델로
#     말더듬이 포함된 segment 개수를 카운트해서 반환.

#     - class 0 : 정상(normal)
#     - class 1~N : 말더듬(stutter)의 세부 유형
#     이라고 가정하고, pred != 0 인 segment 를 모두 말더듬으로 카운트.
#     """
#     if not wav_path or not os.path.exists(wav_path):
#         return 0
#     if stutter_processor is None or stutter_model is None:
#         # 모델을 못 불러온 경우는 안전하게 0 리턴
#         return 0

#     wav = _load_wav_mono_16k(wav_path)
#     segments = _segment_audio(wav)

#     stutter_segments = 0
#     for seg in segments:
#         # seg: 1D Tensor
#         inputs = stutter_processor(
#             seg.numpy(),
#             sampling_rate=STUTTER_SAMPLE_RATE,
#             return_tensors="pt",
#         )
#         input_values = inputs.input_values.to(device)

#         with torch.no_grad():
#             logits = stutter_model(input_values).logits  # [1, num_labels]
#             probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
#             pred = int(np.argmax(probs))

#         # inference_code.ipynb 기준: class 0 = normal, 나머지 = stutter
#         if pred != 0:
#             stutter_segments += 1

#     return int(stutter_segments)

# # =========================
# # 3) Whisper + filler head: WAV → filler 토큰 카운트
# # =========================

# # 환경변수로 조정 가능
# WHISPER_BASE_MODEL = os.getenv("WHISPER_BASE_MODEL", "openai/whisper-small")
# WHISPER_PROJECT_ROOT = Path(
#     os.getenv("WHISPER_PROJECT_ROOT", str(PROJECT_ROOT / "whisper_only"))
# )
# WHISPER_CKPT_DIR = Path(
#     os.getenv(
#         "WHISPER_FILLER_CKPT_DIR",
#         # 예: whisper_only/checkpoints_stage2/checkpoint-3126
#         str(WHISPER_PROJECT_ROOT / "checkpoint-3126"),
#     )
# )


# class WhisperWithFillerHead(WhisperForConditionalGeneration):
#     """
#     Whisper decoder hidden state 위에 2-class filler head를 올린 구조.
#     fine-tuning 할 때 쓴 것과 동일한 구조여야 checkpoint 로드가 정상 동작합니다.
#     """

#     def __init__(self, config):
#         super().__init__(config)
#         self.filler_classifier = torch.nn.Linear(config.d_model, 2)
#         # hidden state를 항상 반환하도록 설정
#         self.config.output_hidden_states = True


# # Whisper processor / 모델 로드
# whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_BASE_MODEL)
# whisper_tokenizer = WhisperTokenizerFast.from_pretrained(
#     WHISPER_BASE_MODEL,
#     language="Korean",
#     task="transcribe",
# )
# whisper_processor = WhisperProcessor(
#     feature_extractor=whisper_feature_extractor,
#     tokenizer=whisper_tokenizer,
# )

# if WHISPER_CKPT_DIR.exists():
#     # fine-tuned 체크포인트가 있다면 그걸 사용
#     whisper_filler_model = WhisperWithFillerHead.from_pretrained(str(WHISPER_CKPT_DIR))
#     print(f"[disfluency] Loaded Whisper+filler model from: {WHISPER_CKPT_DIR}")
# else:
#     # 없으면 base whisper-small 에 랜덤 head (테스트용)
#     whisper_filler_model = WhisperWithFillerHead.from_pretrained(WHISPER_BASE_MODEL)
#     print(
#         f"[disfluency] WARNING: WHISPER_CKPT_DIR not found: {WHISPER_CKPT_DIR} "
#         f"→ base {WHISPER_BASE_MODEL} + random filler head 로 동작합니다."
#     )

# # 한국어 / transcribe 설정
# whisper_filler_model.generation_config.language = "korean"
# whisper_filler_model.generation_config.task = "transcribe"
# whisper_filler_model.generation_config.forced_decoder_ids = None

# whisper_filler_model.to(device)
# whisper_filler_model.eval()


# def transcribe_and_count_filler_from_wav(
#     wav_path: str,
#     max_length: int = 225,
# ) -> Dict[str, any]:
#     """
#     WAV 파일을 받아:
#       1) Whisper로 transcript 생성
#       2) decoder hidden state에 filler head를 적용해
#          filler 토큰 개수를 카운트

#     반환:
#       {
#         "text": str,                   # Whisper transcript
#         "filler_token_count": int,     # filler=1 로 예측된 토큰 개수
#         "tokens": [str, ...],          # Whisper 토큰 문자열
#         "filler_flags": [0/1, ...],    # 각 토큰이 filler인지 여부
#       }
#     """
#     if not wav_path or not os.path.exists(wav_path):
#         raise FileNotFoundError(f"Audio file not found: {wav_path}")

#     # 1) 오디오 로드 (16kHz mono)
#     audio, sr = librosa.load(wav_path, sr=16000, mono=True)

#     # 2) Whisper processor로 입력 생성
#     inputs = whisper_processor(
#         audio,
#         sampling_rate=16000,
#         return_tensors="pt",
#     )
#     input_features = inputs["input_features"].to(device)

#     # 3) transcript 생성
#     with torch.no_grad():
#         gen_out = whisper_filler_model.generate(
#             input_features=input_features,
#             max_length=max_length,
#             return_dict_in_generate=True,
#         )
#     generated_ids = gen_out.sequences  # (1, T)
#     text = whisper_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

#     # 4) decoder hidden state 추출 → filler head 통과
#     with torch.no_grad():
#         outputs = whisper_filler_model(
#             input_features=input_features,
#             decoder_input_ids=generated_ids,
#             use_cache=False,
#             output_hidden_states=True,
#         )

#     # 마지막 decoder layer hidden state
#     decoder_hiddens = outputs.decoder_hidden_states[-1]  # (B, T, H)
#     filler_logits = whisper_filler_model.filler_classifier(decoder_hiddens)  # (B, T, 2)
#     filler_pred = filler_logits.argmax(dim=-1).cpu().numpy()[0]  # (T,)

#     token_ids = generated_ids[0].cpu().numpy().tolist()
#     tokens = whisper_tokenizer.convert_ids_to_tokens(token_ids)

#     filler_token_count = int((filler_pred == 1).sum())

#     return {
#         "text": text,
#         "filler_token_count": filler_token_count,
#         "tokens": tokens,
#         "filler_flags": filler_pred.tolist(),
#     }

# if __name__ == "__main__":
#     sample = "저는 어 어제 학교에 갔는데 음 친구를 만났어요"
#     print("[TEXT ONLY] disfluency counts:", count_disfluency_from_transcript(sample))

#     test_wav = "/path/to/sample.wav"
#     if os.path.exists(test_wav):
#         info = transcribe_and_count_filler_from_wav(test_wav)
#         print("text:", info["text"])
#         print("filler_token_count:", info["filler_token_count"])

# # disfluency.py
# from __future__ import annotations

# import os
# from pathlib import Path
# from typing import Any, Dict, List, Tuple

# import numpy as np
# import torch
# import torch.nn.functional as F
# import librosa
# from transformers import (
#     Wav2Vec2ForSequenceClassification,
#     Wav2Vec2Processor,
#     WhisperFeatureExtractor,
#     WhisperTokenizerFast,
#     WhisperProcessor,
#     WhisperForConditionalGeneration,
# )

# # =========================
# # 0. 경로 및 장치 설정
# # =========================

# PROJECT_ROOT = Path(__file__).resolve().parent

# # wav2vec2 말더듬 모델 경로 (환경변수로도 변경 가능)
# STUTTER_MODEL_DIR = os.getenv(
#     "STUTTER_MODEL_DIR",
#     str(PROJECT_ROOT / "checkpoints_stutter_wav2vec2"),
# )

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # =========================
# # 1. Wav2Vec2 말더듬(stutter) 모델
# # =========================

# STUTTER_SAMPLE_RATE = 16000
# STUTTER_SEGMENT_SEC = float(os.getenv("STUTTER_SEGMENT_SEC", "3.0"))
# STUTTER_HOP_SEC = float(os.getenv("STUTTER_HOP_SEC", "1.5"))

# try:
#     if os.path.isdir(STUTTER_MODEL_DIR):
#         stutter_processor = Wav2Vec2Processor.from_pretrained(STUTTER_MODEL_DIR)
#         stutter_model = Wav2Vec2ForSequenceClassification.from_pretrained(
#             STUTTER_MODEL_DIR
#         )
#         stutter_model.to(device)
#         stutter_model.eval()
#         print(f"[stutter] Loaded wav2vec2 stutter model from {STUTTER_MODEL_DIR}")
#     else:
#         stutter_processor = None
#         stutter_model = None
#         print(f"[stutter] WARNING: STUTTER_MODEL_DIR not found: {STUTTER_MODEL_DIR}")
# except Exception as e:
#     stutter_processor = None
#     stutter_model = None
#     print(f"[stutter] Failed to load wav2vec2 stutter model: {e}")


# def _load_wav_mono_16k(path: str, target_sr: int = STUTTER_SAMPLE_RATE) -> torch.Tensor:
#     """
#     torchaudio 대신 librosa로 WAV 로드 (16kHz mono).
#     - 반환: 1D Tensor [T]
#     """
#     audio, sr = librosa.load(path, sr=target_sr, mono=True)
#     if sr != target_sr:
#         audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
#     wav = torch.from_numpy(audio).float()
#     return wav


# def _segment_audio(
#     wav: torch.Tensor,
#     sr: int = STUTTER_SAMPLE_RATE,
#     seg_sec: float = STUTTER_SEGMENT_SEC,
#     hop_sec: float = STUTTER_HOP_SEC,
# ) -> List[torch.Tensor]:
#     """
#     1D wav 텐서를 seg_sec 길이로 겹치게 잘라 List[Tensor] 로 반환.
#     """
#     if wav.ndim != 1:
#         wav = wav.view(-1)

#     seg_len = int(seg_sec * sr)
#     hop_len = int(hop_sec * sr)
#     if seg_len <= 0 or len(wav) == 0:
#         return []

#     segments: List[torch.Tensor] = []
#     for start in range(0, max(1, len(wav) - seg_len + 1), hop_len):
#         end = start + seg_len
#         if end > len(wav):
#             seg = wav[-seg_len:]
#         else:
#             seg = wav[start:end]
#         segments.append(seg)

#     return segments


# def count_stutter_from_wav(wav_path: str) -> int:
#     """
#     WAV 파일에서 wav2vec2 stutter 모델로 말더듬 segment 개수를 센다.

#     반환:
#         stutter_segments (int): 말더듬으로 분류된 segment 개수
#     """
#     if stutter_processor is None or stutter_model is None:
#         print("[stutter] Model not loaded, returning 0")
#         return 0

#     wav = _load_wav_mono_16k(wav_path, STUTTER_SAMPLE_RATE)
#     segments = _segment_audio(
#         wav, STUTTER_SAMPLE_RATE, STUTTER_SEGMENT_SEC, STUTTER_HOP_SEC
#     )
#     if not segments:
#         return 0

#     stutter_segments = 0
#     for seg in segments:
#         # librosa → numpy → processor 로 입력
#         inputs = stutter_processor(
#             seg.numpy(),
#             sampling_rate=STUTTER_SAMPLE_RATE,
#             return_tensors="pt",
#             padding=True,
#         )
#         input_values = inputs["input_values"].to(device)

#         with torch.no_grad():
#             logits = stutter_model(input_values).logits  # [1, num_labels]
#             probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
#             pred = int(np.argmax(probs))

#         # 보통 0번 라벨을 'fluent', 나머지를 'stutter' 로 가정
#         if pred != 0:
#             stutter_segments += 1

#     return int(stutter_segments)


# # =========================
# # 2. Whisper + Filler Head (디플루언시용)
# # =========================

# WHISPER_MODEL_NAME = os.getenv("WHISPER_BASE_MODEL", "openai/whisper-medium")
# WHISPER_CKPT_DIR = Path(
#     os.getenv(
#         "WHISPER_CKPT_DIR",
#         str(PROJECT_ROOT / "whisper_only" / "checkpoint-3126"),
#     )
# )


# class WhisperWithFillerHead(WhisperForConditionalGeneration):
#     """
#     Whisper decoder의 hidden state 위에 filler 여부(이진 분류) 헤드를 올린 모델.
#     """

#     def __init__(self, config):
#         super().__init__(config)
#         hidden_size = config.d_model
#         self.filler_classifier = torch.nn.Linear(hidden_size, 2)
#         self.filler_loss_fct = torch.nn.CrossEntropyLoss()

#     def forward(
#         self,
#         *args,
#         output_hidden_states: bool = False,
#         **kwargs,
#     ):
#         # filler head를 위해 decoder_hidden_states가 항상 나오도록 강제
#         return super().forward(*args, output_hidden_states=True, **kwargs)


# # Processor (base whisper)
# whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_MODEL_NAME)
# whisper_tokenizer = WhisperTokenizerFast.from_pretrained(
#     WHISPER_MODEL_NAME,
#     language="Korean",
#     task="transcribe",
# )
# whisper_processor = WhisperProcessor(
#     feature_extractor=whisper_feature_extractor,
#     tokenizer=whisper_tokenizer,
# )

# # Filler head가 올라간 fine-tuned checkpoint 로드
# try:
#     whisper_filler_model = WhisperWithFillerHead.from_pretrained(str(WHISPER_CKPT_DIR))
#     whisper_filler_model.to(device)
#     whisper_filler_model.eval()
#     print(f"[disfluency] Loaded Whisper+filler model from: {WHISPER_CKPT_DIR}")
# except Exception as e:
#     whisper_filler_model = None
#     print(
#         f"[disfluency] WARNING: failed to load Whisper+filler model from {WHISPER_CKPT_DIR}: {e}"
#     )


# def transcribe_and_count_filler_from_wav(
#     wav_path: str,
#     max_length: int = 225,
#     filler_threshold: float = 0.5,
# ) -> Dict[str, Any]:
#     """
#     Whisper+filler head 로
#     - 음성을 텍스트로 변환(transcribe)
#     - decoder hidden state 기반 filler 토큰 개수를 카운트.

#     반환 dict 예시:
#     {
#         "text": "...",                     # 전체 transcript
#         "words": [...],                    # 토큰 단위(단순 split 기준)
#         "filler_token_count": 5,           # filler 로 판정된 토큰 수
#         "filler_probs": [0.1, 0.9, ...],   # 각 토큰의 filler 확률
#         "filler_mask": [False, True, ...], # 토큰별 filler 여부
#     }
#     """
#     result: Dict[str, Any] = {
#         "text": "",
#         "words": [],
#         "filler_token_count": 0,
#         "filler_probs": [],
#         "filler_mask": [],
#     }

#     if whisper_filler_model is None:
#         print("[disfluency] Whisper+filler model not loaded, returning empty result")
#         return result

#     # 1) 오디오 로드 (Whisper 권장 샘플레이트 사용)
#     sr = whisper_feature_extractor.sampling_rate
#     audio, _ = librosa.load(wav_path, sr=sr, mono=True)

#     # 2) Whisper 입력 feature 생성
#     inputs = whisper_processor(
#         audio,
#         sampling_rate=sr,
#         return_tensors="pt",
#     )
#     input_features = inputs["input_features"].to(device)

#     # 3) greedy decode 로 transcript 생성
#     with torch.no_grad():
#         generated_ids = whisper_filler_model.generate(
#             input_features=input_features,
#             max_length=max_length,
#         )

#     # 4) 텍스트 디코딩
#     text = whisper_processor.tokenizer.batch_decode(
#         generated_ids, skip_special_tokens=True
#     )[0].strip()
#     result["text"] = text

#     if not text:
#         return result

#     # 단순히 공백 기준으로 토큰 나눔 (필요시 더 정교한 정렬 로직으로 교체 가능)
#     words = text.split()
#     result["words"] = words

#     # 5) decoder hidden state 추출을 위한 teacher-forcing 입력 준비
#     #   - generated_ids 를 한 토큰씩 밀어서 decoder_input_ids / labels 구성
#     decoder_input_ids = generated_ids[:, :-1]
#     labels = generated_ids[:, 1:]  # 현재는 사용하지 않지만, 필요하면 사용 가능

#     with torch.no_grad():
#         out = whisper_filler_model(
#             input_features=input_features,
#             decoder_input_ids=decoder_input_ids,
#         )
#         # decoder_hidden_states: List[Tensor], 마지막 layer 사용
#         hidden_states = out.decoder_hidden_states[-1]  # [B, T, d_model]

#     hidden = hidden_states[0]  # [T, d_model]
#     T = hidden.shape[0]
#     num_words = len(words)

#     # 단순 1:1 매핑(길이 맞추기 위한 잘라내기/복제)으로 word-level hidden 구성
#     if num_words <= T:
#         word_hidden = hidden[:num_words]
#     else:
#         pad = hidden[-1:].repeat(num_words - T, 1)
#         word_hidden = torch.cat([hidden, pad], dim=0)

#     logits = whisper_filler_model.filler_classifier(word_hidden)  # [num_words, 2]
#     probs = torch.softmax(logits, dim=-1)[:, 1]  # filler 클래스 확률
#     filler_mask = probs >= filler_threshold

#     filler_count = int(filler_mask.sum().item())

#     result["filler_token_count"] = filler_count
#     result["filler_probs"] = probs.cpu().tolist()
#     result["filler_mask"] = filler_mask.cpu().tolist()

#     return result


# __all__ = [
#     "count_stutter_from_wav",
#     "transcribe_and_count_filler_from_wav",
# ]

# disfluency.py

import os
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import librosa
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Processor,
    WhisperFeatureExtractor,
    WhisperTokenizerFast,
    WhisperProcessor,
    WhisperForConditionalGeneration,
)
import torch.nn as nn


# =========================
# 공통 설정
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# =========================
# 1) Wav2Vec2 기반 말더듬(stutter) 카운트
# =========================

STUTTER_MODEL_DIR = os.environ.get(
    "STUTTER_MODEL_DIR",
    str(PROJECT_ROOT / "stutter_step1_best"),
)

# 학습 코드와 동일한 설정
TARGET_SR = 16000
MAX_LEN = TARGET_SR * 3  # 3초

if not Path(STUTTER_MODEL_DIR).exists():
    raise FileNotFoundError(
        f"[stutter] STUTTER_MODEL_DIR not found: {STUTTER_MODEL_DIR}"
    )

print(f"[stutter] Loaded wav2vec2 stutter model from {STUTTER_MODEL_DIR}")

stutter_processor = Wav2Vec2Processor.from_pretrained(STUTTER_MODEL_DIR)
stutter_model = Wav2Vec2ForSequenceClassification.from_pretrained(
    STUTTER_MODEL_DIR
).to(device)
stutter_model.eval()


def _load_wav_mono_16k(path: str) -> np.ndarray:
    """
    librosa로 16kHz mono 로드
    """
    wav, sr = librosa.load(path, sr=16000, mono=True)
    wav = wav.astype(np.float32)
    return wav


# def _chunk_audio(
#     wav: np.ndarray,
#     sr: int = 16000,
#     win_sec: float = 1.5,
#     hop_sec: float = 0.75,
# ) -> List[np.ndarray]:
#     """
#     1.5초 윈도우, 0.75초 hop으로 슬라이딩 윈도우 시퀀스 생성
#     """
#     win = int(sr * win_sec)
#     hop = int(sr * hop_sec)

#     if len(wav) <= win:
#         return [wav]

#     segments = []
#     for start in range(0, len(wav) - win + 1, hop):
#         chunk = wav[start : start + win]
#         segments.append(chunk)

#     if not segments:
#         segments = [wav]

#     return segments

def _chunk_audio(
    wav: np.ndarray,
    sr: int = TARGET_SR,
) -> List[np.ndarray]:
    """
    학습 코드(fix_audio)와 동일하게
    - 16kHz 기준 3초(MAX_LEN) 단위로 앞에서부터 쪼갬
    - 마지막 조각이 3초보다 짧으면 뒤를 0으로 패딩
    - 각 segment 길이는 항상 MAX_LEN 샘플
    """
    # sr이 혹시 다르면 16k로 리샘플 (보통은 이미 16k)
    if sr != TARGET_SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)

    total_len = len(wav)
    if total_len == 0:
        return []

    segments: List[np.ndarray] = []
    step = MAX_LEN  # 3초씩 끊기 (겹치지 않게)

    for start in range(0, total_len, step):
        end = start + MAX_LEN
        seg = wav[start:end]

        if len(seg) > MAX_LEN:
            seg = seg[:MAX_LEN]
        elif len(seg) < MAX_LEN:
            # fix_audio와 동일하게 뒤를 0으로 패딩
            seg = np.pad(seg, (0, MAX_LEN - len(seg)), mode="constant")

        segments.append(seg.astype(np.float32))

    return segments



def count_stutter_from_wav(wav_path: str) -> int:
    """
    wav2vec2 기반 말더듬(stutter) 프레임 수 카운트
    - 세그먼트 단위로 stutter(1)로 예측된 segment 개수를 사용
    """
    wav = _load_wav_mono_16k(wav_path)
    segments = _chunk_audio(wav, sr=16000)

    if not segments:
        return 0

    inputs = stutter_processor(
        segments,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        logits = stutter_model(**inputs).logits  # [B, num_labels]

    preds = logits.argmax(dim=-1).cpu().numpy()  # 0: no stutter, 1: stutter
    stutter_frames = int((preds == 1).sum())
    return stutter_frames


# =========================
# 2) Whisper + filler head 기반 filler_count
# =========================

WHISPER_BASE_MODEL = "openai/whisper-small"
WHISPER_CKPT_DIR = os.environ.get(
    "WHISPER_FILLER_CKPT",
    str(PROJECT_ROOT / "whisper_only/checkpoint-3126"),
)

if not Path(WHISPER_CKPT_DIR).exists():
    raise FileNotFoundError(
        f"[disfluency] Whisper+filler checkpoint not found: {WHISPER_CKPT_DIR}"
    )

whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained(
    WHISPER_BASE_MODEL
)
whisper_tokenizer = WhisperTokenizerFast.from_pretrained(
    WHISPER_BASE_MODEL,
    language="Korean",
    task="transcribe",
)
whisper_processor = WhisperProcessor(
    feature_extractor=whisper_feature_extractor,
    tokenizer=whisper_tokenizer,
)


class WhisperWithFillerHead(WhisperForConditionalGeneration):
    """
    fine-tuning 시에 decoder hidden state 위에 filler head를 얹은 모델
    - filler_classifier: 각 토큰(디코더 타임스텝)에 대해 [non-filler, filler] 로짓 출력
    """

    def __init__(self, config):
        super().__init__(config)
        self.filler_classifier = nn.Linear(config.d_model, 2)
        self.filler_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)


whisper_model = WhisperWithFillerHead.from_pretrained(WHISPER_CKPT_DIR).to(device)
whisper_model.eval()
print(f"[disfluency] Loaded Whisper+filler model from: {WHISPER_CKPT_DIR}")


# def transcribe_and_count_filler_from_wav(
#     wav_path: str,
#     max_length: int = 225,
#     filler_threshold: float = 0.5,
# ) -> Dict[str, Any]:
#     """
#     Whisper + filler head 를 사용하여
#     - 음성을 텍스트로 변환
#     - decoder hidden state에 filler head를 적용하여
#       filler 확률이 threshold 이상인 토큰 수를 filler_count로 사용

#     반환 값 예시:
#     {
#         "text": "발화 내용 ...",
#         "filler_token_count": 5,
#     }
#     """
#     # 1) 오디오 로드 (16kHz)
#     waveform, sr = librosa.load(
#         wav_path,
#         sr=whisper_feature_extractor.sampling_rate,
#         mono=True,
#     )

#     # 2) Whisper 입력 특성 추출
#     input_features = whisper_processor.feature_extractor(
#         waveform,
#         sampling_rate=whisper_feature_extractor.sampling_rate,
#         return_tensors="pt",
#     ).input_features.to(device)

#     # 3) generate 호출 (hidden states도 함께 반환)
#     with torch.no_grad():
#         gen_out = whisper_model.generate(
#             input_features,
#             max_length=max_length,
#             output_hidden_states=True,
#             return_dict_in_generate=True,
#         )

#     # 4) 텍스트 디코딩
#     text = whisper_processor.batch_decode(
#         gen_out.sequences,
#         skip_special_tokens=True,
#     )[0]

#     # 5) decoder hidden states 에 filler head 적용
#     #    - gen_out.decoder_hidden_states: 리스트[레이어], 각 텐서는 [B, T, d_model]
#     decoder_hidden = gen_out.decoder_hidden_states[-1]  # 마지막 레이어 사용: [B, T, d_model]

#     with torch.no_grad():
#         filler_logits = whisper_model.filler_classifier(decoder_hidden)  # [B, T, 2]
#         filler_probs = torch.softmax(filler_logits, dim=-1)              # [B, T, 2]

#     # 6) filler 확률이 threshold 이상인 토큰 수를 카운트
#     filler_mask = filler_probs[..., 1] > filler_threshold  # label 1: filler
#     filler_token_count = int(filler_mask.sum().item())

#     return {
#         "text": text,
#         "filler_token_count": filler_token_count,
#     }
def transcribe_and_count_filler_from_wav(
    wav_path: str,
    max_length: int = 225,
    filler_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Whisper + filler head 를 사용하여
    - 음성을 텍스트로 변환
    - decoder hidden state에 filler head를 적용하여
      filler 확률이 threshold 이상인 토큰 수를 filler_token_count로 사용
    """
    # 1) 오디오 로드
    waveform, sr = librosa.load(
        wav_path,
        sr=whisper_feature_extractor.sampling_rate,
        mono=True,
    )

    # 2) 입력 feature 생성
    input_features = whisper_processor.feature_extractor(
        waveform,
        sampling_rate=whisper_feature_extractor.sampling_rate,
        return_tensors="pt",
    ).input_features.to(device)

    # 3) generate로 토큰 시퀀스 얻기
    with torch.no_grad():
        gen_out = whisper_model.generate(
            input_features,
            max_length=max_length,
            return_dict_in_generate=True,
        )

    sequences = gen_out.sequences  # [B, T]
    text = whisper_processor.batch_decode(
        sequences,
        skip_special_tokens=True,
    )[0]

    # 4) decoder hidden state를 얻기 위해 teacher-forcing forward 호출
    decoder_input_ids = sequences[:, :-1]  # 마지막 토큰은 label 용으로 버림

    with torch.no_grad():
        out = whisper_model(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            output_hidden_states=True,
        )
        decoder_hidden = out.decoder_hidden_states[-1]  # [B, T', d_model]

    # 5) filler head 적용
    with torch.no_grad():
        filler_logits = whisper_model.filler_classifier(decoder_hidden)   # [B, T', 2]
        filler_probs = torch.softmax(filler_logits, dim=-1)              # [B, T', 2]

    filler_mask = filler_probs[..., 1] > filler_threshold  # label 1: filler
    filler_token_count = int(filler_mask.sum().item())

    return {
        "text": text,
        "filler_token_count": filler_token_count,
    }

