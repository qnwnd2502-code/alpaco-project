# # whisper_train.py

# import os
# from pathlib import Path
# from collections import Counter
# import json

# import torch
# import torch.nn as nn
# import numpy as np

# from datasets import load_from_disk
# from transformers import (
#     WhisperFeatureExtractor,
#     WhisperTokenizerFast,
#     WhisperProcessor,
#     WhisperForConditionalGeneration,
#     Seq2SeqTrainingArguments,
#     Seq2SeqTrainer,
# )
# from dataclasses import dataclass
# from typing import Any, Dict, List, Union
# import evaluate
# from datasets import Dataset
# from torch.utils.data import DataLoader  # ← filler eval용

# # =========================
# # 환경/경로 설정
# # =========================

# PROJECT_ROOT = Path("/home/alpaco/jws/whisper_only")
# DATA_ROOT    = PROJECT_ROOT / "data"
# PROC_DATA_DIR = DATA_ROOT / "ksponspeech_proc"

# META_DIR = PROJECT_ROOT / "meta"
# META_DIR.mkdir(parents=True, exist_ok=True)
# CLASS_WEIGHT_FILE = META_DIR / "class_weights.json"

# MODEL_NAME = "openai/whisper-small"  # multilingual

# # =========================
# # 1. Processor / Tokenizer 로드
# # =========================

# feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
# tokenizer = WhisperTokenizerFast.from_pretrained(
#     MODEL_NAME,
#     language="Korean",
#     task="transcribe",
# )
# processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# # =========================
# # 2. 전처리된 Kspon 데이터 로드
# # =========================

# print(f"[INFO] Loading preprocessed KsponSpeech from: {PROC_DATA_DIR}")
# kspon_proc = load_from_disk(str(PROC_DATA_DIR))
# print(kspon_proc)
# print(kspon_proc["train"][0].keys())  # 'input_features', 'labels', 'filler_labels'

# # =========================
# # 3. filler class weight 계산 (한 번만) + 재사용
# # =========================

# def count_filler_tokens(dataset):
#     counter = Counter()
#     for ex in dataset:
#         for y in ex["filler_labels"]:
#             if y == -100:
#                 continue
#             counter[int(y)] += 1
#     return counter

# if CLASS_WEIGHT_FILE.exists():
#     print(f"[INFO] Load class weights from {CLASS_WEIGHT_FILE}")
#     with open(CLASS_WEIGHT_FILE, "r", encoding="utf-8") as f:
#         w = json.load(f)
#     w0 = float(w["w0"])
#     w1 = float(w["w1"])
# else:
#     print("[INFO] Compute class weights from kspon_proc['train'] (first time only)")
#     counter = count_filler_tokens(kspon_proc["train"])
#     print("[INFO] filler token counts:", counter)

#     total = counter[0] + counter[1]
#     w0 = float(total / (2 * counter[0]))  # non-filler
#     w1 = float(total / (2 * counter[1]))  # filler

#     with open(CLASS_WEIGHT_FILE, "w", encoding="utf-8") as f:
#         json.dump({"w0": w0, "w1": w1}, f, ensure_ascii=False, indent=2)
#     print(f"[INFO] Saved class weights to {CLASS_WEIGHT_FILE}")

# print(f"[INFO] class weights: non-filler={w0:.4f}, filler={w1:.4f}")
# class_weights = [w0, w1]   # 리스트로 유지

# # =========================
# # 4. Whisper + filler 헤드 모델 정의
# # =========================

# class WhisperWithFillerHead(WhisperForConditionalGeneration):
#     def __init__(self, config, class_weights=None):
#         super().__init__(config)

#         # decoder hidden state → 2-class (non-filler / filler)
#         self.filler_classifier = nn.Linear(config.d_model, 2)

#         # 항상 hidden states 출력
#         self.config.output_hidden_states = True

#         # class_weights는 버퍼로 따로 들고 있지 않고,
#         # loss 안에 weight로만 넘긴다.
#         if class_weights is not None:
#             weight = torch.tensor(class_weights, dtype=torch.float32)
#             self.filler_loss_fct = nn.CrossEntropyLoss(
#                 ignore_index=-100,
#                 weight=weight,
#             )
#         else:
#             self.filler_loss_fct = nn.CrossEntropyLoss(
#                 ignore_index=-100,
#             )

#     def forward(
#         self,
#         input_features=None,
#         attention_mask=None,
#         decoder_input_ids=None,
#         decoder_attention_mask=None,
#         labels=None,
#         filler_labels=None,
#         **kwargs,
#     ):
#         # Trainer가 내부적으로 넘기는 num_items_in_batch 같은 인자는
#         # WhisperForConditionalGeneration.forward 에서는 지원하지 않으므로 제거
#         if "num_items_in_batch" in kwargs:
#             kwargs.pop("num_items_in_batch")

#         # Whisper 기본 forward (ASR loss 포함)
#         outputs = super().forward(
#             input_features=input_features,
#             attention_mask=attention_mask,
#             decoder_input_ids=decoder_input_ids,
#             decoder_attention_mask=decoder_attention_mask,
#             labels=labels,
#             **kwargs,
#         )

#         total_loss = outputs.loss
#         filler_logits = None
#         filler_loss = None

#         # decoder hidden states → filler head
#         if outputs.decoder_hidden_states is not None:
#             last_hidden = outputs.decoder_hidden_states[-1]       # (B, T, d_model)
#             filler_logits = self.filler_classifier(last_hidden)   # (B, T, 2)

#             if filler_labels is not None:
#                 B, T, C = filler_logits.shape
#                 filler_loss = self.filler_loss_fct(
#                     filler_logits.view(B * T, C),
#                     filler_labels.view(B * T),
#                 )
#                 if total_loss is not None:
#                     total_loss = total_loss + filler_loss
#                 else:
#                     total_loss = filler_loss

#         outputs.loss = total_loss
#         outputs.filler_logits = filler_logits
#         outputs.filler_loss = filler_loss
#         return outputs


# # =========================
# # 5. 초기 모델 로딩/저장 (한 번만 base → filler head 초기화)
# # =========================

# INIT_MODEL_DIR = PROJECT_ROOT / "init_whisper_small_filler"

# def has_init_weights(init_dir: Path) -> bool:
#     return (init_dir / "pytorch_model.bin").exists() or (init_dir / "model.safetensors").exists()

# if has_init_weights(INIT_MODEL_DIR):
#     print(f"[INFO] Loading initialized model with filler head from {INIT_MODEL_DIR}")
#     model = WhisperWithFillerHead.from_pretrained(
#         INIT_MODEL_DIR,
#         class_weights=class_weights,
#     )
# else:
#     print(f"[INFO] Loading base Whisper model from {MODEL_NAME} (init or recover)")
#     base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
#     config = base_model.config
#     config.use_cache = False

#     model = WhisperWithFillerHead(config, class_weights=class_weights)
#     model.load_state_dict(base_model.state_dict(), strict=False)

#     INIT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
#     print(f"[INFO] Saving initialized model (with filler head) to {INIT_MODEL_DIR}")
#     model.save_pretrained(INIT_MODEL_DIR, safe_serialization=False)


# # encoder freeze
# for p in model.model.encoder.parameters():
#     p.requires_grad = False
# print("[INFO] Frozen Whisper encoder parameters (encoder is not trainable).")

# # generation 설정
# model.generation_config.language = "korean"
# model.generation_config.task = "transcribe"
# model.generation_config.forced_decoder_ids = None

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# print("[INFO] model loaded on:", device)
# print("[INFO] n_gpu visible to PyTorch:", torch.cuda.device_count())

# # =========================
# # 6. Data Collator (Whisper + filler_labels)
# # =========================

# @dataclass
# class DataCollatorWhisperWithFiller:
#     processor: Any

#     def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, torch.Tensor]:
#         # 1) audio input_features
#         input_features = [{"input_features": f["input_features"]} for f in features]
#         batch = self.processor.feature_extractor.pad(
#             input_features, return_tensors="pt"
#         )

#         # 2) labels (ASR 토큰)
#         label_features = [{"input_ids": f["labels"]} for f in features]
#         labels_batch = self.processor.tokenizer.pad(
#             label_features, return_tensors="pt"
#         )

#         labels = labels_batch["input_ids"]
#         attention_mask = labels_batch["attention_mask"]

#         # pad = -100
#         labels = labels.masked_fill(attention_mask.ne(1), -100)
#         batch["labels"] = labels

#         # 3) filler_labels
#         filler_lists = [f["filler_labels"] for f in features]
#         max_len = labels.shape[1]

#         padded_filler = []
#         for fl in filler_lists:
#             fl = list(fl)
#             pad_len = max_len - len(fl)
#             padded = fl + [-100] * pad_len
#             padded_filler.append(padded[:max_len])

#         filler_labels = torch.tensor(padded_filler, dtype=torch.long)
#         filler_labels = filler_labels.masked_fill(attention_mask.ne(1), -100)

#         batch["filler_labels"] = filler_labels
#         return batch

# data_collator = DataCollatorWhisperWithFiller(processor=processor)

# # =========================
# # 7. 평가 지표 (WER)
# # =========================

# wer_metric = evaluate.load("wer")

# def compute_metrics(pred):
#     pred_ids = pred.predictions
#     label_ids = pred.label_ids

#     # -100 → pad_token_id
#     label_ids[label_ids == -100] = tokenizer.pad_token_id

#     pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
#     label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

#     wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
#     return {"wer": wer}

# # =========================
# # 7-1. train subset (10만 샘플 사용)
# # =========================

# full_train = kspon_proc["train"]
# N_full = len(full_train)          # 299636
# N_sub  = 100_000                  # 10만 개만 사용

# train_subset = full_train.shuffle(seed=42).select(range(N_sub))

# print("[INFO] train subset size:", len(train_subset))


# # =========================
# # 9. filler head 평가 함수 (precision/recall/F1)
# # =========================

# def eval_filler_head(
#     model,
#     dataset,
#     processor,
#     device,
#     max_batches: int = 200,
#     batch_size: int = 8,
# ):
#     """
#     eval split에서 filler head 기준 precision/recall/F1/accuracy 계산.
#     max_batches: 평가에 사용할 배치 수 (전체 다 돌리면 오래 걸릴 수 있으므로 제한)
#     """
#     try:
#         from sklearn.metrics import precision_recall_fscore_support, accuracy_score
#     except ImportError:
#         print("[WARN] scikit-learn이 설치되어 있지 않아 filler precision/recall/F1 계산을 건너뜁니다.")
#         return None

#     model.eval()
#     loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         collate_fn=data_collator,
#     )

#     all_preds = []
#     all_labels = []
#     all_filler_losses = []

#     with torch.no_grad():
#         for step, batch in enumerate(loader):
#             if max_batches is not None and step >= max_batches:
#                 break

#             batch = {
#                 k: (v.to(device) if isinstance(v, torch.Tensor) else v)
#                 for k, v in batch.items()
#             }

#             outputs = model(
#                 input_features=batch["input_features"],
#                 labels=batch["labels"],
#                 filler_labels=batch["filler_labels"],
#             )

#             filler_logits = outputs.filler_logits      # (B, T, 2)
#             filler_labels = batch["filler_labels"]     # (B, T)
#             if outputs.filler_loss is not None:
#                 all_filler_losses.append(outputs.filler_loss.item())

#             preds = filler_logits.argmax(dim=-1)       # (B, T)

#             # -100 위치는 무시
#             mask = filler_labels != -100
#             if mask.sum() == 0:
#                 continue

#             all_preds.append(preds[mask].cpu().numpy())
#             all_labels.append(filler_labels[mask].cpu().numpy())

#     if not all_preds:
#         print("[WARN] filler eval에서 유효한 토큰이 없습니다.")
#         return None

#     all_preds = np.concatenate(all_preds)
#     all_labels = np.concatenate(all_labels)

#     prec, rec, f1, _ = precision_recall_fscore_support(
#         all_labels, all_preds, average="binary", pos_label=1
#     )
#     acc = accuracy_score(all_labels, all_preds)
#     avg_filler_loss = float(np.mean(all_filler_losses)) if all_filler_losses else float("nan")

#     metrics = {
#         "filler_precision": float(prec),
#         "filler_recall": float(rec),
#         "filler_f1": float(f1),
#         "filler_accuracy": float(acc),
#         "filler_loss_eval": avg_filler_loss,
#         "num_tokens": int(len(all_labels)),
#     }
#     return metrics

# from transformers import TrainerCallback

# class FillerEvalCallback(TrainerCallback):
#     """
#     각 epoch 끝날 때 eval split에서 filler head 성능(precision/recall/F1 등)을 평가해서
#     콘솔 로그에 출력하는 콜백.
#     """
#     def __init__(
#         self,
#         eval_dataset,
#         processor,
#         data_collator,
#         max_batches: int = 200,
#         batch_size: int = 8,
#     ):
#         self.eval_dataset = eval_dataset
#         self.processor = processor
#         self.data_collator = data_collator
#         self.max_batches = max_batches
#         self.batch_size = batch_size

#     def on_epoch_end(self, args, state, control, **kwargs):
#         model = kwargs["model"]

#         # 현재 epoch 번호 (float 형태)
#         current_epoch = state.epoch

#         print(f"\n[INFO] ===== FILLER EVAL at epoch {current_epoch:.2f} =====")
#         metrics = eval_filler_head(
#             model=model,
#             dataset=self.eval_dataset,
#             processor=self.processor,
#             device=device,  # 전역으로 정의된 device 사용
#             max_batches=self.max_batches,
#             batch_size=self.batch_size,
#         )
#         if metrics is not None:
#             print(f"[INFO] FILLER HEAD metrics (epoch {current_epoch:.2f}): {metrics}")
#         else:
#             print("[INFO] FILLER HEAD metrics: None (no valid tokens)")


# # =========================
# # 8. Trainer 세팅
# # =========================
# filler_callback = FillerEvalCallback(
#     eval_dataset=kspon_proc["eval"],
#     processor=processor,
#     data_collator=data_collator,
#     max_batches=200,   # 필요하면 100/300 등으로 조절
#     batch_size=8,
# )


# training_args = Seq2SeqTrainingArguments(
#     output_dir=str(PROJECT_ROOT / "checkpoints_2"),
#     per_device_train_batch_size=16,   # ← 6 → 8 (global batch 32)
#     per_device_eval_batch_size=16,
#     gradient_accumulation_steps=2,   # global batch = 8 * 2(gpu) * 2 = 32
#     learning_rate=1e-5,
#     warmup_steps=500,
#     num_train_epochs=3.0,                  # ← 20000 → 8000, 12h 안으로 단축
#     gradient_checkpointing=False,
#     fp16=True,
#     logging_steps=50,                # loss를 더 자주 로그
#     save_steps=2000,
#     dataloader_num_workers=4,
#     eval_accumulation_steps=2,
# )

# trainer = Seq2SeqTrainer(
#     args=training_args,
#     model=model,
#     train_dataset=train_subset,
#     eval_dataset=kspon_proc["eval"],
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,    # WER
#     processing_class=processor,         # tokenizer 대신
#     callbacks=[filler_callback], 
# )


# # =========================
# # 10. main
# # =========================

# def main():
#     print("[INFO] Start training...")
#     trainer.train()
#     print("[INFO] Training finished.")

#     # ------------------------------
#     # 1) ASR 기준 eval (WER) - OOM 나도 학습 결과는 살려두기
#     # ------------------------------
#     asr_metrics = None
#     try:
#         print("[INFO] Running final ASR evaluation on eval split...")
#         asr_metrics = trainer.evaluate(eval_dataset=kspon_proc["eval"])
#         print("[INFO] ASR Eval metrics:", asr_metrics)
#     except torch.cuda.OutOfMemoryError:
#         print("[WARN] CUDA OOM during final ASR evaluation. Skip WER and continue.")
#     except RuntimeError as e:
#         if "CUDA out of memory" in str(e):
#             print("[WARN] CUDA OOM (RuntimeError) during final ASR evaluation. Skip WER and continue.")
#         else:
#             raise

#     # ------------------------------
#     # 2) filler head 기준 precision/recall/F1 - 이것도 OOM 방어
#     # ------------------------------
#     filler_metrics = None
#     try:
#         print("[INFO] Running final FILLER HEAD evaluation on eval split (subset)...")
#         filler_metrics = eval_filler_head(
#             model=model,
#             dataset=kspon_proc["eval"],
#             processor=processor,
#             device=device,
#             max_batches=200,   # eval batch 200개 정도만 사용 (필요하면 늘려도 됨)
#             batch_size=8,
#         )
#         if filler_metrics is not None:
#             print("[INFO] FILLER HEAD metrics:", filler_metrics)
#         else:
#             print("[INFO] FILLER HEAD metrics: None (no valid tokens)")
#     except torch.cuda.OutOfMemoryError:
#         print("[WARN] CUDA OOM during final FILLER evaluation. Skip filler metrics and continue.")
#     except RuntimeError as e:
#         if "CUDA out of memory" in str(e):
#             print("[WARN] CUDA OOM (RuntimeError) during final FILLER evaluation. Skip filler metrics and continue.")
#         else:
#             raise

#     # ------------------------------
#     # 3) 최종 모델 저장 (항상 실행)
#     # ------------------------------
#     print("[INFO] Saving final model...")
#     out_dir = PROJECT_ROOT / "final_model"
#     trainer.save_model(str(out_dir))
#     processor.save_pretrained(str(out_dir))
#     print(f"[INFO] Final model saved to: {out_dir}")


# if __name__ == "__main__":
#     main()

# # whisper_train.py (Stage2 finetune 버전)

# import os
# from pathlib import Path
# from collections import Counter
# import json

# import torch
# import torch.nn as nn
# import numpy as np

# from datasets import load_from_disk
# from transformers import (
#     WhisperFeatureExtractor,
#     WhisperTokenizerFast,
#     WhisperProcessor,
#     WhisperForConditionalGeneration,
#     Seq2SeqTrainingArguments,
#     Seq2SeqTrainer,
#     TrainerCallback,
# )
# from dataclasses import dataclass
# from typing import Any, Dict, List, Union
# import evaluate
# from datasets import Dataset
# from torch.utils.data import DataLoader

# # =========================
# # 환경/경로 설정
# # =========================

# PROJECT_ROOT = Path("/home/alpaco/jws/whisper_only")
# DATA_ROOT    = PROJECT_ROOT / "data"
# PROC_DATA_DIR = DATA_ROOT / "ksponspeech_proc"

# META_DIR = PROJECT_ROOT / "meta"
# META_DIR.mkdir(parents=True, exist_ok=True)
# CLASS_WEIGHT_FILE = META_DIR / "class_weights.json"

# MODEL_NAME = "openai/whisper-small"  # multilingual

# # Stage1에서 가장 성능이 좋았던 checkpoint 경로 (필요에 따라 수정)
# BEST_STAGE1_CKPT = PROJECT_ROOT / "checkpoints" / "checkpoint-4000"

# # base + filler head 초기 모델 (없으면 생성)
# INIT_MODEL_DIR = PROJECT_ROOT / "init_whisper_small_filler"

# # =========================
# # 1. Processor / Tokenizer 로드
# # =========================

# feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
# tokenizer = WhisperTokenizerFast.from_pretrained(
#     MODEL_NAME,
#     language="Korean",
#     task="transcribe",
# )
# processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# # =========================
# # 2. 전처리된 Kspon 데이터 로드
# # =========================

# print(f"[INFO] Loading preprocessed KsponSpeech from: {PROC_DATA_DIR}")
# kspon_proc = load_from_disk(str(PROC_DATA_DIR))
# print(kspon_proc)
# print(kspon_proc["train"][0].keys())  # 'input_features', 'labels', 'filler_labels'

# # =========================
# # 3. filler class weight 계산 (raw) + Stage2용 완화 weight 설정
# # =========================

# def count_filler_tokens(dataset):
#     counter = Counter()
#     for ex in dataset:
#         for y in ex["filler_labels"]:
#             if y == -100:
#                 continue
#             counter[int(y)] += 1
#     return counter

# if CLASS_WEIGHT_FILE.exists():
#     print(f"[INFO] Load class weights from {CLASS_WEIGHT_FILE}")
#     with open(CLASS_WEIGHT_FILE, "r", encoding="utf-8") as f:
#         w = json.load(f)
#     w0_raw = float(w["w0"])
#     w1_raw = float(w["w1"])
# else:
#     print("[INFO] Compute class weights from kspon_proc['train'] (first time only)")
#     counter = count_filler_tokens(kspon_proc["train"])
#     print("[INFO] filler token counts:", counter)

#     total = counter[0] + counter[1]
#     w0_raw = float(total / (2 * counter[0]))  # non-filler
#     w1_raw = float(total / (2 * counter[1]))  # filler

#     with open(CLASS_WEIGHT_FILE, "w", encoding="utf-8") as f:
#         json.dump({"w0": w0_raw, "w1": w1_raw}, f, ensure_ascii=False, indent=2)
#     print(f"[INFO] Saved class weights to {CLASS_WEIGHT_FILE}")

# print(f"[INFO] raw class weights from counts: non-filler={w0_raw:.4f}, filler={w1_raw:.4f}")

# # --- Stage2: class weight 완화 (precision 조금 올리되 recall은 유지하는 방향) ---
# #  예: non-filler=1.0, filler=20.0  (기존 ≈ 1 : 117 에서 1 : 20 정도로 완화)
# W0_STAGE2 = 1.0
# W1_STAGE2 = 20.0
# class_weights = [W0_STAGE2, W1_STAGE2]
# print(f"[INFO] (Stage2) override class weights: non-filler={W0_STAGE2}, filler={W1_STAGE2}")

# # =========================
# # 4. Whisper + filler 헤드 모델 정의
# # =========================

# class WhisperWithFillerHead(WhisperForConditionalGeneration):
#     def __init__(self, config, class_weights=None):
#         super().__init__(config)

#         # decoder hidden state → 2-class (non-filler / filler)
#         self.filler_classifier = nn.Linear(config.d_model, 2)

#         # 항상 hidden states 출력
#         self.config.output_hidden_states = True

#         # 여기서는 class_weights는 무시하고, 아래에서 별도로 재설정
#         self.filler_loss_fct = nn.CrossEntropyLoss(
#             ignore_index=-100,
#         )

#     def forward(
#         self,
#         input_features=None,
#         attention_mask=None,
#         decoder_input_ids=None,
#         decoder_attention_mask=None,
#         labels=None,
#         filler_labels=None,
#         **kwargs,
#     ):
#         # Trainer가 내부적으로 넘기는 num_items_in_batch 같은 인자는
#         # WhisperForConditionalGeneration.forward 에서는 지원하지 않으므로 제거
#         if "num_items_in_batch" in kwargs:
#             kwargs.pop("num_items_in_batch")

#         # Whisper 기본 forward (ASR loss 포함)
#         outputs = super().forward(
#             input_features=input_features,
#             attention_mask=attention_mask,
#             decoder_input_ids=decoder_input_ids,
#             decoder_attention_mask=decoder_attention_mask,
#             labels=labels,
#             **kwargs,
#         )

#         total_loss = outputs.loss
#         filler_logits = None
#         filler_loss = None

#         # decoder hidden states → filler head
#         if outputs.decoder_hidden_states is not None:
#             last_hidden = outputs.decoder_hidden_states[-1]       # (B, T, d_model)
#             filler_logits = self.filler_classifier(last_hidden)   # (B, T, 2)

#             if filler_labels is not None:
#                 B, T, C = filler_logits.shape
#                 filler_loss = self.filler_loss_fct(
#                     filler_logits.view(B * T, C),
#                     filler_labels.view(B * T),
#                 )
#                 if total_loss is not None:
#                     total_loss = total_loss + filler_loss
#                 else:
#                     total_loss = filler_loss

#         outputs.loss = total_loss
#         outputs.filler_logits = filler_logits
#         outputs.filler_loss = filler_loss
#         return outputs

# # =========================
# # 5. 초기 모델 로딩 (Stage1 checkpoint → 없으면 base/init)
# # =========================

# def has_init_weights(init_dir: Path) -> bool:
#     return (init_dir / "pytorch_model.bin").exists() or (init_dir / "model.safetensors").exists()

# if BEST_STAGE1_CKPT.exists():
#     # Stage1에서 학습된 best checkpoint를 초기값으로 사용
#     print(f"[INFO] Loading model from Stage1 checkpoint: {BEST_STAGE1_CKPT}")
#     model = WhisperWithFillerHead.from_pretrained(str(BEST_STAGE1_CKPT))
# elif has_init_weights(INIT_MODEL_DIR):
#     # base + filler head로 한 번 초기화해 둔 모델이 있으면 사용
#     print(f"[INFO] Loading initialized model with filler head from {INIT_MODEL_DIR}")
#     model = WhisperWithFillerHead.from_pretrained(str(INIT_MODEL_DIR))
# else:
#     # 완전 처음일 때 (base Whisper → filler head로 확장)
#     print(f"[INFO] Loading base Whisper model from {MODEL_NAME} (init)")
#     base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
#     config = base_model.config
#     config.use_cache = False

#     model = WhisperWithFillerHead(config)
#     model.load_state_dict(base_model.state_dict(), strict=False)

#     INIT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
#     print(f"[INFO] Saving initialized model (with filler head) to {INIT_MODEL_DIR}")
#     model.save_pretrained(INIT_MODEL_DIR, safe_serialization=False)

# # --- Stage2용 class weight를 loss에 반영 (checkpoint에서 로드된 weight 덮어쓰기) ---
# if class_weights is not None:
#     w_tensor = torch.tensor(class_weights, dtype=torch.float32)
#     model.filler_loss_fct = nn.CrossEntropyLoss(
#         ignore_index=-100,
#         weight=w_tensor,
#     )
# else:
#     model.filler_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

# # encoder freeze (Stage2에서도 encoder는 고정)
# for p in model.model.encoder.parameters():
#     p.requires_grad = False
# print("[INFO] Frozen Whisper encoder parameters (encoder is not trainable).")

# # generation 설정
# model.generation_config.language = "korean"
# model.generation_config.task = "transcribe"
# model.generation_config.forced_decoder_ids = None

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# print("[INFO] model loaded on:", device)
# print("[INFO] n_gpu visible to PyTorch:", torch.cuda.device_count())

# # =========================
# # 6. Data Collator (Whisper + filler_labels)
# # =========================

# @dataclass
# class DataCollatorWhisperWithFiller:
#     processor: Any

#     def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, torch.Tensor]:
#         # 1) audio input_features
#         input_features = [{"input_features": f["input_features"]} for f in features]
#         batch = self.processor.feature_extractor.pad(
#             input_features, return_tensors="pt"
#         )

#         # 2) labels (ASR 토큰)
#         label_features = [{"input_ids": f["labels"]} for f in features]
#         labels_batch = self.processor.tokenizer.pad(
#             label_features, return_tensors="pt"
#         )

#         labels = labels_batch["input_ids"]
#         attention_mask = labels_batch["attention_mask"]

#         # pad = -100
#         labels = labels.masked_fill(attention_mask.ne(1), -100)
#         batch["labels"] = labels

#         # 3) filler_labels
#         filler_lists = [f["filler_labels"] for f in features]
#         max_len = labels.shape[1]

#         padded_filler = []
#         for fl in filler_lists:
#             fl = list(fl)
#             pad_len = max_len - len(fl)
#             padded = fl + [-100] * pad_len
#             padded_filler.append(padded[:max_len])

#         filler_labels = torch.tensor(padded_filler, dtype=torch.long)
#         filler_labels = filler_labels.masked_fill(attention_mask.ne(1), -100)

#         batch["filler_labels"] = filler_labels
#         return batch

# data_collator = DataCollatorWhisperWithFiller(processor=processor)

# # =========================
# # 7. 평가 지표 (WER)
# # =========================

# wer_metric = evaluate.load("wer")

# def compute_metrics(pred):
#     pred_ids = pred.predictions
#     label_ids = pred.label_ids

#     # -100 → pad_token_id
#     label_ids[label_ids == -100] = tokenizer.pad_token_id

#     pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
#     label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

#     wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
#     return {"wer": wer}

# # =========================
# # 7-1. train subset (10만 샘플 사용)
# # =========================

# full_train = kspon_proc["train"]
# N_full = len(full_train)          # 299636
# N_sub  = 100_000                  # 10만 개만 사용

# train_subset = full_train.shuffle(seed=42).select(range(N_sub))

# print("[INFO] train subset size:", len(train_subset))

# # =========================
# # 9. filler head 평가 함수 (precision/recall/F1)
# # =========================

# def eval_filler_head(
#     model,
#     dataset,
#     processor,
#     device,
#     max_batches: int = 200,
#     batch_size: int = 8,
# ):
#     """
#     eval split에서 filler head 기준 precision/recall/F1/accuracy 계산.
#     max_batches: 평가에 사용할 배치 수 (전체 다 돌리면 오래 걸릴 수 있으므로 제한)
#     """
#     try:
#         from sklearn.metrics import precision_recall_fscore_support, accuracy_score
#     except ImportError:
#         print("[WARN] scikit-learn이 설치되어 있지 않아 filler precision/recall/F1 계산을 건너뜁니다.")
#         return None

#     model.eval()
#     loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         collate_fn=data_collator,
#     )

#     all_preds = []
#     all_labels = []
#     all_filler_losses = []

#     with torch.no_grad():
#         for step, batch in enumerate(loader):
#             if max_batches is not None and step >= max_batches:
#                 break

#             batch = {
#                 k: (v.to(device) if isinstance(v, torch.Tensor) else v)
#                 for k, v in batch.items()
#             }

#             outputs = model(
#                 input_features=batch["input_features"],
#                 labels=batch["labels"],
#                 filler_labels=batch["filler_labels"],
#             )

#             filler_logits = outputs.filler_logits      # (B, T, 2)
#             filler_labels = batch["filler_labels"]     # (B, T)
#             if outputs.filler_loss is not None:
#                 all_filler_losses.append(outputs.filler_loss.item())

#             preds = filler_logits.argmax(dim=-1)       # (B, T)

#             # -100 위치는 무시
#             mask = filler_labels != -100
#             if mask.sum() == 0:
#                 continue

#             all_preds.append(preds[mask].cpu().numpy())
#             all_labels.append(filler_labels[mask].cpu().numpy())

#     if not all_preds:
#         print("[WARN] filler eval에서 유효한 토큰이 없습니다.")
#         return None

#     all_preds = np.concatenate(all_preds)
#     all_labels = np.concatenate(all_labels)

#     prec, rec, f1, _ = precision_recall_fscore_support(
#         all_labels, all_preds, average="binary", pos_label=1
#     )
#     acc = accuracy_score(all_labels, all_preds)
#     avg_filler_loss = float(np.mean(all_filler_losses)) if all_filler_losses else float("nan")

#     metrics = {
#         "filler_precision": float(prec),
#         "filler_recall": float(rec),
#         "filler_f1": float(f1),
#         "filler_accuracy": float(acc),
#         "filler_loss_eval": avg_filler_loss,
#         "num_tokens": int(len(all_labels)),
#     }
#     return metrics

# # =========================
# # 9-1. Epoch마다 filler metric 찍는 콜백
# # =========================

# class FillerEvalCallback(TrainerCallback):
#     """
#     각 epoch 끝날 때 eval split에서 filler head 성능(precision/recall/F1 등)을 평가해서
#     콘솔 로그에 출력하는 콜백.
#     """
#     def __init__(
#         self,
#         eval_dataset,
#         processor,
#         data_collator,
#         max_batches: int = 200,
#         batch_size: int = 8,
#     ):
#         self.eval_dataset = eval_dataset
#         self.processor = processor
#         self.data_collator = data_collator
#         self.max_batches = max_batches
#         self.batch_size = batch_size

#     def on_epoch_end(self, args, state, control, **kwargs):
#         model = kwargs["model"]

#         # 현재 epoch 번호 (float 형태)
#         current_epoch = state.epoch

#         print(f"\n[INFO] ===== FILLER EVAL at epoch {current_epoch:.2f} =====")
#         metrics = eval_filler_head(
#             model=model,
#             dataset=self.eval_dataset,
#             processor=self.processor,
#             device=device,  # 전역으로 정의된 device 사용
#             max_batches=self.max_batches,
#             batch_size=self.batch_size,
#         )
#         if metrics is not None:
#             print(f"[INFO] FILLER HEAD metrics (epoch {current_epoch:.2f}): {metrics}")
#         else:
#             print("[INFO] FILLER HEAD metrics: None (no valid tokens)")

# # =========================
# # 8. Trainer 세팅 (Stage2: 작은 lr, 짧은 epoch)
# # =========================

# filler_callback = FillerEvalCallback(
#     eval_dataset=kspon_proc["eval"],
#     processor=processor,
#     data_collator=data_collator,
#     max_batches=200,   # 필요하면 100/300 등으로 조절
#     batch_size=8,
# )

# training_args = Seq2SeqTrainingArguments(
#     output_dir=str(PROJECT_ROOT / "checkpoints_stage2"),
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     gradient_accumulation_steps=2,   # global batch ≈ 16 * 2(gpu) * 2 = 64
#     learning_rate=3e-6,              # Stage2: 작은 lr
#     warmup_steps=200,
#     num_train_epochs=2.0,            # Stage2: 1~2 epoch만 추가
#     gradient_checkpointing=False,
#     fp16=True,
#     logging_steps=50,                # loss를 자주 로그
#     save_steps=1000,
#     dataloader_num_workers=4,
#     eval_accumulation_steps=2,
# )

# trainer = Seq2SeqTrainer(
#     args=training_args,
#     model=model,
#     train_dataset=train_subset,
#     eval_dataset=kspon_proc["eval"],
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,    # WER
#     processing_class=processor,
#     callbacks=[filler_callback],
# )

# # =========================
# # 10. main
# # =========================

# def main():
#     print("[INFO] Start training...")
#     trainer.train()
#     print("[INFO] Training finished.")

#     # 1) ASR 기준 eval (WER) - OOM 나도 학습 결과는 살려두기
#     asr_metrics = None
#     try:
#         print("[INFO] Running final ASR evaluation on eval split...")
#         asr_metrics = trainer.evaluate(eval_dataset=kspon_proc["eval"])
#         print("[INFO] ASR Eval metrics:", asr_metrics)
#     except torch.cuda.OutOfMemoryError:
#         print("[WARN] CUDA OOM during final ASR evaluation. Skip WER and continue.")
#     except RuntimeError as e:
#         if "CUDA out of memory" in str(e):
#             print("[WARN] CUDA OOM (RuntimeError) during final ASR evaluation. Skip WER and continue.")
#         else:
#             raise

#     # 2) filler head 기준 precision/recall/F1 - 이것도 OOM 방어
#     filler_metrics = None
#     try:
#         print("[INFO] Running final FILLER HEAD evaluation on eval split (subset)...")
#         filler_metrics = eval_filler_head(
#             model=model,
#             dataset=kspon_proc["eval"],
#             processor=processor,
#             device=device,
#             max_batches=200,
#             batch_size=8,
#         )
#         if filler_metrics is not None:
#             print("[INFO] FILLER HEAD metrics:", filler_metrics)
#         else:
#             print("[INFO] FILLER HEAD metrics: None (no valid tokens)")
#     except torch.cuda.OutOfMemoryError:
#         print("[WARN] CUDA OOM during final FILLER evaluation. Skip filler metrics and continue.")
#     except RuntimeError as e:
#         if "CUDA out of memory" in str(e):
#             print("[WARN] CUDA OOM (RuntimeError) during final FILLER evaluation. Skip filler metrics and continue.")
#         else:
#             raise

#     # 3) 최종 모델 저장 (항상 실행)
#     print("[INFO] Saving final model...")
#     out_dir = PROJECT_ROOT / "final_model_stage2"
#     trainer.save_model(str(out_dir))
#     processor.save_pretrained(str(out_dir))
#     print(f"[INFO] Final model saved to: {out_dir}")


# if __name__ == "__main__":
#     main()

# whisper_train.py  (Stage2' : λ=3.0 + MLP head + 150k subset + Stage2 best init)

import os
from pathlib import Path
from collections import Counter
import json

import torch
import torch.nn as nn
import numpy as np

from datasets import load_from_disk
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizerFast,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from datasets import Dataset
from torch.utils.data import DataLoader

# =========================
# 환경/경로 설정
# =========================

PROJECT_ROOT = Path("/home/alpaco/jws/whisper_only")
DATA_ROOT    = PROJECT_ROOT / "data"
PROC_DATA_DIR = DATA_ROOT / "ksponspeech_proc"

META_DIR = PROJECT_ROOT / "meta"
META_DIR.mkdir(parents=True, exist_ok=True)
CLASS_WEIGHT_FILE = META_DIR / "class_weights.json"

MODEL_NAME = "openai/whisper-small"  # multilingual

# Stage2에서 가장 성능이 좋았던 checkpoint 경로
# → 우상이가 eval에서 f1 제일 좋았던 checkpoint 번호로 수정해서 사용
STAGE2_BEST_CKPT = PROJECT_ROOT / "checkpoints_stage2" / "checkpoint-3126"

# base + filler head 초기 모델 (없으면 생성할 때 사용)
INIT_MODEL_DIR = PROJECT_ROOT / "init_whisper_small_filler"

# Stage2'에서 "best filler F1" 모델 저장 위치
BEST_STAGE2P_DIR = PROJECT_ROOT / "best_stage2p_filler"

# FILLER loss 비중 (λ)
LAMBDA_FILLER = 3.0

# =========================
# 1. Processor / Tokenizer 로드
# =========================

feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer = WhisperTokenizerFast.from_pretrained(
    MODEL_NAME,
    language="Korean",
    task="transcribe",
)
processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# =========================
# 2. 전처리된 Kspon 데이터 로드
# =========================

print(f"[INFO] Loading preprocessed KsponSpeech from: {PROC_DATA_DIR}")
kspon_proc = load_from_disk(str(PROC_DATA_DIR))
print(kspon_proc)
print(kspon_proc["train"][0].keys())  # 'input_features', 'labels', 'filler_labels'

# =========================
# 3. filler class weight 계산 (raw) + Stage2용 완화 weight 설정
# =========================

def count_filler_tokens(dataset):
    counter = Counter()
    for ex in dataset:
        for y in ex["filler_labels"]:
            if y == -100:
                continue
            counter[int(y)] += 1
    return counter

if CLASS_WEIGHT_FILE.exists():
    print(f"[INFO] Load class weights from {CLASS_WEIGHT_FILE}")
    with open(CLASS_WEIGHT_FILE, "r", encoding="utf-8") as f:
        w = json.load(f)
    w0_raw = float(w["w0"])
    w1_raw = float(w["w1"])
else:
    print("[INFO] Compute class weights from kspon_proc['train'] (first time only)")
    counter = count_filler_tokens(kspon_proc["train"])
    print("[INFO] filler token counts:", counter)

    total = counter[0] + counter[1]
    w0_raw = float(total / (2 * counter[0]))  # non-filler
    w1_raw = float(total / (2 * counter[1]))  # filler

    with open(CLASS_WEIGHT_FILE, "w", encoding="utf-8") as f:
        json.dump({"w0": w0_raw, "w1": w1_raw}, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved class weights to {CLASS_WEIGHT_FILE}")

print(f"[INFO] raw class weights from counts: non-filler={w0_raw:.4f}, filler={w1_raw:.4f}")

# --- Stage2 / Stage2' : class weight 완화 ---
#  예: non-filler=1.0, filler=20.0
W0_STAGE2 = 1.0
W1_STAGE2 = 20.0
class_weights = [W0_STAGE2, W1_STAGE2]
print(f"[INFO] (Stage2/2') override class weights: non-filler={W0_STAGE2}, filler={W1_STAGE2}")

# =========================
# 4. Whisper + filler 헤드 모델 정의 (λ=3.0, 2-layer MLP head)
# =========================

class WhisperWithFillerHead(WhisperForConditionalGeneration):
    def __init__(self, config, lambda_filler: float = 3.0, class_weights=None):
        super().__init__(config)

        # decoder hidden state → 2-class (non-filler / filler)
        # 작은 MLP 헤드: d_model → d_model → 2
        self.filler_classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.d_model, 2),
        )

        # 항상 hidden states 출력
        self.config.output_hidden_states = True

        # λ (ASR vs FILLER loss 비중)
        self.lambda_filler = float(lambda_filler)

        # class_weights 적용
        if class_weights is not None:
            w_tensor = torch.tensor(class_weights, dtype=torch.float32)
            self.filler_loss_fct = nn.CrossEntropyLoss(
                ignore_index=-100,
                weight=w_tensor,
            )
        else:
            self.filler_loss_fct = nn.CrossEntropyLoss(
                ignore_index=-100,
            )

    def forward(
        self,
        input_features=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        filler_labels=None,
        **kwargs,
    ):
        # Trainer가 내부적으로 넘기는 num_items_in_batch 같은 인자는
        # WhisperForConditionalGeneration.forward 에서는 지원하지 않으므로 제거
        if "num_items_in_batch" in kwargs:
            kwargs.pop("num_items_in_batch")

        # Whisper 기본 forward (ASR loss 포함)
        outputs = super().forward(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            **kwargs,
        )

        asr_loss = outputs.loss
        total_loss = asr_loss
        filler_logits = None
        filler_loss = None

        # decoder hidden states → filler head
        if outputs.decoder_hidden_states is not None:
            last_hidden = outputs.decoder_hidden_states[-1]       # (B, T, d_model)
            filler_logits = self.filler_classifier(last_hidden)   # (B, T, 2)

            if filler_labels is not None:
                B, T, C = filler_logits.shape
                filler_loss = self.filler_loss_fct(
                    filler_logits.view(B * T, C),
                    filler_labels.view(B * T),
                )
                # λ를 곱해서 total_loss에 반영
                if total_loss is not None:
                    total_loss = total_loss + self.lambda_filler * filler_loss
                else:
                    total_loss = self.lambda_filler * filler_loss

        outputs.loss = total_loss
        outputs.filler_logits = filler_logits
        outputs.filler_loss = filler_loss
        outputs.asr_loss = asr_loss
        return outputs

# =========================
# 5. 초기 모델 로딩 (Stage2 best → 없으면 init/base)
# =========================

def has_init_weights(init_dir: Path) -> bool:
    return (init_dir / "pytorch_model.bin").exists() or (init_dir / "model.safetensors").exists()

if STAGE2_BEST_CKPT.exists():
    # Stage2에서 학습된 best checkpoint를 초기값으로 사용
    print(f"[INFO] Loading model from Stage2 best checkpoint: {STAGE2_BEST_CKPT}")
    model = WhisperWithFillerHead.from_pretrained(
        str(STAGE2_BEST_CKPT),
        lambda_filler=LAMBDA_FILLER,
        class_weights=class_weights,
        ignore_mismatched_sizes=True,  # 새 MLP head 때문에 크기 불일치 허용
    )
elif has_init_weights(INIT_MODEL_DIR):
    # base + filler head로 초기화해 둔 모델이 있으면 사용
    print(f"[INFO] Loading initialized model with filler head from {INIT_MODEL_DIR}")
    model = WhisperWithFillerHead.from_pretrained(
        str(INIT_MODEL_DIR),
        lambda_filler=LAMBDA_FILLER,
        class_weights=class_weights,
        ignore_mismatched_sizes=True,
    )
else:
    # 완전 처음일 때 (base Whisper → filler head로 확장)
    print(f"[INFO] Loading base Whisper model from {MODEL_NAME} (init)")
    base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    config = base_model.config
    config.use_cache = False

    model = WhisperWithFillerHead(
        config,
        lambda_filler=LAMBDA_FILLER,
        class_weights=class_weights,
    )
    model.load_state_dict(base_model.state_dict(), strict=False)

    INIT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving initialized model (with filler head) to {INIT_MODEL_DIR}")
    model.save_pretrained(INIT_MODEL_DIR, safe_serialization=False)

# encoder freeze (Stage2'/fine-tune에서도 encoder는 고정)
for p in model.model.encoder.parameters():
    p.requires_grad = False
print("[INFO] Frozen Whisper encoder parameters (encoder is not trainable).")

# generation 설정
model.generation_config.language = "korean"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("[INFO] model loaded on:", device)
print("[INFO] n_gpu visible to PyTorch:", torch.cuda.device_count())

# =========================
# 6. Data Collator (Whisper + filler_labels)
# =========================

@dataclass
class DataCollatorWhisperWithFiller:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, torch.Tensor]:
        # 1) audio input_features
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # 2) labels (ASR 토큰)
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        labels = labels_batch["input_ids"]
        attention_mask = labels_batch["attention_mask"]

        # pad = -100
        labels = labels.masked_fill(attention_mask.ne(1), -100)
        batch["labels"] = labels

        # 3) filler_labels
        filler_lists = [f["filler_labels"] for f in features]
        max_len = labels.shape[1]

        padded_filler = []
        for fl in filler_lists:
            fl = list(fl)
            pad_len = max_len - len(fl)
            padded = fl + [-100] * pad_len
            padded_filler.append(padded[:max_len])

        filler_labels = torch.tensor(padded_filler, dtype=torch.long)
        filler_labels = filler_labels.masked_fill(attention_mask.ne(1), -100)

        batch["filler_labels"] = filler_labels
        return batch

data_collator = DataCollatorWhisperWithFiller(processor=processor)

# =========================
# 7. 평가 지표 (WER) – Trainer용 (ASR)
# =========================

wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # -100 → pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# =========================
# 7-1. train subset (15만 샘플 사용)
# =========================

full_train = kspon_proc["train"]
N_full = len(full_train)          # 299636
N_sub  = 150_000                  # 15만 개 사용 (Stage2' 확장)

train_subset = full_train.shuffle(seed=42).select(range(N_sub))

print("[INFO] train subset size:", len(train_subset))

# =========================
# 9. filler head 평가 함수 (precision/recall/F1)
# =========================

def eval_filler_head(
    model,
    dataset,
    processor,
    device,
    max_batches: int = 200,
    batch_size: int = 8,
):
    """
    eval split에서 filler head 기준 precision/recall/F1/accuracy 계산.
    max_batches: 평가에 사용할 배치 수 (전체 다 돌리면 오래 걸릴 수 있으므로 제한)
    """
    try:
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    except ImportError:
        print("[WARN] scikit-learn이 설치되어 있지 않아 filler precision/recall/F1 계산을 건너뜁니다.")
        return None

    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    all_preds = []
    all_labels = []
    all_filler_losses = []

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if max_batches is not None and step >= max_batches:
                break

            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }

            outputs = model(
                input_features=batch["input_features"],
                labels=batch["labels"],
                filler_labels=batch["filler_labels"],
            )

            filler_logits = outputs.filler_logits      # (B, T, 2)
            filler_labels = batch["filler_labels"]     # (B, T)
            if outputs.filler_loss is not None:
                all_filler_losses.append(outputs.filler_loss.item())

            preds = filler_logits.argmax(dim=-1)       # (B, T)

            # -100 위치는 무시
            mask = filler_labels != -100
            if mask.sum() == 0:
                continue

            all_preds.append(preds[mask].cpu().numpy())
            all_labels.append(filler_labels[mask].cpu().numpy())

    if not all_preds:
        print("[WARN] filler eval에서 유효한 토큰이 없습니다.")
        return None

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", pos_label=1
    )
    acc = accuracy_score(all_labels, all_preds)
    avg_filler_loss = float(np.mean(all_filler_losses)) if all_filler_losses else float("nan")

    metrics = {
        "filler_precision": float(prec),
        "filler_recall": float(rec),
        "filler_f1": float(f1),
        "filler_accuracy": float(acc),
        "filler_loss_eval": avg_filler_loss,
        "num_tokens": int(len(all_labels)),
    }
    return metrics

# =========================
# 9-1. Epoch마다 filler metric 찍고 best 모델 저장하는 콜백
# =========================

class FillerEvalCallback(TrainerCallback):
    """
    각 epoch 끝날 때 eval split에서 filler head 성능(precision/recall/F1 등)을 평가해서
    콘솔 로그에 출력 + F1 기준 best 모델을 BEST_STAGE2P_DIR에 저장.
    """
    def __init__(
        self,
        eval_dataset,
        processor,
        data_collator,
        best_model_dir: Path,
        max_batches: int = 200,
        batch_size: int = 8,
    ):
        self.eval_dataset = eval_dataset
        self.processor = processor
        self.data_collator = data_collator
        self.max_batches = max_batches
        self.batch_size = batch_size
        self.best_model_dir = Path(best_model_dir)
        self.best_f1 = -1.0
        self.best_epoch = None

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]

        # 현재 epoch 번호 (float 형태)
        current_epoch = state.epoch

        print(f"\n[INFO] ===== FILLER EVAL at epoch {current_epoch:.2f} =====")
        metrics = eval_filler_head(
            model=model,
            dataset=self.eval_dataset,
            processor=self.processor,
            device=device,  # 전역으로 정의된 device 사용
            max_batches=self.max_batches,
            batch_size=self.batch_size,
        )
        if metrics is not None:
            print(f"[INFO] FILLER HEAD metrics (epoch {current_epoch:.2f}): {metrics}")
            f1 = metrics.get("filler_f1", 0.0)
            if f1 > self.best_f1:
                self.best_f1 = f1
                self.best_epoch = current_epoch
                # best 모델 저장
                self.best_model_dir.mkdir(parents=True, exist_ok=True)
                print(
                    f"[INFO] New BEST filler F1={f1:.4f} at epoch {current_epoch:.2f}. "
                    f"Saving to {self.best_model_dir}"
                )
                model.save_pretrained(str(self.best_model_dir))
                # processor는 전역 객체 사용
                processor.save_pretrained(str(self.best_model_dir))
        else:
            print("[INFO] FILLER HEAD metrics: None (no valid tokens)")

# =========================
# 8. Trainer 세팅 (Stage2' : λ=3.0, 작은 lr, 2 epoch)
# =========================

filler_callback = FillerEvalCallback(
    eval_dataset=kspon_proc["eval"],
    processor=processor,
    data_collator=data_collator,
    best_model_dir=BEST_STAGE2P_DIR,
    max_batches=200,   # 필요하면 100/300 등으로 조절
    batch_size=8,
)

training_args = Seq2SeqTrainingArguments(
    output_dir=str(PROJECT_ROOT / "checkpoints_stage2p"),
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,   # global batch ≈ 16 * 2(gpu) * 2 = 64
    learning_rate=3e-6,              # Stage2': 작은 lr
    warmup_steps=200,
    num_train_epochs=3.0,            # Stage2': 2 epoch 추가
    gradient_checkpointing=False,
    fp16=True,
    logging_steps=50,                # loss를 자주 로그
    save_steps=1000,
    dataloader_num_workers=4,
    eval_accumulation_steps=2,
    # evaluation_strategy/ predict_with_generate 등은 여기서는 사용하지 않음
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_subset,
    eval_dataset=kspon_proc["eval"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,    # (사용 안 해도 됨, WER용)
    processing_class=processor,
    callbacks=[filler_callback],
)

# =========================
# 10. main
# =========================

def main():
    print("[INFO] Start training...")
    trainer.train()
    print("[INFO] Training finished.")

    # 1) 학습이 끝난 "마지막" 모델은 바로 저장 (ASR eval 때문에 막히지 않도록)
    final_dir = PROJECT_ROOT / "final_model_stage2p"
    print(f"[INFO] Saving final (last) model to: {final_dir}")
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"[INFO] Final model saved to: {final_dir}")

    # ASR / filler 최종 평가는 별도 ipynb에서 수행
    print("[INFO] Done. (Best filler F1 model is in BEST_STAGE2P_DIR, "
          "and last model is in final_model_stage2p.)")


if __name__ == "__main__":
    main()
