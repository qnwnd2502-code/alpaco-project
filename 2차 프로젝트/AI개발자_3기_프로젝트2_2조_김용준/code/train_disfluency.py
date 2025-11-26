# # [tmux로 이 스크립트 실행하는 방법]
# #
# # 1) 서버 터미널에서 세션 생성
# #    tmux new -s kspon_train
# #
# # 2) 새로 열린 tmux 창 안에서 (conda 환경 활성화 후)
# #    conda activate jws
# #    cd /home/USER/프로젝트경로   # train_disfluency.py 위치로 이동
# #    python train_disfluency.py 2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log
# #
# # 3) 세션 분리(백그라운드로 보내기)
# #    Ctrl + b, d
# #
# # 4) 다시 접속
# #    tmux attach -t kspon_train
# #
# # 이렇게 하면 SSH가 끊겨도 tmux 세션 안에서 학습은 계속 진행됩니다.

# ##############################################
# # 0. 기본 설정 & 라이브러리
# ##############################################
# import os
# import json
# from pathlib import Path
# from collections import Counter

# import numpy as np
# from tqdm import tqdm

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader

# from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
# from sklearn.metrics import classification_report, f1_score

# # ---- (중요) 마지막 GPU만 사용 ----
# # GPU가 3개(CUDA:0,1,2)라고 가정하고, 물리 GPU 2번만 보이도록 설정
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("device =", device)

# # 재현성 (seed 고정)
# def set_seed(seed: int = 42):
#     import random
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# set_seed(42)

# ##############################################
# # 1. 라벨 정의
# ##############################################

# LABEL_LIST = [
#     "O",
#     "B-FIL", "I-FIL",
#     "B-REP", "I-REP",
#     "B-AMB", "I-AMB",
# ]
# label2id = {label: i for i, label in enumerate(LABEL_LIST)}
# id2label = {i: label for label, i in label2id.items()}

# print("label2id:", label2id)

# ##############################################
# # 2. JSONL 로딩 함수
# ##############################################

# def load_jsonl(path: Path):
#     data = []
#     with open(path, encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             obj = json.loads(line)
#             data.append(obj)
#     return data

# ##############################################
# # 3. Train / Dev / Eval 데이터 로드
# ##############################################

# DATA_DIR = Path("Dataset/kspon_disfluency")  # 필요하면 수정
# train_path = DATA_DIR / "kspon_disfluency_train.jsonl"
# dev_path   = DATA_DIR / "kspon_disfluency_dev.jsonl"          # 없으면 train만 써도 됨
# eval_path  = DATA_DIR / "kspon_disfluency_eval_clean.jsonl"   # 이름은 환경에 맞게 수정

# train_examples = load_jsonl(train_path)
# dev_examples   = load_jsonl(dev_path) if dev_path.exists() else []
# eval_examples  = load_jsonl(eval_path) if eval_path.exists() else []

# print("train size:", len(train_examples))
# print("dev size:", len(dev_examples))
# print("eval size:", len(eval_examples))

# ##############################################
# # 4. class weight 계산 (단어 단위 라벨 기준)
# ##############################################

# label_counter = Counter()
# for ex in train_examples:
#     label_counter.update(ex["labels"])  # word-level BIO 라벨

# print("label counts:", label_counter)

# # balanced weight = N / (K * n_i)
# N = sum(label_counter.values())
# K = len(LABEL_LIST)

# raw_weights = []
# for label in LABEL_LIST:
#     n_i = label_counter.get(label, 1)
#     w_i = N / (K * n_i)
#     raw_weights.append(w_i)

# raw_weights = np.array(raw_weights, dtype=np.float32)

# # 너무 극단적인 차이를 막기 위해 스케일링
# # - O: 항상 1.0
# # - 나머지: 최소 weight가 1.0이 되도록 비례 스케일
# non_o = raw_weights[1:]
# min_non_o = non_o.min()
# scale = 1.0 / min_non_o
# scaled_weights = raw_weights * scale
# scaled_weights[0] = 1.0   # O 클래스는 1.0으로 고정

# class_weights = torch.tensor(scaled_weights, dtype=torch.float)
# print("class_weights:", class_weights)

# ##############################################
# # 5. 토크나이저 & Dataset 정의
# ##############################################

# MODEL_NAME = "klue/roberta-large"   # 전이학습 모델

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# MAX_LEN = 128  # 발화 길이에 맞게 조정 가능

# class KsponDisfluencyDataset(Dataset):
#     """
#     JSONL 예제:
#     {
#       "tokens": ["아", "그래서", ...],
#       "labels": ["B-FIL", "O", ...]
#     }
#     """
#     def __init__(self, examples, tokenizer, max_len: int = 128):
#         self.examples = examples
#         self.tokenizer = tokenizer
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.examples)

#     def __getitem__(self, idx):
#         ex = self.examples[idx]
#         tokens = ex["tokens"]
#         labels = ex["labels"]

#         # 토크나이저: 단어 리스트 기반 토큰화
#         encoding = self.tokenizer(
#             tokens,
#             is_split_into_words=True,
#             return_tensors="pt",
#             padding="max_length",
#             truncation=True,
#             max_length=self.max_len,
#             return_attention_mask=True,
#         )

#         input_ids = encoding["input_ids"][0]             # (max_len,)
#         attention_mask = encoding["attention_mask"][0]   # (max_len,)

#         # word_ids: 서브워드 → 원래 단어 인덱스 매핑
#         word_ids = encoding.word_ids(0)  # 길이 max_len

#         # 라벨을 서브워드에 맞게 확장
#         # - 한 단어의 첫 서브워드에만 라벨 부여
#         # - 나머지 서브워드는 ignore_index(-100)
#         label_ids = []
#         previous_word_id = None
#         for word_id in word_ids:
#             if word_id is None:
#                 label_ids.append(-100)
#             elif word_id != previous_word_id:
#                 label_str = labels[word_id]
#                 label_ids.append(label2id[label_str])
#             else:
#                 label_ids.append(-100)
#             previous_word_id = word_id

#         label_ids = torch.tensor(label_ids, dtype=torch.long)

#         return {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "labels": label_ids,
#         }

# ##############################################
# # 6. DataLoader 정의
# ##############################################

# train_dataset = KsponDisfluencyDataset(train_examples, tokenizer, max_len=MAX_LEN)
# dev_dataset   = KsponDisfluencyDataset(dev_examples, tokenizer, max_len=MAX_LEN) if dev_examples else None
# eval_dataset  = KsponDisfluencyDataset(eval_examples, tokenizer, max_len=MAX_LEN) if eval_examples else None

# BATCH_SIZE = 16

# def collate_fn(batch):
#     # 이미 padding="max_length"로 맞춰놨기 때문에 단순 스택
#     input_ids = torch.stack([b["input_ids"] for b in batch])
#     attention_mask = torch.stack([b["attention_mask"] for b in batch])
#     labels = torch.stack([b["labels"] for b in batch])
#     return {
#         "input_ids": input_ids,
#         "attention_mask": attention_mask,
#         "labels": labels,
#     }

# train_loader = DataLoader(
#     train_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     collate_fn=collate_fn,
# )

# dev_loader = DataLoader(
#     dev_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     collate_fn=collate_fn,
# ) if dev_dataset is not None else None

# eval_loader = DataLoader(
#     eval_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     collate_fn=collate_fn,
# ) if eval_dataset is not None else None

# ##############################################
# # 7. BERT 기반 Tagger 모델 정의
# ##############################################

# class BertTagger(nn.Module):
#     def __init__(self, model_name: str, num_labels: int, drop_prob: float = 0.1):
#         super().__init__()
#         self.bert = AutoModel.from_pretrained(model_name)
#         hidden_size = self.bert.config.hidden_size
#         self.dropout = nn.Dropout(drop_prob)
#         self.classifier = nn.Linear(hidden_size, num_labels)

#     def forward(self, input_ids, attention_mask=None, labels=None, class_weights=None):
#         outputs = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#         )
#         # last_hidden_state: (batch, seq_len, hidden)
#         sequence_output = outputs.last_hidden_state
#         sequence_output = self.dropout(sequence_output)
#         logits = self.classifier(sequence_output)  # (batch, seq_len, num_labels)

#         loss = None
#         if labels is not None:
#             # CrossEntropy: (N, C) vs (N,)
#             loss_fct = nn.CrossEntropyLoss(
#                 weight=class_weights.to(logits.device) if class_weights is not None else None,
#                 ignore_index=-100,
#             )
#             loss = loss_fct(
#                 logits.view(-1, logits.size(-1)),   # (batch*seq_len, num_labels)
#                 labels.view(-1)                      # (batch*seq_len,)
#             )
#         return logits, loss

# ##############################################
# # 8. 모델 / 옵티마이저 / 스케줄러
# ##############################################

# model = BertTagger(MODEL_NAME, num_labels=len(LABEL_LIST))
# model.to(device)

# LR = 5e-5
# EPOCHS = 3   # 필요 시 3, 5 등으로 조정

# optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# # 스케줄러 (linear warmup)
# num_training_steps = len(train_loader) * EPOCHS
# num_warmup_steps = int(0.1 * num_training_steps)

# scheduler = get_linear_schedule_with_warmup(
#     optimizer,
#     num_warmup_steps=num_warmup_steps,
#     num_training_steps=num_training_steps,
# )

# ##############################################
# # 9. 평가 함수 (macro F1 + 보고서)
# ##############################################

# def evaluate(model, data_loader, desc="valid"):
#     model.eval()
#     all_preds = []
#     all_labels = []

#     with torch.no_grad():
#         for batch in tqdm(data_loader, desc=desc):
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             labels = batch["labels"].to(device)

#             logits, _ = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 labels=None,
#                 class_weights=None,
#             )

#             # (batch, seq_len, num_labels)
#             preds = torch.argmax(logits, dim=-1)

#             # -100(ignore) 위치 제거 후 flatten
#             for p, l in zip(preds, labels):
#                 p = p.cpu().numpy()
#                 l = l.cpu().numpy()
#                 for pi, li in zip(p, l):
#                     if li == -100:
#                         continue
#                     all_preds.append(int(pi))
#                     all_labels.append(int(li))

#     # macro F1 (전체 라벨 기준)
#     f1_macro = f1_score(all_labels, all_preds, average="macro")
#     print(f"[{desc}] macro F1 (all labels): {f1_macro:.4f}")

#     # FIL/REP/AMB만 따로 보고 싶으면:
#     target_indices = [label2id[x] for x in ["B-FIL","I-FIL","B-REP","I-REP","B-AMB","I-AMB"]]
#     f1_events = f1_score(
#         all_labels, all_preds,
#         labels=target_indices,
#         average="macro",
#         zero_division=0,
#     )
#     print(f"[{desc}] macro F1 (disfluency labels only): {f1_events:.4f}")

#     print("\n[Classification Report]")
#     print(classification_report(
#         all_labels,
#         all_preds,
#         target_names=[id2label[i] for i in range(len(LABEL_LIST))],
#         digits=3,
#         zero_division=0,
#     ))

#     return f1_macro, f1_events

# ##############################################
# # 10. 학습 루프 (에폭별 체크포인트 저장 추가)
# ##############################################

# # best_dev_f1는 "디플루언시 라벨만 기준" F1로 관리
# best_dev_f1 = -1.0   # 최소 한 번은 best_model이 저장되도록 -1로 설정

# save_dir = Path("checkpoints_kspon_disfluency")
# save_dir.mkdir(parents=True, exist_ok=True)

# for epoch in range(1, EPOCHS + 1):
#     model.train()
#     epoch_loss = 0.0

#     print(f"\n===== Epoch {epoch}/{EPOCHS} =====")
#     for batch in tqdm(train_loader, desc="train"):
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         labels = batch["labels"].to(device)

#         optimizer.zero_grad()

#         logits, loss = model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             labels=labels,
#             class_weights=class_weights,  # class weight 적용
#         )

#         loss.backward()
#         # gradient clipping (폭발 방지)
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
#         scheduler.step()

#         epoch_loss += loss.item()

#     avg_loss = epoch_loss / len(train_loader)
#     print(f"[Epoch {epoch}] train loss: {avg_loss:.4f}")

#     # (1) 에폭별 체크포인트 항상 저장
#     epoch_ckpt_path = save_dir / f"epoch_{epoch}.pt"
#     torch.save({
#         "model_state_dict": model.state_dict(),
#         "epoch": epoch,
#         "train_loss": avg_loss,
#         "label2id": label2id,
#         "id2label": id2label,
#     }, epoch_ckpt_path)
#     print(f"→ Saved epoch checkpoint to {epoch_ckpt_path}")

#     # (2) dev 평가 + best 모델 저장
#     if dev_loader is not None:
#         dev_f1_macro, dev_f1_events = evaluate(model, dev_loader, desc="dev")

#         # dev F1 (disfluency labels 기준)로 best 모델 저장
#         if dev_f1_events > best_dev_f1:
#             best_dev_f1 = dev_f1_events
#             save_path = save_dir / "best_model.pt"
#             torch.save({
#                 "model_state_dict": model.state_dict(),
#                 "epoch": epoch,
#                 "dev_f1_events": dev_f1_events,
#                 "label2id": label2id,
#                 "id2label": id2label,
#             }, save_path)
#             print(f"→ Best model updated. Saved to {save_path}")
#     else:
#         # dev 세트가 없다면 train 성능만 기록
#         _ = evaluate(model, train_loader, desc="train_eval")

# ##############################################
# # 11. 저장된 best 모델로 eval 세트 평가 (선택)
# ##############################################

# best_ckpt = save_dir / "best_model.pt"
# if best_ckpt.exists() and eval_loader is not None:
#     ckpt = torch.load(best_ckpt, map_location=device)
#     model.load_state_dict(ckpt["model_state_dict"])
#     print(f"\nLoaded best model from epoch {ckpt['epoch']}, dev_f1_events={ckpt['dev_f1_events']:.4f}")
#     _ = evaluate(model, eval_loader, desc="eval")
# else:
#     print("\nNo eval set or best checkpoint not found.")

# [tmux로 이 스크립트 실행하는 방법]
#
# 1) 서버 터미널에서 세션 생성
#    tmux new -s kspon_train
#
# 2) 새로 열린 tmux 창 안에서 (conda 환경 활성화 후)
#    conda activate jws
#    cd /home/USER/프로젝트경로   # train_disfluency.py 위치로 이동
#    python train_disfluency.py 2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log
#
# 3) 세션 분리(백그라운드로 보내기)
#    Ctrl + b, d
#
# 4) 다시 접속
#    tmux attach -t kspon_train
#
# 이렇게 하면 SSH가 끊겨도 tmux 세션 안에서 학습은 계속 진행됩니다.

# ##############################################
# # 0. 기본 설정 & 라이브러리
# ##############################################
# import os
# import json
# from pathlib import Path
# from collections import Counter

# import numpy as np
# from tqdm import tqdm

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader

# from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
# from sklearn.metrics import classification_report, f1_score

# # ---- (중요) 마지막 GPU만 사용 ----
# # GPU가 3개(CUDA:0,1,2)라고 가정하고, 물리 GPU 2번만 보이도록 설정
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("device =", device)

# # 재현성 (seed 고정)
# def set_seed(seed: int = 42):
#     import random
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# set_seed(42)

# ##############################################
# # 1. 라벨 정의
# ##############################################

# LABEL_LIST = [
#     "O",
#     "B-FIL", "I-FIL",
#     "B-REP", "I-REP",
#     "B-AMB", "I-AMB",
# ]
# label2id = {label: i for i, label in enumerate(LABEL_LIST)}
# id2label = {i: label for label, i in label2id.items()}

# print("label2id:", label2id)

# ##############################################
# # 2. JSONL 로딩 함수
# ##############################################

# def load_jsonl(path: Path):
#     data = []
#     with open(path, encoding="utf-8") as f:  # JSONL은 UTF-8로 저장되어 있다고 가정
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             obj = json.loads(line)
#             data.append(obj)
#     return data

# ##############################################
# # 3. Train / Dev / Eval 데이터 로드
# ##############################################

# DATA_DIR = Path("Dataset/kspon_disfluency")  # 필요하면 수정
# train_path = DATA_DIR / "kspon_disfluency_train.jsonl"
# dev_path   = DATA_DIR / "kspon_disfluency_dev.jsonl"          # 없으면 train만 써도 됨
# eval_path  = DATA_DIR / "kspon_disfluency_eval_clean.jsonl"   # 이름은 환경에 맞게 수정

# train_examples = load_jsonl(train_path)
# dev_examples   = load_jsonl(dev_path) if dev_path.exists() else []
# eval_examples  = load_jsonl(eval_path) if eval_path.exists() else []

# print("train size:", len(train_examples))
# print("dev size:", len(dev_examples))
# print("eval size:", len(eval_examples))

# ##############################################
# # 4. class weight 계산 (단어 단위 라벨 기준)
# ##############################################

# label_counter = Counter()
# for ex in train_examples:
#     label_counter.update(ex["labels"])  # word-level BIO 라벨

# print("label counts:", label_counter)

# # balanced weight = N / (K * n_i)
# N = sum(label_counter.values())
# K = len(LABEL_LIST)

# raw_weights = []
# for label in LABEL_LIST:
#     n_i = label_counter.get(label, 1)
#     w_i = N / (K * n_i)
#     raw_weights.append(w_i)

# raw_weights = np.array(raw_weights, dtype=np.float32)

# # 너무 극단적인 차이를 막기 위해 스케일링
# # - O: 항상 1.0
# # - 나머지: 최소 weight가 1.0이 되도록 비례 스케일
# non_o = raw_weights[1:]
# min_non_o = non_o.min()
# scale = 1.0 / min_non_o
# scaled_weights = raw_weights * scale
# scaled_weights[0] = 1.0   # O 클래스는 1.0으로 고정

# class_weights = torch.tensor(scaled_weights, dtype=torch.float)
# print("class_weights:", class_weights)

# ##############################################
# # 5. 토크나이저 & Dataset 정의
# ##############################################

# MODEL_NAME = "klue/roberta-large"   # 전이학습 모델

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# MAX_LEN = 128  # 발화 길이에 맞게 조정 가능

# class KsponDisfluencyDataset(Dataset):
#     """
#     JSONL 예제:
#     {
#       "tokens": ["아", "그래서", ...],
#       "labels": ["B-FIL", "O", ...]
#     }
#     """
#     def __init__(self, examples, tokenizer, max_len: int = 128):
#         self.examples = examples
#         self.tokenizer = tokenizer
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.examples)

#     def __getitem__(self, idx):
#         ex = self.examples[idx]
#         tokens = ex["tokens"]
#         labels = ex["labels"]

#         # 토크나이저: 단어 리스트 기반 토큰화
#         encoding = self.tokenizer(
#             tokens,
#             is_split_into_words=True,
#             return_tensors="pt",
#             padding="max_length",
#             truncation=True,
#             max_length=self.max_len,
#             return_attention_mask=True,
#         )

#         input_ids = encoding["input_ids"][0]             # (max_len,)
#         attention_mask = encoding["attention_mask"][0]   # (max_len,)

#         # word_ids: 서브워드 → 원래 단어 인덱스 매핑
#         word_ids = encoding.word_ids(0)  # 길이 max_len

#         # 라벨을 서브워드에 맞게 확장
#         # - 한 단어의 첫 서브워드에만 라벨 부여
#         # - 나머지 서브워드는 ignore_index(-100)
#         label_ids = []
#         previous_word_id = None
#         for word_id in word_ids:
#             if word_id is None:
#                 label_ids.append(-100)
#             elif word_id != previous_word_id:
#                 label_str = labels[word_id]
#                 label_ids.append(label2id[label_str])
#             else:
#                 label_ids.append(-100)
#             previous_word_id = word_id

#         label_ids = torch.tensor(label_ids, dtype=torch.long)

#         return {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "labels": label_ids,
#         }

# ##############################################
# # 6. DataLoader 정의
# ##############################################

# train_dataset = KsponDisfluencyDataset(train_examples, tokenizer, max_len=MAX_LEN)
# dev_dataset   = KsponDisfluencyDataset(dev_examples, tokenizer, max_len=MAX_LEN) if dev_examples else None
# eval_dataset  = KsponDisfluencyDataset(eval_examples, tokenizer, max_len=MAX_LEN) if eval_examples else None

# BATCH_SIZE = 8

# def collate_fn(batch):
#     # 이미 padding="max_length"로 맞춰놨기 때문에 단순 스택
#     input_ids = torch.stack([b["input_ids"] for b in batch])
#     attention_mask = torch.stack([b["attention_mask"] for b in batch])
#     labels = torch.stack([b["labels"] for b in batch])
#     return {
#         "input_ids": input_ids,
#         "attention_mask": attention_mask,
#         "labels": labels,
#     }

# train_loader = DataLoader(
#     train_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     collate_fn=collate_fn,
# )

# dev_loader = DataLoader(
#     dev_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     collate_fn=collate_fn,
# ) if dev_dataset is not None else None

# eval_loader = DataLoader(
#     eval_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     collate_fn=collate_fn,
# ) if eval_dataset is not None else None

# ##############################################
# # 7. BERT 기반 Tagger 모델 정의
# ##############################################

# class BertTagger(nn.Module):
#     def __init__(self, model_name: str, num_labels: int, drop_prob: float = 0.1):
#         super().__init__()
#         self.bert = AutoModel.from_pretrained(model_name)
#         hidden_size = self.bert.config.hidden_size
#         self.dropout = nn.Dropout(drop_prob)
#         self.classifier = nn.Linear(hidden_size, num_labels)

#     def forward(self, input_ids, attention_mask=None, labels=None, class_weights=None):
#         outputs = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#         )
#         # last_hidden_state: (batch, seq_len, hidden)
#         sequence_output = outputs.last_hidden_state
#         sequence_output = self.dropout(sequence_output)
#         logits = self.classifier(sequence_output)  # (batch, seq_len, num_labels)

#         loss = None
#         if labels is not None:
#             # CrossEntropy: (N, C) vs (N,)
#             loss_fct = nn.CrossEntropyLoss(
#                 weight=class_weights.to(logits.device) if class_weights is not None else None,
#                 ignore_index=-100,
#             )
#             loss = loss_fct(
#                 logits.view(-1, logits.size(-1)),   # (batch*seq_len, num_labels)
#                 labels.view(-1)                      # (batch*seq_len,)
#             )
#         return logits, loss

# ##############################################
# # 8. 모델 / 옵티마이저 / 스케줄러
# ##############################################

# model = BertTagger(MODEL_NAME, num_labels=len(LABEL_LIST))
# model.to(device)

# LR = 5e-5
# EPOCHS = 3   # 예시로 3 epoch

# optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# # 스케줄러 (linear warmup)
# num_training_steps = len(train_loader) * EPOCHS
# num_warmup_steps = int(0.1 * num_training_steps)

# scheduler = get_linear_schedule_with_warmup(
#     optimizer,
#     num_warmup_steps=num_warmup_steps,
#     num_training_steps=num_training_steps,
# )

# ##############################################
# # 9. 평가 함수 (macro F1 + 보고서 + 파일 저장)
# ##############################################

# def evaluate(model, data_loader, desc="valid", epoch=None, log_dir=None):
#     """
#     desc: 'dev', 'train', 'eval' 등 구분용 문자열
#     epoch: 저장 시 파일 이름에 사용할 epoch 번호 (int)
#     log_dir: 보고서를 저장할 폴더 경로 (str 또는 Path)
#     """
#     model.eval()
#     all_preds = []
#     all_labels = []

#     with torch.no_grad():
#         for batch in tqdm(data_loader, desc=desc):
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             labels = batch["labels"].to(device)

#             logits, _ = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 labels=None,
#                 class_weights=None,
#             )

#             # (batch, seq_len, num_labels)
#             preds = torch.argmax(logits, dim=-1)

#             # -100(ignore) 위치 제거 후 flatten
#             for p, l in zip(preds, labels):
#                 p = p.cpu().numpy()
#                 l = l.cpu().numpy()
#                 for pi, li in zip(p, l):
#                     if li == -100:
#                         continue
#                     all_preds.append(int(pi))
#                     all_labels.append(int(li))

#     # macro F1 (전체 라벨 기준)
#     f1_macro = f1_score(all_labels, all_preds, average="macro")
#     print(f"[{desc}] macro F1 (all labels): {f1_macro:.4f}")

#     # FIL/REP/AMB만 따로 보고 싶으면:
#     target_indices = [label2id[x] for x in ["B-FIL","I-FIL","B-REP","I-REP","B-AMB","I-AMB"]]
#     f1_events = f1_score(
#         all_labels, all_preds,
#         labels=target_indices,
#         average="macro",
#         zero_division=0,
#     )
#     print(f"[{desc}] macro F1 (disfluency labels only): {f1_events:.4f}")

#     # classification_report 문자열로 생성
#     report_str = classification_report(
#         all_labels,
#         all_preds,
#         target_names=[id2label[i] for i in range(len(LABEL_LIST))],
#         digits=3,
#         zero_division=0,
#     )

#     print("\n[Classification Report]")
#     print(report_str)

#     # epoch & log_dir가 주어졌으면 파일로 저장
#     if log_dir is not None and epoch is not None:
#         log_dir = Path(log_dir)
#         log_dir.mkdir(parents=True, exist_ok=True)
#         log_path = log_dir / f"{desc}_epoch{epoch:02d}.txt"
#         with open(log_path, "w", encoding="utf-8") as f:
#             f.write(f"[{desc}] macro F1 (all labels): {f1_macro:.4f}\n")
#             f.write(f"[{desc}] macro F1 (disfluency labels only): {f1_events:.4f}\n\n")
#             f.write(report_str)
#         print(f"[INFO] Saved classification report to {log_path}")

#     return f1_macro, f1_events

# ##############################################
# # 10. 학습 루프
# ##############################################

# best_dev_f1 = 0.0
# save_dir = Path("checkpoints_kspon_disfluency")
# save_dir.mkdir(parents=True, exist_ok=True)

# # dev report 저장 폴더
# DEV_REPORT_DIR = Path("logs/dev_reports")

# for epoch in range(1, EPOCHS + 1):
#     model.train()
#     epoch_loss = 0.0

#     print(f"\n===== Epoch {epoch}/{EPOCHS} =====")
#     for batch in tqdm(train_loader, desc="train"):
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         labels = batch["labels"].to(device)

#         optimizer.zero_grad()

#         logits, loss = model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             labels=labels,
#             class_weights=class_weights,  # class weight 적용
#         )

#         loss.backward()
#         # gradient clipping (폭발 방지)
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
#         scheduler.step()

#         epoch_loss += loss.item()

#     avg_loss = epoch_loss / len(train_loader)
#     print(f"[Epoch {epoch}] train loss: {avg_loss:.4f}")

#     # dev 평가
#     if dev_loader is not None:
#         dev_f1_macro, dev_f1_events = evaluate(
#             model,
#             dev_loader,
#             desc="dev",
#             epoch=epoch,
#             log_dir=DEV_REPORT_DIR,  # ← 여기서 epoch별 dev report 텍스트 저장
#         )

#         # dev F1 (disfluency labels 기준)로 best 모델 저장
#         if dev_f1_events > best_dev_f1:
#             best_dev_f1 = dev_f1_events
#             save_path = save_dir / "best_model.pt"
#             torch.save({
#                 "model_state_dict": model.state_dict(),
#                 "epoch": epoch,
#                 "dev_f1_events": dev_f1_events,
#                 "label2id": label2id,
#                 "id2label": id2label,
#             }, save_path)
#             print(f"→ Best model updated. Saved to {save_path}")
#     else:
#         # dev 세트가 없다면 train 성능만 기록
#         _ = evaluate(model, train_loader, desc="train_eval")

# ##############################################
# # 11. 저장된 best 모델로 eval 세트 평가 (선택)
# ##############################################

# best_ckpt = save_dir / "best_model.pt"
# if best_ckpt.exists() and eval_loader is not None:
#     ckpt = torch.load(best_ckpt, map_location=device)
#     model.load_state_dict(ckpt["model_state_dict"])
#     print(f"\nLoaded best model from epoch {ckpt['epoch']}, dev_f1_events={ckpt['dev_f1_events']:.4f}")
#     # eval 결과는 필요하면 따로 파일로 저장해도 됨
#     _ = evaluate(model, eval_loader, desc="eval")
# else:
#     print("\nNo eval set or best checkpoint not found.")

# [tmux로 이 스크립트 실행하는 방법]
#
# 1) 서버 터미널에서 세션 생성
#    tmux new -s kspon_train
#
# 2) 새로 열린 tmux 창 안에서 (conda 환경 활성화 후)
#    conda activate jws
#    cd /home/USER/프로젝트경로   # train_disfluency.py 위치로 이동
#    python train_disfluency.py 2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log
#
# 3) 세션 분리(백그라운드로 보내기)
#    Ctrl + b, d
#
# 4) 다시 접속
#    tmux attach -t kspon_train
#
# 이렇게 하면 SSH가 끊겨도 tmux 세션 안에서 학습은 계속 진행됩니다.

##############################################
# 0. 기본 설정 & 라이브러리
##############################################
# import os
# import json
# from pathlib import Path
# from collections import Counter

# import numpy as np
# from tqdm import tqdm

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader

# from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
# from sklearn.metrics import classification_report, f1_score

# # ---- (중요) 마지막 GPU만 사용 ----
# # GPU가 3개(CUDA:0,1,2)라고 가정하고, 물리 GPU 2번만 보이도록 설정
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("device =", device)

# # 재현성 (seed 고정)
# def set_seed(seed: int = 42):
#     import random
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# set_seed(42)

# ##############################################
# # 1. 라벨 정의 (FIL vs O 이진 태깅)
# ##############################################

# # 학습 시에는 FIL만 별도 클래스로 두고 나머지는 모두 O로 취급
# LABEL_LIST = [
#     "O",
#     "B-FIL", "I-FIL",
# ]
# label2id = {label: i for i, label in enumerate(LABEL_LIST)}
# id2label = {i: label for label, i in label2id.items()}

# print("label2id:", label2id)

# def to_binary_label(lab: str) -> str:
#     """
#     원본 라벨(B-FIL/B-REP/B-AMB/...)을
#     FIL vs O 이진 태깅으로 매핑:
#       - B-FIL, I-FIL -> 그대로 유지
#       - 그 외(B-REP, I-REP, B-AMB, I-AMB 등) -> 'O'
#     """
#     if lab in ("B-FIL", "I-FIL"):
#         return lab
#     else:
#         return "O"

# ##############################################
# # 2. JSONL 로딩 함수
# ##############################################

# def load_jsonl(path: Path):
#     data = []
#     with open(path, encoding="utf-8") as f:  # JSONL은 UTF-8로 저장되어 있다고 가정
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             obj = json.loads(line)
#             data.append(obj)
#     return data

# ##############################################
# # 3. Train / Dev / Eval 데이터 로드
# ##############################################

# DATA_DIR = Path("Dataset/kspon_disfluency")  # 필요하면 수정
# train_path = DATA_DIR / "kspon_disfluency_train.jsonl"
# dev_path   = DATA_DIR / "kspon_disfluency_dev.jsonl"          # 없으면 train만 써도 됨
# eval_path  = DATA_DIR / "kspon_disfluency_eval_clean.jsonl"   # 이름은 환경에 맞게 수정

# train_examples = load_jsonl(train_path)
# dev_examples   = load_jsonl(dev_path) if dev_path.exists() else []
# eval_examples  = load_jsonl(eval_path) if eval_path.exists() else []

# print("train size:", len(train_examples))
# print("dev size:", len(dev_examples))
# print("eval size:", len(eval_examples))

# ##############################################
# # 4. class weight 계산 (단어 단위 라벨 기준, FIL vs O)
# ##############################################

# label_counter = Counter()
# for ex in train_examples:
#     # 원본 라벨을 FIL vs O 이진 라벨로 매핑해서 카운트
#     bin_labels = [to_binary_label(lab) for lab in ex["labels"]]
#     label_counter.update(bin_labels)

# print("label counts (binary):", label_counter)

# # balanced weight = N / (K * n_i)
# N = sum(label_counter.values())
# K = len(LABEL_LIST)

# raw_weights = []
# for label in LABEL_LIST:
#     n_i = label_counter.get(label, 1)
#     w_i = N / (K * n_i)
#     raw_weights.append(w_i)

# raw_weights = np.array(raw_weights, dtype=np.float32)

# # 너무 극단적인 차이를 막기 위해 스케일링
# # - O: 항상 1.0
# # - 나머지: 최소 weight가 1.0이 되도록 비례 스케일
# non_o = raw_weights[1:]
# min_non_o = non_o.min()
# scale = 1.0 / min_non_o
# scaled_weights = raw_weights * scale
# scaled_weights[0] = 1.0   # O 클래스는 1.0으로 고정

# class_weights = torch.tensor(scaled_weights, dtype=torch.float)
# print("class_weights:", class_weights)

# ##############################################
# # 5. 토크나이저 & Dataset 정의
# ##############################################

# MODEL_NAME = "klue/roberta-large"   # 전이학습 모델

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# MAX_LEN = 128  # 발화 길이에 맞게 조정 가능

# class KsponDisfluencyDataset(Dataset):
#     """
#     JSONL 예제:
#     {
#       "tokens": ["아", "그래서", ...],
#       "labels": ["B-FIL", "O", "B-REP", ...]  # 원본 다중 라벨
#     }
#     → __getitem__에서 FIL vs O 이진 라벨로 매핑해서 사용
#     """
#     def __init__(self, examples, tokenizer, max_len: int = 128):
#         self.examples = examples
#         self.tokenizer = tokenizer
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.examples)

#     def __getitem__(self, idx):
#         ex = self.examples[idx]
#         tokens = ex["tokens"]
#         orig_labels = ex["labels"]

#         # --- 1) 다중 라벨 → FIL vs O 이진 라벨로 매핑 ---
#         labels = [to_binary_label(lab) for lab in orig_labels]

#         # --- 2) 토크나이저: 단어 리스트 기반 토큰화 ---
#         encoding = self.tokenizer(
#             tokens,
#             is_split_into_words=True,
#             return_tensors="pt",
#             padding="max_length",
#             truncation=True,
#             max_length=self.max_len,
#             return_attention_mask=True,
#         )

#         input_ids = encoding["input_ids"][0]             # (max_len,)
#         attention_mask = encoding["attention_mask"][0]   # (max_len,)

#         # word_ids: 서브워드 → 원래 단어 인덱스 매핑
#         word_ids = encoding.word_ids(0)  # 길이 max_len

#         # --- 3) 라벨을 서브워드에 맞게 확장 ---
#         # - 한 단어의 첫 서브워드에만 라벨 부여
#         # - 나머지 서브워드는 ignore_index(-100)
#         label_ids = []
#         previous_word_id = None
#         for word_id in word_ids:
#             if word_id is None:
#                 label_ids.append(-100)
#             elif word_id != previous_word_id:
#                 label_str = labels[word_id]
#                 label_ids.append(label2id[label_str])
#             else:
#                 label_ids.append(-100)
#             previous_word_id = word_id

#         label_ids = torch.tensor(label_ids, dtype=torch.long)

#         return {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "labels": label_ids,
#         }

# ##############################################
# # 6. DataLoader 정의
# ##############################################

# train_dataset = KsponDisfluencyDataset(train_examples, tokenizer, max_len=MAX_LEN)
# dev_dataset   = KsponDisfluencyDataset(dev_examples, tokenizer, max_len=MAX_LEN) if dev_examples else None
# eval_dataset  = KsponDisfluencyDataset(eval_examples, tokenizer, max_len=MAX_LEN) if eval_examples else None

# BATCH_SIZE = 16

# def collate_fn(batch):
#     # 이미 padding="max_length"로 맞춰놨기 때문에 단순 스택
#     input_ids = torch.stack([b["input_ids"] for b in batch])
#     attention_mask = torch.stack([b["attention_mask"] for b in batch])
#     labels = torch.stack([b["labels"] for b in batch])
#     return {
#         "input_ids": input_ids,
#         "attention_mask": attention_mask,
#         "labels": labels,
#     }

# train_loader = DataLoader(
#     train_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     collate_fn=collate_fn,
# )

# dev_loader = DataLoader(
#     dev_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     collate_fn=collate_fn,
# ) if dev_dataset is not None else None

# eval_loader = DataLoader(
#     eval_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     collate_fn=collate_fn,
# ) if eval_dataset is not None else None

# ##############################################
# # 7. BERT 기반 Tagger 모델 정의
# ##############################################

# class BertTagger(nn.Module):
#     def __init__(self, model_name: str, num_labels: int, drop_prob: float = 0.1):
#         super().__init__()
#         self.bert = AutoModel.from_pretrained(model_name)
#         hidden_size = self.bert.config.hidden_size
#         self.dropout = nn.Dropout(drop_prob)
#         self.classifier = nn.Linear(hidden_size, num_labels)

#     def forward(self, input_ids, attention_mask=None, labels=None, class_weights=None):
#         outputs = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#         )
#         # last_hidden_state: (batch, seq_len, hidden)
#         sequence_output = outputs.last_hidden_state
#         sequence_output = self.dropout(sequence_output)
#         logits = self.classifier(sequence_output)  # (batch, seq_len, num_labels)

#         loss = None
#         if labels is not None:
#             # CrossEntropy: (N, C) vs (N,)
#             loss_fct = nn.CrossEntropyLoss(
#                 weight=class_weights.to(logits.device) if class_weights is not None else None,
#                 ignore_index=-100,
#             )
#             loss = loss_fct(
#                 logits.view(-1, logits.size(-1)),   # (batch*seq_len, num_labels)
#                 labels.view(-1)                      # (batch*seq_len,)
#             )
#         return logits, loss

# ##############################################
# # 8. 모델 / 옵티마이저 / 스케줄러
# ##############################################

# model = BertTagger(MODEL_NAME, num_labels=len(LABEL_LIST))
# model.to(device)

# LR = 5e-5
# EPOCHS = 3   # 예시로 3 epoch

# optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# # 스케줄러 (linear warmup)
# num_training_steps = len(train_loader) * EPOCHS
# num_warmup_steps = int(0.1 * num_training_steps)

# scheduler = get_linear_schedule_with_warmup(
#     optimizer,
#     num_warmup_steps=num_warmup_steps,
#     num_training_steps=num_training_steps,
# )

# ##############################################
# # 9. 평가 함수 (macro F1 + 보고서 + 파일 저장)
# ##############################################

# def evaluate(model, data_loader, desc="valid", epoch=None, log_dir=None):
#     """
#     desc: 'dev', 'train', 'eval' 등 구분용 문자열
#     epoch: 저장 시 파일 이름에 사용할 epoch 번호 (int)
#     log_dir: 보고서를 저장할 폴더 경로 (str 또는 Path)
#     """
#     model.eval()
#     all_preds = []
#     all_labels = []

#     with torch.no_grad():
#         for batch in tqdm(data_loader, desc=desc):
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             labels = batch["labels"].to(device)

#             logits, _ = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 labels=None,
#                 class_weights=None,
#             )

#             # (batch, seq_len, num_labels)
#             preds = torch.argmax(logits, dim=-1)

#             # -100(ignore) 위치 제거 후 flatten
#             for p, l in zip(preds, labels):
#                 p = p.cpu().numpy()
#                 l = l.cpu().numpy()
#                 for pi, li in zip(p, l):
#                     if li == -100:
#                         continue
#                     all_preds.append(int(pi))
#                     all_labels.append(int(li))

#     # macro F1 (전체 라벨 기준)
#     f1_macro = f1_score(all_labels, all_preds, average="macro")
#     print(f"[{desc}] macro F1 (all labels): {f1_macro:.4f}")

#     # FIL만 따로 보고 싶으면: B-FIL, I-FIL
#     target_indices = [label2id[x] for x in ["B-FIL", "I-FIL"]]
#     f1_fil = f1_score(
#         all_labels, all_preds,
#         labels=target_indices,
#         average="macro",
#         zero_division=0,
#     )
#     print(f"[{desc}] macro F1 (FIL only): {f1_fil:.4f}")

#     # classification_report 문자열로 생성
#     report_str = classification_report(
#         all_labels,
#         all_preds,
#         target_names=[id2label[i] for i in range(len(LABEL_LIST))],
#         digits=3,
#         zero_division=0,
#     )

#     print("\n[Classification Report]")
#     print(report_str)

#     # epoch & log_dir가 주어졌으면 파일로 저장
#     if log_dir is not None and epoch is not None:
#         log_dir = Path(log_dir)
#         log_dir.mkdir(parents=True, exist_ok=True)
#         log_path = log_dir / f"{desc}_epoch{epoch:02d}.txt"
#         with open(log_path, "w", encoding="utf-8") as f:
#             f.write(f"[{desc}] macro F1 (all labels): {f1_macro:.4f}\n")
#             f.write(f"[{desc}] macro F1 (FIL only): {f1_fil:.4f}\n\n")
#             f.write(report_str)
#         print(f"[INFO] Saved classification report to {log_path}")

#     return f1_macro, f1_fil

# ##############################################
# # 10. 학습 루프 (에폭별 체크포인트 + dev report 저장)
# ##############################################

# # best_dev_f1는 "FIL 라벨만 기준" F1로 관리
# best_dev_f1 = -1.0   # 최소 한 번은 best_model이 저장되도록 -1로 설정

# save_dir = Path("checkpoints_kspon_disfluency_2")
# save_dir.mkdir(parents=True, exist_ok=True)

# # dev report 저장 폴더
# DEV_REPORT_DIR = Path("logs/dev_reports")

# for epoch in range(1, EPOCHS + 1):
#     model.train()
#     epoch_loss = 0.0

#     print(f"\n===== Epoch {epoch}/{EPOCHS} =====")
#     for batch in tqdm(train_loader, desc="train"):
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         labels = batch["labels"].to(device)

#         optimizer.zero_grad()

#         logits, loss = model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             labels=labels,
#             class_weights=class_weights,  # class weight 적용
#         )

#         loss.backward()
#         # gradient clipping (폭발 방지)
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
#         scheduler.step()

#         epoch_loss += loss.item()

#     avg_loss = epoch_loss / len(train_loader)
#     print(f"[Epoch {epoch}] train loss: {avg_loss:.4f}")

#     # (1) 에폭별 체크포인트 항상 저장
#     epoch_ckpt_path = save_dir / f"epoch_{epoch:02d}.pt"
#     torch.save({
#         "model_state_dict": model.state_dict(),
#         "epoch": epoch,
#         "train_loss": avg_loss,
#         "label2id": label2id,
#         "id2label": id2label,
#     }, epoch_ckpt_path)
#     print(f"→ Saved epoch checkpoint to {epoch_ckpt_path}")

#     # (2) dev 평가 + best 모델 저장
#     if dev_loader is not None:
#         dev_f1_macro, dev_f1_fil = evaluate(
#             model,
#             dev_loader,
#             desc="dev",
#             epoch=epoch,
#             log_dir=DEV_REPORT_DIR,  # epoch별 dev report 텍스트 저장
#         )

#         # dev FIL F1 기준으로 best 모델 저장
#         if dev_f1_fil > best_dev_f1:
#             best_dev_f1 = dev_f1_fil
#             save_path = save_dir / "best_model.pt"
#             torch.save({
#                 "model_state_dict": model.state_dict(),
#                 "epoch": epoch,
#                 "dev_f1_fil": dev_f1_fil,
#                 "label2id": label2id,
#                 "id2label": id2label,
#             }, save_path)
#             print(f"→ Best model updated. Saved to {save_path}")
#     else:
#         # dev 세트가 없다면 train 성능만 기록
#         _ = evaluate(model, train_loader, desc="train_eval")

# ##############################################
# # 11. 저장된 best 모델로 eval 세트 평가 (선택)
# ##############################################

# best_ckpt = save_dir / "best_model.pt"
# if best_ckpt.exists() and eval_loader is not None:
#     ckpt = torch.load(best_ckpt, map_location=device)
#     model.load_state_dict(ckpt["model_state_dict"])
#     print(f"\nLoaded best model from epoch {ckpt['epoch']}, dev_f1_fil={ckpt['dev_f1_fil']:.4f}")
#     _ = evaluate(model, eval_loader, desc="eval")
# else:
#     print("\nNo eval set or best checkpoint not found.")

# [tmux로 이 스크립트 실행하는 방법]
#
# 1) 서버 터미널에서 세션 생성
#    tmux new -s kspon_train
#
# 2) 새로 열린 tmux 창 안에서 (conda 환경 활성화 후)
#    conda activate jws
#    cd /home/USER/프로젝트경로   # train_disfluency.py 위치로 이동
#    python train_disfluency.py 2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log
#
# 3) 세션 분리(백그라운드로 보내기)
#    Ctrl + b, d
#
# 4) 다시 접속
#    tmux attach -t kspon_train
#
# 이렇게 하면 SSH가 끊겨도 tmux 세션 안에서 학습은 계속 진행됩니다.

##############################################
# 0. 기본 설정 & 라이브러리
# ##############################################
# import os
# import json
# from pathlib import Path
# from collections import Counter

# import numpy as np
# from tqdm import tqdm

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader

# from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
# from sklearn.metrics import classification_report, f1_score

# # ---- (중요) GPU 1, 2 두 개 사용 ----
# #   물리 GPU 0,1,2 중에서 1,2만 보이도록 설정
# #   (논리적으로는 cuda:0, cuda:1로 보입니다.)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("device =", device)
# print("visible cuda device count =", torch.cuda.device_count())

# # 재현성 (seed 고정)
# def set_seed(seed: int = 42):
#     import random
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# set_seed(42)

# ##############################################
# # 1. 라벨 정의 (FIL vs O 이진 태깅)
# ##############################################

# # 학습 시에는 FIL만 별도 클래스로 두고 나머지는 모두 O로 취급
# LABEL_LIST = [
#     "O",
#     "B-FIL", "I-FIL",
# ]
# label2id = {label: i for i, label in enumerate(LABEL_LIST)}
# id2label = {i: label for label, i in label2id.items()}

# print("label2id:", label2id)

# def to_binary_label(lab: str) -> str:
#     """
#     원본 라벨(B-FIL/B-REP/B-AMB/...)을
#     FIL vs O 이진 태깅으로 매핑:
#       - B-FIL, I-FIL -> 그대로 유지
#       - 그 외(B-REP, I-REP, B-AMB, I-AMB 등) -> 'O'
#     """
#     if lab in ("B-FIL", "I-FIL"):
#         return lab
#     else:
#         return "O"

# ##############################################
# # 2. JSONL 로딩 함수
# ##############################################

# def load_jsonl(path: Path):
#     data = []
#     with open(path, encoding="utf-8") as f:  # JSONL은 UTF-8로 저장되어 있다고 가정
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             obj = json.loads(line)
#             data.append(obj)
#     return data

# ##############################################
# # 3. Train / Dev / Eval 데이터 로드
# ##############################################

# DATA_DIR = Path("Dataset/kspon_disfluency")  # 필요하면 수정
# train_path = DATA_DIR / "kspon_disfluency_train.jsonl"
# dev_path   = DATA_DIR / "kspon_disfluency_dev.jsonl"          # 없으면 train만 써도 됨
# eval_path  = DATA_DIR / "kspon_disfluency_eval_clean.jsonl"   # 이름은 환경에 맞게 수정

# train_examples = load_jsonl(train_path)
# dev_examples   = load_jsonl(dev_path) if dev_path.exists() else []
# eval_examples  = load_jsonl(eval_path) if eval_path.exists() else []

# print("train size:", len(train_examples))
# print("dev size:", len(dev_examples))
# print("eval size:", len(eval_examples))

# ##############################################
# # 4. class weight 계산 (단어 단위 라벨 기준, FIL vs O)
# ##############################################

# label_counter = Counter()
# for ex in train_examples:
#     # 원본 라벨을 FIL vs O 이진 라벨로 매핑해서 카운트
#     bin_labels = [to_binary_label(lab) for lab in ex["labels"]]
#     label_counter.update(bin_labels)

# print("label counts (binary):", label_counter)

# # balanced weight = N / (K * n_i)
# N = sum(label_counter.values())
# K = len(LABEL_LIST)

# raw_weights = []
# for label in LABEL_LIST:
#     n_i = label_counter.get(label, 1)
#     w_i = N / (K * n_i)
#     raw_weights.append(w_i)

# raw_weights = np.array(raw_weights, dtype=np.float32)

# # 너무 극단적인 차이를 막기 위해 스케일링
# # - O: 항상 1.0
# # - 나머지: 최소 weight가 1.0이 되도록 비례 스케일
# non_o = raw_weights[1:]
# min_non_o = non_o.min()
# scale = 1.0 / min_non_o
# scaled_weights = raw_weights * scale
# scaled_weights[0] = 1.0   # O 클래스는 1.0으로 고정

# class_weights = torch.tensor(scaled_weights, dtype=torch.float)
# print("class_weights:", class_weights)

# ##############################################
# # 5. 토크나이저 & Dataset 정의
# ##############################################

# MODEL_NAME = "klue/roberta-large"   # 전이학습 모델

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# MAX_LEN = 128  # 발화 길이에 맞게 조정 가능

# class KsponDisfluencyDataset(Dataset):
#     """
#     JSONL 예제:
#     {
#       "tokens": ["아", "그래서", ...],
#       "labels": ["B-FIL", "O", "B-REP", ...]  # 원본 다중 라벨
#     }
#     → __getitem__에서 FIL vs O 이진 라벨로 매핑해서 사용
#     """
#     def __init__(self, examples, tokenizer, max_len: int = 128):
#         self.examples = examples
#         self.tokenizer = tokenizer
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.examples)

#     def __getitem__(self, idx):
#         ex = self.examples[idx]
#         tokens = ex["tokens"]
#         orig_labels = ex["labels"]

#         # --- 1) 다중 라벨 → FIL vs O 이진 라벨로 매핑 ---
#         labels = [to_binary_label(lab) for lab in orig_labels]

#         # --- 2) 토크나이저: 단어 리스트 기반 토큰화 ---
#         encoding = self.tokenizer(
#             tokens,
#             is_split_into_words=True,
#             return_tensors="pt",
#             padding="max_length",
#             truncation=True,
#             max_length=self.max_len,
#             return_attention_mask=True,
#         )

#         input_ids = encoding["input_ids"][0]             # (max_len,)
#         attention_mask = encoding["attention_mask"][0]   # (max_len,)

#         # word_ids: 서브워드 → 원래 단어 인덱스 매핑
#         word_ids = encoding.word_ids(0)  # 길이 max_len

#         # --- 3) 라벨을 서브워드에 맞게 확장 ---
#         # - 한 단어의 첫 서브워드에만 라벨 부여
#         # - 나머지 서브워드는 ignore_index(-100)
#         label_ids = []
#         previous_word_id = None
#         for word_id in word_ids:
#             if word_id is None:
#                 label_ids.append(-100)
#             elif word_id != previous_word_id:
#                 label_str = labels[word_id]
#                 label_ids.append(label2id[label_str])
#             else:
#                 label_ids.append(-100)
#             previous_word_id = word_id

#         label_ids = torch.tensor(label_ids, dtype=torch.long)

#         return {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "labels": label_ids,
#         }

# ##############################################
# # 6. DataLoader 정의
# ##############################################

# train_dataset = KsponDisfluencyDataset(train_examples, tokenizer, max_len=MAX_LEN)
# dev_dataset   = KsponDisfluencyDataset(dev_examples, tokenizer, max_len=MAX_LEN) if dev_examples else None
# eval_dataset  = KsponDisfluencyDataset(eval_examples, tokenizer, max_len=MAX_LEN) if eval_examples else None

# BATCH_SIZE = 16

# def collate_fn(batch):
#     # 이미 padding="max_length"로 맞춰놨기 때문에 단순 스택
#     input_ids = torch.stack([b["input_ids"] for b in batch])
#     attention_mask = torch.stack([b["attention_mask"] for b in batch])
#     labels = torch.stack([b["labels"] for b in batch])
#     return {
#         "input_ids": input_ids,
#         "attention_mask": attention_mask,
#         "labels": labels,
#     }

# train_loader = DataLoader(
#     train_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     collate_fn=collate_fn,
# )

# dev_loader = DataLoader(
#     dev_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     collate_fn=collate_fn,
# ) if dev_dataset is not None else None

# eval_loader = DataLoader(
#     eval_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     collate_fn=collate_fn,
# ) if eval_dataset is not None else None

# ##############################################
# # 7. BERT 기반 Tagger 모델 정의
# ##############################################

# class BertTagger(nn.Module):
#     def __init__(self, model_name: str, num_labels: int, drop_prob: float = 0.1):
#         super().__init__()
#         self.bert = AutoModel.from_pretrained(model_name)
#         hidden_size = self.bert.config.hidden_size
#         self.dropout = nn.Dropout(drop_prob)
#         self.classifier = nn.Linear(hidden_size, num_labels)

#     def forward(self, input_ids, attention_mask=None, labels=None, class_weights=None):
#         outputs = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#         )
#         # last_hidden_state: (batch, seq_len, hidden)
#         sequence_output = outputs.last_hidden_state
#         sequence_output = self.dropout(sequence_output)
#         logits = self.classifier(sequence_output)  # (batch, seq_len, num_labels)

#         loss = None
#         if labels is not None:
#             # CrossEntropy: (N, C) vs (N,)
#             loss_fct = nn.CrossEntropyLoss(
#                 weight=class_weights.to(logits.device) if class_weights is not None else None,
#                 ignore_index=-100,
#             )
#             loss = loss_fct(
#                 logits.view(-1, logits.size(-1)),   # (batch*seq_len, num_labels)
#                 labels.view(-1)                      # (batch*seq_len,)
#             )
#         return logits, loss

# ##############################################
# # 8. 모델 / 옵티마이저 / 스케줄러 (멀티 GPU)
# ##############################################

# model = BertTagger(MODEL_NAME, num_labels=len(LABEL_LIST))

# # 멀티 GPU (GPU 1,2 두 개) 사용: DataParallel
# if torch.cuda.device_count() > 1:
#     print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
#     model = nn.DataParallel(model)

# model.to(device)

# LR = 5e-5
# EPOCHS = 3   # 예시로 3 epoch

# optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# # 스케줄러 (linear warmup)
# num_training_steps = len(train_loader) * EPOCHS
# num_warmup_steps = int(0.1 * num_training_steps)

# scheduler = get_linear_schedule_with_warmup(
#     optimizer,
#     num_warmup_steps=num_warmup_steps,
#     num_training_steps=num_training_steps,
# )

# ##############################################
# # 8-1. DataParallel 상태에서 save/load 헬퍼
# ##############################################

# def get_state_dict(m: nn.Module):
#     """DataParallel 여부에 상관없이 항상 base 모델의 state_dict를 얻기 위한 헬퍼"""
#     if isinstance(m, nn.DataParallel):
#         return m.module.state_dict()
#     return m.state_dict()

# def load_state_dict(m: nn.Module, state_dict):
#     """DataParallel 여부에 상관없이 적절한 위치에 state_dict를 로드"""
#     if isinstance(m, nn.DataParallel):
#         m.module.load_state_dict(state_dict)
#     else:
#         m.load_state_dict(state_dict)

# ##############################################
# # 9. 평가 함수 (macro F1 + 보고서 + 파일 저장)
# ##############################################

# def evaluate(model, data_loader, desc="valid", epoch=None, log_dir=None):
#     """
#     desc: 'dev', 'train', 'eval' 등 구분용 문자열
#     epoch: 저장 시 파일 이름에 사용할 epoch 번호 (int)
#     log_dir: 보고서를 저장할 폴더 경로 (str 또는 Path)
#     """
#     model.eval()
#     all_preds = []
#     all_labels = []

#     with torch.no_grad():
#         for batch in tqdm(data_loader, desc=desc):
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             labels = batch["labels"].to(device)

#             logits, _ = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 labels=None,
#                 class_weights=None,
#             )

#             # (batch, seq_len, num_labels)
#             preds = torch.argmax(logits, dim=-1)

#             # -100(ignore) 위치 제거 후 flatten
#             for p, l in zip(preds, labels):
#                 p = p.cpu().numpy()
#                 l = l.cpu().numpy()
#                 for pi, li in zip(p, l):
#                     if li == -100:
#                         continue
#                     all_preds.append(int(pi))
#                     all_labels.append(int(li))

#     # macro F1 (전체 라벨 기준)
#     f1_macro = f1_score(all_labels, all_preds, average="macro")
#     print(f"[{desc}] macro F1 (all labels): {f1_macro:.4f}")

#     # FIL만 따로 보고 싶으면: B-FIL, I-FIL
#     target_indices = [label2id[x] for x in ["B-FIL", "I-FIL"]]
#     f1_fil = f1_score(
#         all_labels, all_preds,
#         labels=target_indices,
#         average="macro",
#         zero_division=0,
#     )
#     print(f"[{desc}] macro F1 (FIL only): {f1_fil:.4f}")

#     # classification_report 문자열로 생성
#     report_str = classification_report(
#         all_labels,
#         all_preds,
#         target_names=[id2label[i] for i in range(len(LABEL_LIST))],
#         digits=3,
#         zero_division=0,
#     )

#     print("\n[Classification Report]")
#     print(report_str)

#     # epoch & log_dir가 주어졌으면 파일로 저장
#     if log_dir is not None and epoch is not None:
#         log_dir = Path(log_dir)
#         log_dir.mkdir(parents=True, exist_ok=True)
#         log_path = log_dir / f"{desc}_epoch{epoch:02d}.txt"
#         with open(log_path, "w", encoding="utf-8") as f:
#             f.write(f"[{desc}] macro F1 (all labels): {f1_macro:.4f}\n")
#             f.write(f"[{desc}] macro F1 (FIL only): {f1_fil:.4f}\n\n")
#             f.write(report_str)
#         print(f"[INFO] Saved classification report to {log_path}")

#     return f1_macro, f1_fil

# ##############################################
# # 10. 학습 루프 (에폭별 체크포인트 + dev report 저장)
# ##############################################

# # best_dev_f1는 "FIL 라벨만 기준" F1로 관리
# best_dev_f1 = -1.0   # 최소 한 번은 best_model이 저장되도록 -1로 설정

# save_dir = Path("checkpoints_kspon_disfluency_2")
# save_dir.mkdir(parents=True, exist_ok=True)

# # dev report 저장 폴더
# DEV_REPORT_DIR = Path("logs/dev_reports")

# for epoch in range(1, EPOCHS + 1):
#     model.train()
#     epoch_loss = 0.0

#     print(f"\n===== Epoch {epoch}/{EPOCHS} =====")
#     for batch in tqdm(train_loader, desc="train"):
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         labels = batch["labels"].to(device)

#         optimizer.zero_grad()

#         logits, loss = model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             labels=labels,
#             class_weights=class_weights,  # class weight 적용
#         )

#         loss.backward()
#         # gradient clipping (폭발 방지)
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
#         scheduler.step()

#         epoch_loss += loss.item()

#     avg_loss = epoch_loss / len(train_loader)
#     print(f"[Epoch {epoch}] train loss: {avg_loss:.4f}")

#     # (1) 에폭별 체크포인트 항상 저장 (base state_dict 저장)
#     epoch_ckpt_path = save_dir / f"epoch_{epoch:02d}.pt"
#     state_dict = get_state_dict(model)
#     torch.save({
#         "model_state_dict": state_dict,
#         "epoch": epoch,
#         "train_loss": avg_loss,
#         "label2id": label2id,
#         "id2label": id2label,
#     }, epoch_ckpt_path)
#     print(f"→ Saved epoch checkpoint to {epoch_ckpt_path}")

#     # (2) dev 평가 + best 모델 저장
#     if dev_loader is not None:
#         dev_f1_macro, dev_f1_fil = evaluate(
#             model,
#             dev_loader,
#             desc="dev",
#             epoch=epoch,
#             log_dir=DEV_REPORT_DIR,  # epoch별 dev report 텍스트 저장
#         )

#         # dev FIL F1 기준으로 best 모델 저장
#         if dev_f1_fil > best_dev_f1:
#             best_dev_f1 = dev_f1_fil
#             save_path = save_dir / "best_model.pt"
#             state_dict = get_state_dict(model)
#             torch.save({
#                 "model_state_dict": state_dict,
#                 "epoch": epoch,
#                 "dev_f1_fil": dev_f1_fil,
#                 "label2id": label2id,
#                 "id2label": id2label,
#             }, save_path)
#             print(f"→ Best model updated. Saved to {save_path}")
#     else:
#         # dev 세트가 없다면 train 성능만 기록
#         _ = evaluate(model, train_loader, desc="train_eval")

# ##############################################
# # 11. 저장된 best 모델로 eval 세트 평가 (선택)
# ##############################################

# best_ckpt = save_dir / "best_model.pt"
# if best_ckpt.exists() and eval_loader is not None:
#     ckpt = torch.load(best_ckpt, map_location=device)
#     load_state_dict(model, ckpt["model_state_dict"])
#     print(f"\nLoaded best model from epoch {ckpt['epoch']}, dev_f1_fil={ckpt['dev_f1_fil']:.4f}")
#     _ = evaluate(model, eval_loader, desc="eval")
# else:
#     print("\nNo eval set or best checkpoint not found.")

# [tmux로 이 스크립트 실행하는 방법]
#
# 1) 서버 터미널에서 세션 생성
#    tmux new -s kspon_train
#
# 2) 새로 열린 tmux 창 안에서 (conda 환경 활성화 후)
#    conda activate jws
#    cd /home/USER/프로젝트경로   # train_disfluency.py 위치
#    python train_disfluency.py 2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log
#
# 3) 세션 분리(백그라운드로 보내기)
#    Ctrl + b, d
#
# 4) 다시 접속
#    tmux attach -t kspon_train
#
# 이렇게 하면 SSH가 끊겨도 tmux 세션 안에서 학습은 계속 진행됩니다.

##############################################
# 0. 기본 설정 & 라이브러리
##############################################
import os
import json
from pathlib import Path
from collections import Counter

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, f1_score

# ---- GPU 1, 2 두 개 사용 ----
# 물리 GPU 0,1,2 중에서 1,2만 보이도록 설정
# (논리적으로는 cuda:0, cuda:1로 보입니다.)
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device =", device)
print("visible cuda device count =", torch.cuda.device_count())

# 재현성 (seed 고정)
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

##############################################
# 1. 라벨 정의 (FIL vs O 이진 태깅)
##############################################

# FIL vs O 이진 태깅
LABEL_LIST = [
    "O",
    "B-FIL", "I-FIL",
]
label2id = {label: i for i, label in enumerate(LABEL_LIST)}
id2label = {i: label for label, i in label2id.items()}

print("label2id:", label2id)

def to_binary_label(lab: str) -> str:
    """
    원본 라벨(B-FIL/B-REP/B-AMB/...)을
    FIL vs O 이진 태깅으로 매핑:
      - B-FIL, I-FIL -> 그대로 유지
      - 그 외(B-REP, I-REP, B-AMB, I-AMB 등) -> 'O'
    """
    if lab in ("B-FIL", "I-FIL"):
        return lab
    else:
        return "O"

##############################################
# 2. JSONL 로딩 함수
##############################################

def load_jsonl(path: Path):
    data = []
    with open(path, encoding="utf-8") as f:  # JSONL은 UTF-8로 저장되어 있다고 가정
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            data.append(obj)
    return data

##############################################
# 3. Train / Dev / Eval 데이터 로드
##############################################

DATA_DIR = Path("Dataset/kspon_disfluency")  # 필요하면 수정
train_path = DATA_DIR / "kspon_disfluency_train.jsonl"
dev_path   = DATA_DIR / "kspon_disfluency_dev.jsonl"
eval_path  = DATA_DIR / "kspon_disfluency_eval_clean.jsonl"

train_examples = load_jsonl(train_path)
dev_examples   = load_jsonl(dev_path) if dev_path.exists() else []
eval_examples  = load_jsonl(eval_path) if eval_path.exists() else []

print("train size:", len(train_examples))
print("dev size:", len(dev_examples))
print("eval size:", len(eval_examples))

##############################################
# 4. (선택) class weight 계산 (지금은 참고만, Loss에는 사용 X)
##############################################

label_counter = Counter()
for ex in train_examples:
    bin_labels = [to_binary_label(lab) for lab in ex["labels"]]
    label_counter.update(bin_labels)

print("label counts (binary):", label_counter)

N = sum(label_counter.values())
K = len(LABEL_LIST)

raw_weights = []
for label in LABEL_LIST:
    n_i = label_counter.get(label, 1)
    w_i = N / (K * n_i)
    raw_weights.append(w_i)

raw_weights = np.array(raw_weights, dtype=np.float32)

non_o = raw_weights[1:]
min_non_o = non_o.min()
scale = 1.0 / min_non_o
scaled_weights = raw_weights * scale
scaled_weights[0] = 1.0

class_weights = torch.tensor(scaled_weights, dtype=torch.float)
print("class_weights (not used in loss now):", class_weights)

##############################################
# 5. 토크나이저 & Dataset 정의
##############################################

MODEL_NAME = "klue/roberta-large"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

MAX_LEN = 128  # 필요시 조정

class KsponDisfluencyDataset(Dataset):
    """
    JSONL 예제:
    {
      "tokens": ["아", "그래서", ...],
      "labels": ["B-FIL", "O", "B-REP", ...]  # 원본 다중 라벨
    }
    → __getitem__에서 FIL vs O 이진 라벨로 매핑해서 사용
    """
    def __init__(self, examples, tokenizer, max_len: int = 128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        tokens = ex["tokens"]
        orig_labels = ex["labels"]

        # 1) 다중 라벨 → FIL vs O 이진 라벨
        labels = [to_binary_label(lab) for lab in orig_labels]

        # 2) 토크나이저
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_attention_mask=True,
        )

        input_ids = encoding["input_ids"][0]
        attention_mask = encoding["attention_mask"][0]

        # word_ids: 서브워드 → 원래 단어 인덱스 매핑
        word_ids = encoding.word_ids(0)

        # 3) 라벨을 서브워드에 맞게 확장
        label_ids = []
        previous_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                label_str = labels[word_id]
                label_ids.append(label2id[label_str])
            else:
                label_ids.append(-100)
            previous_word_id = word_id

        label_ids = torch.tensor(label_ids, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_ids,
        }

##############################################
# 6. DataLoader 정의
##############################################

train_dataset = KsponDisfluencyDataset(train_examples, tokenizer, max_len=MAX_LEN)
dev_dataset   = KsponDisfluencyDataset(dev_examples, tokenizer, max_len=MAX_LEN) if dev_examples else None
eval_dataset  = KsponDisfluencyDataset(eval_examples, tokenizer, max_len=MAX_LEN) if eval_examples else None

BATCH_SIZE = 128

def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
) if dev_dataset is not None else None

eval_loader = DataLoader(
    eval_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
) if eval_dataset is not None else None

##############################################
# 7. BERT 기반 Tagger 모델 정의
##############################################

class BertTagger(nn.Module):
    def __init__(self, model_name: str, num_labels: int, drop_prob: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(drop_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # (batch, seq_len, num_labels)

        loss = None
        if labels is not None:
            # → 여기서는 weight 안 씀 (에러 방지 + baseline 확인)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
        return logits, loss

##############################################
# 8. 모델 / 옵티마이저 / 스케줄러 (멀티 GPU)
##############################################

model = BertTagger(MODEL_NAME, num_labels=len(LABEL_LIST))

# 멀티 GPU 사용
if torch.cuda.device_count() > 1:
    print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model.to(device)

LR = 5e-5
EPOCHS = 3

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

num_training_steps = len(train_loader) * EPOCHS
num_warmup_steps = int(0.1 * num_training_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)

def get_state_dict(m: nn.Module):
    if isinstance(m, nn.DataParallel):
        return m.module.state_dict()
    return m.state_dict()

def load_state_dict(m: nn.Module, state_dict):
    if isinstance(m, nn.DataParallel):
        m.module.load_state_dict(state_dict)
    else:
        m.load_state_dict(state_dict)

##############################################
# 9. 평가 함수 (macro F1 + 보고서 + 파일 저장)
##############################################

def evaluate(model, data_loader, desc="valid", epoch=None, log_dir=None):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=desc):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
            )

            preds = torch.argmax(logits, dim=-1)

            for p, l in zip(preds, labels):
                p = p.cpu().numpy()
                l = l.cpu().numpy()
                for pi, li in zip(p, l):
                    if li == -100:
                        continue
                    all_preds.append(int(pi))
                    all_labels.append(int(li))

    f1_macro = f1_score(all_labels, all_preds, average="macro")
    print(f"[{desc}] macro F1 (all labels): {f1_macro:.4f}")

    target_indices = [label2id[x] for x in ["B-FIL", "I-FIL"]]
    f1_fil = f1_score(
        all_labels, all_preds,
        labels=target_indices,
        average="macro",
        zero_division=0,
    )
    print(f"[{desc}] macro F1 (FIL only): {f1_fil:.4f}")

    report_str = classification_report(
        all_labels,
        all_preds,
        target_names=[id2label[i] for i in range(len(LABEL_LIST))],
        digits=3,
        zero_division=0,
    )

    print("\n[Classification Report]")
    print(report_str)

    if log_dir is not None and epoch is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{desc}_epoch{epoch:02d}.txt"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"[{desc}] macro F1 (all labels): {f1_macro:.4f}\n")
            f.write(f"[{desc}] macro F1 (FIL only): {f1_fil:.4f}\n\n")
            f.write(report_str)
        print(f"[INFO] Saved classification report to {log_path}")

    return f1_macro, f1_fil

##############################################
# 10. 학습 루프
##############################################

best_dev_f1 = -1.0

save_dir = Path("checkpoints_kspon_disfluency")
save_dir.mkdir(parents=True, exist_ok=True)

DEV_REPORT_DIR = Path("logs/dev_reports")

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0

    print(f"\n===== Epoch {epoch}/{EPOCHS} =====")
    for batch in tqdm(train_loader, desc="train"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # logits, loss = model(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     labels=labels,
        # )

        # loss.backward()
        logits, loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        # DataParallel 사용 시 각 GPU loss가 (ngpu,) 텐서로 모여 들어오기 때문에
        # 항상 스칼라로 줄여준다.
        if isinstance(loss, torch.Tensor) and loss.dim() > 0:
            loss = loss.mean()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    print(f"[Epoch {epoch}] train loss: {avg_loss:.4f}")

    # (1) 에폭별 체크포인트 저장
    epoch_ckpt_path = save_dir / f"epoch_{epoch:02d}.pt"
    state_dict = get_state_dict(model)
    torch.save({
        "model_state_dict": state_dict,
        "epoch": epoch,
        "train_loss": avg_loss,
        "label2id": label2id,
        "id2label": id2label,
    }, epoch_ckpt_path)
    print(f"→ Saved epoch checkpoint to {epoch_ckpt_path}")

    # (2) dev 평가 + best 모델 저장
    if dev_loader is not None:
        dev_f1_macro, dev_f1_fil = evaluate(
            model,
            dev_loader,
            desc="dev",
            epoch=epoch,
            log_dir=DEV_REPORT_DIR,
        )

        if dev_f1_fil > best_dev_f1:
            best_dev_f1 = dev_f1_fil
            save_path = save_dir / "best_model.pt"
            state_dict = get_state_dict(model)
            torch.save({
                "model_state_dict": state_dict,
                "epoch": epoch,
                "dev_f1_fil": dev_f1_fil,
                "label2id": label2id,
                "id2label": id2label,
            }, save_path)
            print(f"→ Best model updated. Saved to {save_path}")
    else:
        _ = evaluate(model, train_loader, desc="train_eval")

##############################################
# 11. 저장된 best 모델로 eval 세트 평가 (선택)
##############################################

best_ckpt = save_dir / "best_model.pt"
if best_ckpt.exists() and eval_loader is not None:
    ckpt = torch.load(best_ckpt, map_location=device)
    load_state_dict(model, ckpt["model_state_dict"])
    print(f"\nLoaded best model from epoch {ckpt['epoch']}, dev_f1_fil={ckpt['dev_f1_fil']:.4f}")
    _ = evaluate(model, eval_loader, desc="eval")
else:
    print("\nNo eval set or best checkpoint not found.")
