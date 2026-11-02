# -*- coding: utf-8 -*-
"""
EGD_CXR_Gaze_Calibration_IWAIT2027_CONVERSATION_RESTORED_ARCHIVAL.py

대화 기반 연구 코드 복구본 + 당시 결과 보존본
=============================================

PROVENANCE / 주의
-----------------
- 이 파일은 당시 서버/Colab 작업 디렉터리에서 직접 회수한 단일 원본 스크립트가 아니다.
- 2026-08-14 사용자가 업로드한 `EGD_CXR_Gaze_Celibration.ipynb` 부분은
  실제 업로드 파일에서 직접 변환한 원본 기반 구간이다.
- 그 이후 구간은 이 대화에서 확인 가능한 코드 제공 기록, 수정 기록,
  실행 출력, 연구 진행 기록을 서로 대조해 복구한 conversation-based reconstruction이다.
- 따라서 후반부는 "서버 원본" 또는 "byte-for-byte 최종 원본"으로 간주하면 안 된다.
- 당시 checkpoint/data 환경에서 이 통합 파일 전체를 처음부터 다시 실행하여
  동일 결과가 재현되는지는 검증하지 않았다.

복구 원칙
---------
- 확인 가능한 코드 로직은 임의로 바꾸지 않는다.
- 확인되지 않은 실행 셀/호출은 재현 편의를 위해 새로 만들어 넣지 않는다.
- Shuffled-Gaze는 초기 derangement 버전 -> patient-level 수정 버전 이력을 함께 보존한다.
- 본 복구본의 selective-prediction threshold 계산 코드는
  `np.quantile(confidence, 1.0 - target_coverage)` 구현을 보존한다.
- 별도 대화 기록에는 `ceil(n × coverage)` → confidence 내림차순 정렬 →
  해당 순위값 선택 방식의 threshold 구현도 존재한다.
- 당시 최종 실행에 실제 사용된 threshold 방식이 어느 쪽인지는 현재 자료만으로 확정하지 않는다.
- 2026-08-19 확인된 수정인 `target_coverage >= 1.0 -> -np.inf`는 유지한다.
- Lesion mask union 생성 함수는 복구되어 있으나, 이 파일에 보존된 기록만으로는
  실제 `save_ext_ils_union_masks(...)` 호출 셀을 확인할 수 없어 임의 호출을 추가하지 않는다.
- 코드 계산 결과와 별도로, 당시 실행 출력/최종 보고에서 확인된 수치는
  파일 마지막의 ARCHIVAL REPORTED RESULTS에 고정값으로 보존한다.

연구 시점 관련 메모
-------------------
- 사용자 설명에 따르면 실험 코드는 IWAIT 2027 초록 제출 시점까지 완료되었고
  이후 full paper에서도 같은 실험 코드/결과를 사용했다.
- 이 시점 이력은 사용자 설명 및 대화 기록에 근거한 것이며,
  별도의 서버 Git history로 독립 검증한 사실은 아니다.

Git 권장 설명:
"Conversation-based reconstruction from the original uploaded notebook,
chat code/history, and recovered outputs. Not a recovered server-original
script; end-to-end reproduction against the original runtime is unverified."
"""


# -*- coding: utf-8 -*-
"""
SOURCE-PROVENANCE: EXACT EXPORT — PART A ONLY

This provenance note applies only to PART A below.
PART A was generated directly from the user-uploaded
`EGD_CXR_Gaze_Celibration.ipynb` (uploaded 2026-08-14).

- Cell order is preserved.
- Markdown is preserved as comments.
- Code-cell source is preserved.
- The only mechanical change is that Colab `!shell` cells are wrapped with
  `get_ipython().system(...)` so this file is valid Python syntax.

Later reconstructed experiment code is included after PART A and is not covered
by this EXACT EXPORT statement.
"""


# %% [markdown] ORIGINAL CELL 0
# ⭐⭐런타임 재연결 시 실행 코드⭐⭐
# Google Drive 연결 + 경로 정의 + 저장된 split 불러오기 코드
#
# EGD-CXR처럼 다시 다운받기 번거로운 데이터는 Drive쪽에 저장. (런타임 끊겨도 유지)
#
# /content/drive/MyDrive ➡ 내 드라이브
#
# /content/drive/MyDrive/X-ray ➡ 연구 프로젝트 폴더

# %% ORIGINAL CELL 1
from google.colab import drive
drive.mount("/content/drive")  # 드라이브 마운트 경로

# %% [markdown] ORIGINAL CELL 2
# ⭐⭐기본 import + 경로 + cohort 불러오기⭐⭐

# %% ORIGINAL CELL 3
from pathlib import Path
import pandas as pd

EGD_DATA_ROOT = Path(
    "/content/drive/MyDrive/X-ray/data/egd-cxr/"
    "physionet.org/files/egd-cxr/1.0.0"
)

SPLIT_ROOT = Path(
    "/content/drive/MyDrive/X-ray/gaze_calibration_research/splits"
)

cohort = pd.read_csv(
    SPLIT_ROOT / "egd_cxr_cohort_with_split_seed42.csv"
)

patient_split = pd.read_csv(
    SPLIT_ROOT / "patient_split_seed42_70_15_15.csv"
)

print("EGD 데이터 존재: ", EGD_DATA_ROOT.exists())
print("cohort: ", cohort.shape)
print("patient split: ", patient_split.shape)

# %% [markdown] ORIGINAL CELL 4
# ⭐⭐실제 JPG 경로 다시 붙이기⭐⭐

# %% ORIGINAL CELL 5
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from PIL import Image

# 프로젝트 루트
PROJECT_ROOT = Path("/content/drive/MyDrive/X-ray")

# 다운로드한 MIMIC-CXR JPG 폴더
IMAGE_ROOT = (
    PROJECT_ROOT
    / "data/mimic-cxr-jpg-egdcxr/images"
)

# heatmap 정보까지 저장된 cohort CSV
COHORT_PATH = (
    PROJECT_ROOT
    / "gaze_calibration_research"
    / "splits"
    / "egd_cxr_cohort_with_split_and_heatmap_seed42.csv"
)

# cohort 다시 불러오기
cohort_with_heatmap = pd.read_csv(COHORT_PATH)

print("PROJECT_ROOT 존재:", PROJECT_ROOT.exists())
print("IMAGE_ROOT 존재:", IMAGE_ROOT.exists())
print("COHORT_PATH 존재:", COHORT_PATH.exists())
print("cohort shape:", cohort_with_heatmap.shape)

# %% ORIGINAL CELL 6
from pathlib import Path

BASE_IMAGE_ROOT = Path(
    "/content/drive/MyDrive/X-ray/data/mimic-cxr-jpg-egdcxr/images/physionet.org"
)

cohort_with_heatmap["image_path"] = (
    cohort_with_heatmap["path"]
    .astype(str)
    .apply(lambda p: str(BASE_IMAGE_ROOT / p.replace(".dcm", ".jpg")))
)

print(
    "이미지 경로 누락:",
    cohort_with_heatmap["image_path"].isna().sum()
)

print(
    "실제 존재하지 않는 이미지:",
    (~cohort_with_heatmap["image_path"].apply(lambda p: Path(p).exists())).sum()
)

print(
    "heatmap 경로 누락:",
    cohort_with_heatmap["heatmap_path"].isna().sum()
)

# %% [markdown] ORIGINAL CELL 7
# ⭐⭐label 다시 만들기⭐⭐

# %% ORIGINAL CELL 8
label_map = {
    "Normal": 0,
    "CHF": 1,
    "Pneumonia": 2
}

cohort_with_heatmap["label"] = (
    cohort_with_heatmap["target"].map(label_map)
)

print(
    cohort_with_heatmap[["target", "label"]]
    .drop_duplicates()
    .sort_values("label")
)

# %% [markdown] ORIGINAL CELL 9
# ⭐⭐Dataset 클래스 + transform 다시 정의⭐⭐

# %% ORIGINAL CELL 10
import torchvision.transforms as T
from torch.utils.data import Dataset

image_transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class EGDCXRDataset(Dataset):
    def __init__(self, df, image_transform=None):
        self.df = df.reset_index(drop=True)
        self.image_transform = image_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(row["image_path"]).convert("RGB")

        if self.image_transform is not None:
            img = self.image_transform(img)

        heatmap = np.load(
            row["heatmap_path"]
        ).astype(np.float32)

        heatmap = torch.from_numpy(
            heatmap
        ).unsqueeze(0)

        label = torch.tensor(
            int(row["label"]),
            dtype=torch.long
        )

        return {
            "image": img,
            "heatmap": heatmap,
            "label": label,
            "dicom_id": row["dicom_id"]
        }

# %% [markdown] ORIGINAL CELL 11
# ⭐⭐train / val / test Dataset + DataLoader 다시 만들기⭐⭐

# %% ORIGINAL CELL 12
from torch.utils.data import DataLoader

train_df = cohort_with_heatmap[
    cohort_with_heatmap["split"] == "train"
].copy()

val_df = cohort_with_heatmap[
    cohort_with_heatmap["split"] == "val"
].copy()

test_df = cohort_with_heatmap[
    cohort_with_heatmap["split"] == "test"
].copy()

train_dataset = EGDCXRDataset(
    train_df,
    image_transform=image_transform
)

val_dataset = EGDCXRDataset(
    val_df,
    image_transform=image_transform
)

test_dataset = EGDCXRDataset(
    test_df,
    image_transform=image_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print(
    len(train_dataset),
    len(val_dataset),
    len(test_dataset)
)

# %% [markdown] ORIGINAL CELL 13
# ⭐⭐DenseNet-121 Baseline 모델 다시 정의⭐⭐

# %% ORIGINAL CELL 14
import torch

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("device:", device)

# %% ORIGINAL CELL 15
import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

model = densenet121(
    weights=DenseNet121_Weights.IMAGENET1K_V1
)

# 3-class classification으로 변경
in_features = model.classifier.in_features
model.classifier = nn.Linear(in_features, 3)

model = model.to(device)

print(model.classifier)

# %% [markdown] ORIGINAL CELL 16
# ⭐⭐Loss와 Optimizer 정의⭐⭐

# %% ORIGINAL CELL 17
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

print("criterion / optimizer 정의 완료")

# %% [markdown] ORIGINAL CELL 18
# ⭐⭐GPU에서 한 배치 forward 테스트⭐⭐

# %% ORIGINAL CELL 19
batch = next(iter(train_loader))

images = batch["image"].to(device)
labels = batch["label"].to(device)

with torch.no_grad():
    logits = model(images)

loss = criterion(logits, labels)

print("image device:", images.device)
print("logits shape:", logits.shape)
print("labels shape:", labels.shape)
print("loss:", loss.item())

# %% [markdown] ORIGINAL CELL 20
# ⭐⭐checkpoint/history 저장 경로 정의⭐⭐

# %% ORIGINAL CELL 21
from pathlib import Path

CHECKPOINT_DIR = Path(
    "/content/drive/MyDrive/X-ray/gaze_calibration_research/checkpoints/baseline_densenet121"
)

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pt"
LAST_CHECKPOINT_PATH = CHECKPOINT_DIR / "last_checkpoint.pt"
HISTORY_PATH = CHECKPOINT_DIR / "history.pt"

print(CHECKPOINT_DIR)

# %% [markdown] ORIGINAL CELL 22
# ⭐⭐Baseline 학습 함수 다시 정의⭐⭐

# %% ORIGINAL CELL 23
from tqdm.auto import tqdm
import torch

def train_baseline(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=10,
    start_epoch=1,
    best_val_loss=float("inf"),
    history=None
):
    # 처음 학습할 때
    if history is None:
        history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

    for epoch in range(start_epoch, epochs + 1):

        print(f"\n===== Epoch {epoch}/{epochs} =====")

        # =====================
        # Train
        # =====================
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(
            train_loader,
            desc=f"Train {epoch}/{epochs}"
        )

        for batch in train_bar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            batch_size = images.size(0)

            train_loss += loss.item() * batch_size

            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += batch_size

            current_loss = train_loss / train_total
            current_acc = train_correct / train_total

            train_bar.set_postfix(
                loss=f"{current_loss:.4f}",
                acc=f"{current_acc:.4f}"
            )

        train_loss /= train_total
        train_acc = train_correct / train_total

        # =====================
        # Validation
        # =====================
        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_bar = tqdm(
            val_loader,
            desc=f"Val {epoch}/{epochs}"
        )

        with torch.no_grad():
            for batch in val_bar:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                logits = model(images)
                loss = criterion(logits, labels)

                batch_size = images.size(0)

                val_loss += loss.item() * batch_size

                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += batch_size

                current_val_loss = val_loss / val_total
                current_val_acc = val_correct / val_total

                val_bar.set_postfix(
                    loss=f"{current_val_loss:.4f}",
                    acc=f"{current_val_acc:.4f}"
                )

        val_loss /= val_total
        val_acc = val_correct / val_total

        # =====================
        # History 저장
        # =====================
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # history 자체도 Drive 저장
        torch.save(history, HISTORY_PATH)

        # =====================
        # Epoch 결과 출력
        # =====================
        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"val acc {val_acc:.4f}"
        )

        # =====================
        # Best model 저장
        # =====================
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "history": history
                },
                BEST_MODEL_PATH
            )

            print("→ Best model 저장")

        # =====================
        # 마지막 완료 epoch 저장
        # =====================
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "history": history
            },
            LAST_CHECKPOINT_PATH
        )

        print("→ Last checkpoint 저장")
        print("→ History 저장")

    print("\n학습 완료")
    print("Best val loss:", best_val_loss)

    return model, history

# %% [markdown] ORIGINAL CELL 24
#
#
# ---
#

# %% [markdown] ORIGINAL CELL 25
# 프로젝트 폴더 생성
#
# ```
# X-ray/
# ├── EGD_CXR_Gaze_Calibration.ipynb
# └── data/
#     └── egd-cxr/
# ```
#

# %% ORIGINAL CELL 26
from pathlib import Path

PROJECT_ROOT = Path("/content/drive/MyDrive/X-ray") # 전체 연구 프로젝트 위치
DATA_ROOT = PROJECT_ROOT / "data"   # 데이터 전용 폴더
EGD_ROOT = DATA_ROOT / "egd-cxr"    # EGD-CXR 전용 폴더

EGD_ROOT.mkdir(parents=True, exist_ok=True)   # 실제 폴더 생성

print("프로젝트 폴더: ", PROJECT_ROOT)
print("데이터 폴더: ", DATA_ROOT)
print("EGD-CXR 폴더: ", EGD_ROOT)
print("EGD-CXR 폴더 존재: ", EGD_ROOT.exists())

# %% [markdown] ORIGINAL CELL 27
# PhysioNet에서 EGD-CXR 다운로드
#
#
# ```
# egd-cxr/
# └── physionet.org/
#     └── files/
#         └── egd-cxr/
#             └── 1.0.0/
# ```
#

# %% ORIGINAL CELL 28
import os

os.chdir(EGD_ROOT)
print("현재 작업 폴더: ", os.getcwd())

# %% ORIGINAL CELL 29
get_ipython().system('wget -r -N -c -np --user mijin --ask-password https://physionet.org/files/egd-cxr/1.0.0/')

# %% [markdown] ORIGINAL CELL 30
# 실제 대이터 폴더 확인
#
#
#
# ```
# 최상위 파일/폴더:
# - LICENSE.txt
# - SHA256SUMS.txt
# - audio_segmentation_transcripts
# - bounding_boxes.csv
# - eye_gaze.csv
# - fixations.csv
# - inclusion_exclusion_criteria_outputs
# - index.html
# - master_sheet.csv
# - table_descriptions.pdf
# ```
#

# %% ORIGINAL CELL 31
from pathlib import Path

EGD_DATA_ROOT = EGD_ROOT / "physionet.org" / "files" / "egd-cxr" / "1.0.0"

print("EGD-CXR 실제 데이터 폴더:")
print(EGD_DATA_ROOT)

print("\n폴더 존재 여부:")
print(EGD_DATA_ROOT.exists())

print("\n최상위 파일/폴더:")
for item in sorted(EGD_DATA_ROOT.iterdir()):
    print("-", item.name)

# %% [markdown] ORIGINAL CELL 32
# 다운로드 무결성 검사
#
#
#
# ```
# ======================================================================
# SHA256 검증 완료
# ======================================================================
# 전체 파일: 7,593
# 정상 파일: 7,590
# 손상/불일치: 0
# 누락 파일: 3
# 총 소요시간: 23.0분
#
# ❌ 누락 파일:
# - audio_segmentation_transcripts/0a1d6c66-b214a6e8-09501125-827f5b5a-3fa1bcc5/.DS_Store
# - audio_segmentation_transcripts/0a90c3bb-3cf21ec9-835a1537-faf370a1-21053fd6/.DS_Store
# - audio_segmentation_transcripts/feefc68a-0a26d0a0-65e4a2a0-b1efc19d-0abcd7d1/.DS_Store
# ➡ 연구에는 필요없는 파일이라 다운로드 검증 완료로 처리.
# ```
#

# %% ORIGINAL CELL 33
from pathlib import Path
import hashlib
import time

# SHA256 목록 파일
checksum_file = EGD_DATA_ROOT / "SHA256SUMS.txt"

# 1. SHA256SUMS.txt 읽기
entries = []

with open(checksum_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        expected_hash, relative_path = line.split(maxsplit=1)

        # sha256sum 형식에서 파일명 앞의 *가 있을 경우 제거
        relative_path = relative_path.lstrip("*")

        entries.append((expected_hash, relative_path))

total = len(entries)

print(f"검사할 파일 수: {total:,}개")
print("SHA256 검증 시작...\n")

start_time = time.time()

ok_count = 0
failed = []
missing = []

# 2. 파일 하나씩 검사
for i, (expected_hash, relative_path) in enumerate(entries, start=1):

    file_path = EGD_DATA_ROOT / relative_path

    # 파일이 아예 없는 경우
    if not file_path.exists():
        missing.append(relative_path)

    else:
        sha256 = hashlib.sha256()

        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)   # 1 MB씩 읽기

                if not chunk:
                    break

                sha256.update(chunk)

        actual_hash = sha256.hexdigest()

        if actual_hash == expected_hash:
            ok_count += 1
        else:
            failed.append(relative_path)

    # 10개마다 진행 상황 출력
    if i % 10 == 0 or i == total:

        elapsed = time.time() - start_time
        speed = i / elapsed

        remaining = total - i

        if speed > 0:
            eta = remaining / speed
        else:
            eta = 0

        print(
            f"[{i:,}/{total:,}] "
            f"{i/total*100:5.1f}% | "
            f"OK: {ok_count:,} | "
            f"경과: {elapsed/60:.1f}분 | "
            f"예상 남은 시간: {eta/60:.1f}분 | "
            f"현재: {relative_path}"
        )

# 3. 최종 결과
elapsed = time.time() - start_time

print("\n" + "=" * 70)
print("SHA256 검증 완료")
print("=" * 70)

print(f"전체 파일: {total:,}")
print(f"정상 파일: {ok_count:,}")
print(f"손상/불일치: {len(failed):,}")
print(f"누락 파일: {len(missing):,}")
print(f"총 소요시간: {elapsed/60:.1f}분")

if len(failed) == 0 and len(missing) == 0:
    print("\n✅ 모든 파일이 정상입니다.")
else:
    if failed:
        print("\n❌ SHA256 불일치 파일:")
        for p in failed:
            print("-", p)

    if missing:
        print("\n❌ 누락 파일:")
        for p in missing:
            print("-", p)

# %% [markdown] ORIGINAL CELL 34
# EGD-CXR 데이터 확인

# %% [markdown] ORIGINAL CELL 35
# 실제 파일들의 크기, 행 수. 컬럼명 확인
#
# ➡ 공통 연결 키: DICOM ID
#
# ➡ eye_gaze.csv: 프레임/샘플 단위의 더 촘촘한 원시 gaze 기록.
# ➡ fixations.csv: eye_gaze.csv 중 fixation 단위로 정리된 데이터.
#   > FPOGX, FPOGY, FPOGD, X_ORIGINAL, Y_ORIGINAL 같은 fixation 위치와 지속시간 관련 값. (같은 DICOM에서 행 수가 훨씬 적을 것.)
#
# 따라서, 첫 실험에서는 fixations.csv부터 쓰는 것이 적절.
#
#
#
# ---
#
# **bounding_boxes.csv**: cardiac silhouette, left clavicle, left lower lung zone 같은 **해부학적 영역 박스**가 포함되어 있음.
#

# %% ORIGINAL CELL 36
#모든 파일X, 처음 몇 행만 읽어서 구조만 파악

import pandas as pd

files = {
    "master_sheet": EGD_DATA_ROOT / "master_sheet.csv",
    "fixations": EGD_DATA_ROOT / "fixations.csv",
    "eye_gaze": EGD_DATA_ROOT / "eye_gaze.csv",
    "bounding_boxes": EGD_DATA_ROOT / "bounding_boxes.csv",
}

for name, path in files.items():
    df_sample = pd.read_csv(path, nrows=5)

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    print("파일 크기:", round(path.stat().st_size / (1024**2), 2), "MB")
    print("컬럼 수:", len(df_sample.columns))

    print("\n컬럼명:")
    print(df_sample.columns.tolist())

    print("\n처음 5행:")
    display(df_sample)

# %% [markdown] ORIGINAL CELL 37
# table_descriptions.pdf 내용 확인.
# 🐾크롬에서 PDF 텍스트 바로 뽑기.

# %% ORIGINAL CELL 38
get_ipython().system('apt-get update -qq\napt-get install -y poppler-utils')

# %% ORIGINAL CELL 39
import subprocess

pdf_path = EGD_DATA_ROOT / "table_descriptions.pdf"
txt_path = EGD_DATA_ROOT / "table_descriptions.txt"

subprocess.run([
    "pdftotext",
    str(pdf_path),
    str(txt_path)
])

with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

print(text[:12000])

# %% [markdown] ORIGINAL CELL 40
# 연구 타깃 결정
#
# > pneumonia: ED 퇴원 ICD-9 진단코드로 판별한 임상적 폐렴 진단.
#
# > pneumonia_chx: MIMIC-CXR 리포트에 CheXpert labeler를 적용해 얻은 영상 리포트 기반 라벨.
#
# Normal / CHF / Pneumonia 3-class가 메인 실험으로 실제 가능한지 검증하는 cohort feasibility 단계

# %% ORIGINAL CELL 41
import pandas as pd

# ============================================================
# 1. 데이터 불러오기
# ============================================================

master = pd.read_csv(EGD_DATA_ROOT / "master_sheet.csv")
fix = pd.read_csv(EGD_DATA_ROOT / "fixations.csv")

class_cols = ["Normal", "CHF", "pneumonia"]

print("=" * 70)
print("1. 기본 데이터 규모")
print("=" * 70)

print("전체 이미지 행 수:", len(master))
print("고유 DICOM 수:", master["dicom_id"].nunique())
print("고유 study 수:", master["study_id"].nunique())
print("고유 patient 수:", master["patient_id"].nunique())


# ============================================================
# 2. 세 임상 클래스의 양성 수 확인
# ============================================================

print("\n" + "=" * 70)
print("2. Normal / CHF / Pneumonia 클래스 분포")
print("=" * 70)

for col in class_cols:
    positive = (master[col] == 1).sum()
    print(f"{col:10s}: {positive:,}개")


# ============================================================
# 3. 클래스 중복 확인
# ============================================================

# 1인 경우에만 해당 클래스 양성으로 처리
class_binary = pd.DataFrame({
    col: (master[col] == 1).astype(int)
    for col in class_cols
})

print("\n" + "=" * 70)
print("3. 클래스 조합")
print("=" * 70)

combination = (
    class_binary
    .value_counts()
    .rename("count")
    .reset_index()
    .sort_values("count", ascending=False)
)

print(combination.to_string(index=False))


# 한 영상에 몇 개 클래스가 동시에 양성인지
class_binary["positive_count"] = class_binary[class_cols].sum(axis=1)

print("\n한 영상의 양성 클래스 개수:")
print(
    class_binary["positive_count"]
    .value_counts()
    .sort_index()
)

print(
    "\n2개 이상 클래스가 동시에 양성인 이미지:",
    (class_binary["positive_count"] >= 2).sum()
)


# ============================================================
# 4. Fixation 데이터가 실제로 연결되는지 확인
# ============================================================

print("\n" + "=" * 70)
print("4. Gaze / Fixation 연결 가능성")
print("=" * 70)

master_ids = set(master["dicom_id"].astype(str))
fix_ids = set(fix["DICOM_ID"].dropna().astype(str))

print("master_sheet DICOM 수:", len(master_ids))
print("fixation이 있는 DICOM 수:", len(fix_ids))
print("둘 다 존재하는 DICOM 수:", len(master_ids & fix_ids))


# ============================================================
# 5. 클래스별 gaze 사용 가능 이미지 수
# ============================================================

tmp = master.copy()

tmp["dicom_id"] = tmp["dicom_id"].astype(str)
tmp["has_fixation"] = tmp["dicom_id"].isin(fix_ids)

print("\n" + "=" * 70)
print("5. 클래스별 fixation 보유 이미지")
print("=" * 70)

for col in class_cols:

    total_positive = (tmp[col] == 1).sum()

    gaze_positive = (
        (tmp[col] == 1) &
        (tmp["has_fixation"])
    ).sum()

    print(
        f"{col:10s}: "
        f"전체 {total_positive:,}개 / "
        f"fixation 있음 {gaze_positive:,}개"
    )


# ============================================================
# 6. '순수한' 3-class 후보 cohort 확인
# ============================================================

# Normal / CHF / Pneumonia 중 정확히 하나만 1인 경우
tmp["class_positive_count"] = (
    tmp[class_cols]
    .eq(1)
    .sum(axis=1)
)

pure = tmp[
    (tmp["class_positive_count"] == 1) &
    (tmp["has_fixation"])
].copy()

def assign_class(row):
    if row["Normal"] == 1:
        return "Normal"
    elif row["CHF"] == 1:
        return "CHF"
    elif row["pneumonia"] == 1:
        return "Pneumonia"
    return None

pure["target"] = pure.apply(assign_class, axis=1)

print("\n" + "=" * 70)
print("6. 겹치지 않는 3-class 후보 cohort")
print("=" * 70)

print(pure["target"].value_counts())

print("\n3-class 후보 전체 이미지:", len(pure))
print("3-class 후보 고유 환자:", pure["patient_id"].nunique())


# ============================================================
# 7. 각 클래스의 환자 수
# ============================================================

print("\n" + "=" * 70)
print("7. 클래스별 고유 환자 수")
print("=" * 70)

patient_counts = (
    pure.groupby("target")["patient_id"]
    .nunique()
    .sort_index()
)

print(patient_counts)

# %% [markdown] ORIGINAL CELL 42
# 환자 단위 label consistency 검사
#
#
#
# ---
#
# 같은 환자가 서로 다른 클래스에 걸쳐있지 않음.
#
# ➡ patient_id를 기준으로 Normal / CHF / Pneumonia를 stratification해서 나누면, 환자 leakage(테스트 데이터가 훈련 과정에서 섞여 들어가는 현상)를 막으면서 클래스 비율도 유지 가능.

# %% ORIGINAL CELL 43
# ============================================================
# 다음 단계:
# 같은 환자가 여러 class에 등장하는지 확인
# ============================================================

patient_target = (
    pure.groupby("patient_id")["target"]
        .agg(
            image_count="count",
            target_count="nunique",
            targets=lambda x: sorted(x.unique())
        )
        .reset_index()
)

print("=" * 70)
print("1. 환자당 이미지 수")
print("=" * 70)

print(
    patient_target["image_count"]
    .value_counts()
    .sort_index()
)

print("\n여러 이미지를 가진 환자 수:",
      (patient_target["image_count"] > 1).sum())


print("\n" + "=" * 70)
print("2. 같은 환자가 여러 class에 등장하는지")
print("=" * 70)

multi_target = patient_target[
    patient_target["target_count"] > 1
].copy()

print("여러 class에 속하는 환자 수:", len(multi_target))

if len(multi_target) > 0:
    display(multi_target)
else:
    print("✅ 모든 환자의 class가 일관적입니다.")


print("\n" + "=" * 70)
print("3. 여러 이미지를 가진 환자 예시")
print("=" * 70)

display(
    patient_target[
        patient_target["image_count"] > 1
    ].head(20)
)

# %% [markdown] ORIGINAL CELL 44
# 지금까지 확인한 데이터 기준 메인 실험
#
#
#
# ```
#                        EGD-CXR
#                           │
#                 1,083 CXR + Fixation
#                           │
#             ┌─────────────┼─────────────┐
#             │             │             │
#          Normal          CHF        Pneumonia
#           360            363           360
#             │             │             │
#             └─────────────┼─────────────┘
#                           │
#                   Patient-level split
#                           │
#           ┌───────────────┼────────────────┐
#           │               │                │
#        Baseline      Original Gaze     Controls
#                                       ├─ Shuffled
#                                       ├─ Random
#                                       └─ ROI/Anatomy
#                           │
#              ┌────────────┴────────────┐
#              │                         │
#        Discrimination              Reliability
#        AUROC / F1 etc.        ECE / Brier / NLL
#                                      │
#                            High-confidence errors
#                                      │
#                             Risk-Coverage / AURC
# ```
#

# %% [markdown] ORIGINAL CELL 45
# 연구에서 모든 실험군이 공통으로 사용할 고정 train/validation/test split을 생성
#
# 단순 분류 연구X, calibration 연구 진행 예정. 따라서 validation set이 매우 중요
#
# (Calibration 연구: 인공신경망인 센서 등의 출력값과 신뢰도를 실제 값이나 확률에 정확히 일치시키는 기술.)
#
#
# *   Train: 모델 가중치 학습
# *   Validation: early stopping, hyperparameter 선택, temperature scaling fitting
# *   Test: 마지막 성능 평가만.
#
# ⭐ temperautre scaling에 test set을 사용하면 leakage가 생기기 때문에 반드시 calidation을 별도로 보존해야한다.
#
#
#

# %% [markdown] ORIGINAL CELL 46
# 1차 메인 split: **70/15/15**

# %% [markdown] ORIGINAL CELL 47
# patient-level split 생성 + Drive 저장
#
# > 런타임이 끊겨도 Drive 마운트 + EGD_DATA_ROOT 설정 후 이 셀만 다시 돌려도 됨.

# %% ORIGINAL CELL 48
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ============================================================
# 1. master_sheet 불러오기
# ============================================================

master = pd.read_csv(EGD_DATA_ROOT / "master_sheet.csv")

class_cols = ["Normal", "CHF", "pneumonia"]


# ============================================================
# 2. 3-class target 생성
# ============================================================

def assign_target(row):
    if row["Normal"] == 1:
        return "Normal"
    elif row["CHF"] == 1:
        return "CHF"
    elif row["pneumonia"] == 1:
        return "Pneumonia"
    return None


master["target"] = master.apply(assign_target, axis=1)

# 세 클래스 중 정확히 하나만 양성인지 재검증
master["positive_count"] = master[class_cols].eq(1).sum(axis=1)

cohort = master[
    (master["positive_count"] == 1) &
    (master["target"].notna())
].copy()

print("전체 cohort:", len(cohort))
print(cohort["target"].value_counts())


# ============================================================
# 3. 환자별 대표 target 생성
# ============================================================

patient_df = (
    cohort.groupby("patient_id")
          .agg(
              target=("target", "first"),
              target_count=("target", "nunique"),
              image_count=("dicom_id", "count")
          )
          .reset_index()
)

# 혹시라도 한 환자가 여러 target이면 즉시 중단
assert patient_df["target_count"].max() == 1, \
    "한 환자가 여러 target에 속합니다."

print("\n고유 환자 수:", len(patient_df))
print("\n환자 기준 클래스 분포:")
print(patient_df["target"].value_counts())


# ============================================================
# 4. Patient-level 70 / 15 / 15 split
# ============================================================

RANDOM_SEED = 42

train_patients, temp_patients = train_test_split(
    patient_df,
    test_size=0.30,
    random_state=RANDOM_SEED,
    stratify=patient_df["target"]
)

val_patients, test_patients = train_test_split(
    temp_patients,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=temp_patients["target"]
)


# split 이름 부여
train_patients = train_patients.copy()
val_patients = val_patients.copy()
test_patients = test_patients.copy()

train_patients["split"] = "train"
val_patients["split"] = "val"
test_patients["split"] = "test"

patient_split = pd.concat(
    [train_patients, val_patients, test_patients],
    ignore_index=True
)


# ============================================================
# 5. 이미지 단위 cohort에 split 연결
# ============================================================

cohort_split = cohort.merge(
    patient_split[["patient_id", "split"]],
    on="patient_id",
    how="left"
)

assert cohort_split["split"].isna().sum() == 0


# ============================================================
# 6. Leakage 검사
# ============================================================

train_ids = set(
    cohort_split.loc[
        cohort_split["split"] == "train",
        "patient_id"
    ]
)

val_ids = set(
    cohort_split.loc[
        cohort_split["split"] == "val",
        "patient_id"
    ]
)

test_ids = set(
    cohort_split.loc[
        cohort_split["split"] == "test",
        "patient_id"
    ]
)

assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)

print("\n✅ Patient leakage 없음")


# ============================================================
# 7. Split 결과 확인
# ============================================================

print("\n" + "=" * 70)
print("환자 기준 split")
print("=" * 70)

print(
    pd.crosstab(
        patient_split["split"],
        patient_split["target"],
        margins=True
    )
)


print("\n" + "=" * 70)
print("이미지 기준 split")
print("=" * 70)

print(
    pd.crosstab(
        cohort_split["split"],
        cohort_split["target"],
        margins=True
    )
)


# ============================================================
# 8. Drive에 영구 저장
# ============================================================

SAVE_ROOT = Path(
    "/content/drive/MyDrive/X-ray/"
    "gaze_calibration_research/splits"
)

SAVE_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

patient_split_path = (
    SAVE_ROOT /
    "patient_split_seed42_70_15_15.csv"
)

cohort_split_path = (
    SAVE_ROOT /
    "egd_cxr_cohort_with_split_seed42.csv"
)

patient_split.to_csv(
    patient_split_path,
    index=False
)

cohort_split.to_csv(
    cohort_split_path,
    index=False
)

print("\n✅ 저장 완료")
print("Patient split:")
print(patient_split_path)

print("\nImage cohort split:")
print(cohort_split_path)

# %% [markdown] ORIGINAL CELL 49
# EGD-CXE-JPG 다운로드

# %% ORIGINAL CELL 50
import os
import subprocess
from pathlib import Path

# ============================================================
# 1. 저장 폴더
# ============================================================

DOWNLOAD_ROOT = Path(
    "/content/drive/MyDrive/X-ray/data/mimic-cxr-jpg-egdcxr/images"
)

DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. 다운로드 목록 파일 생성
# ============================================================

url_list_path = DOWNLOAD_ROOT.parent / "egd_cxr_1083_urls.txt"

base_url = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/"

urls = [
    base_url + p
    for p in cohort["jpg_relative_path"]
]

with open(url_list_path, "w") as f:
    for url in urls:
        f.write(url + "\n")

print("URL 수:", len(urls))
print("목록 파일:", url_list_path)

# %% ORIGINAL CELL 51
get_ipython().system('wget \\\n  -c \\\n  -N \\\n  -x \\\n  --cut-dirs=3 \\\n  --user mijin \\\n  --ask-password \\\n  --progress=bar:force:noscroll \\\n  -i "/content/drive/MyDrive/X-ray/data/mimic-cxr-jpg-egdcxr/egd_cxr_1083_urls.txt" \\\n  -P "/content/drive/MyDrive/X-ray/data/mimic-cxr-jpg-egdcxr/images"')

# %% ORIGINAL CELL 52
from pathlib import Path
import pandas as pd

DOWNLOAD_ROOT = Path(
    "/content/drive/MyDrive/X-ray/data/mimic-cxr-jpg-egdcxr/images"
)

SPLIT_ROOT = Path(
    "/content/drive/MyDrive/X-ray/gaze_calibration_research/splits"
)

cohort = pd.read_csv(
    SPLIT_ROOT / "egd_cxr_cohort_with_split_seed42.csv"
)

# 실제 다운로드된 JPG 찾기
jpg_files = list(DOWNLOAD_ROOT.rglob("*.jpg"))

# 실제 JPG 파일명의 dicom_id
downloaded_ids = {p.stem for p in jpg_files}

# 연구에 필요한 dicom_id
expected_ids = set(cohort["dicom_id"].astype(str))

missing_ids = expected_ids - downloaded_ids
extra_ids = downloaded_ids - expected_ids

print("연구 대상 이미지:", len(expected_ids))
print("다운로드된 JPG:", len(jpg_files))
print("일치한 이미지:", len(expected_ids & downloaded_ids))
print("누락:", len(missing_ids))
print("추가 파일:", len(extra_ids))

if missing_ids:
    print("\n누락된 DICOM ID 예시:")
    print(list(missing_ids)[:10])

# %% ORIGINAL CELL 53
from pathlib import Path
from PIL import Image

IMAGE_ROOT = Path(
    "/content/drive/MyDrive/X-ray/data/mimic-cxr-jpg-egdcxr/images"
)

jpg_files = list(IMAGE_ROOT.rglob("*.jpg"))

corrupt_files = []

for i, p in enumerate(jpg_files, 1):
    try:
        with Image.open(p) as img:
            img.verify()
    except Exception as e:
        corrupt_files.append((str(p), str(e)))

    if i % 100 == 0:
        print(f"{i}/{len(jpg_files)} 검사 완료")

print("\n전체 JPG:", len(jpg_files))
print("손상 파일:", len(corrupt_files))

if corrupt_files:
    print("\n손상 파일 예시:")
    for item in corrupt_files[:10]:
        print(item)

# %% ORIGINAL CELL 54
from pathlib import Path
from PIL import Image

IMAGE_ROOT = Path(
    "/content/drive/MyDrive/X-ray/data/mimic-cxr-jpg-egdcxr/images"
)

jpg_files = list(IMAGE_ROOT.rglob("*.jpg"))

corrupt_files = []

for i, p in enumerate(jpg_files, 1):
    try:
        with Image.open(p) as img:
            img.verify()
    except Exception as e:
        corrupt_files.append((str(p), str(e)))

    if i % 100 == 0:
        print(f"{i}/{len(jpg_files)} 검사 완료")

print("\n전체 JPG:", len(jpg_files))
print("손상 파일:", len(corrupt_files))

# %% [markdown] ORIGINAL CELL 55
# JPG해상도와 원본 DICOM 좌표계가 맞는지 확인.
#
# ➡ fixations.csv의 X_ORIGINAL, Y_ORIGINAL이 원본 DICOM 좌표 기준이라서, 우리가 받은 JPG와 크기가 다르면 시선점이 엉뚱한위치에 찍힐 수 있기 때문에 진행.

# %% [markdown] ORIGINAL CELL 56
# MIMIC-CXR 공식 metadata 파일을 다운.

# %% ORIGINAL CELL 57
import subprocess
from getpass import getpass

PHYSIONET_USERNAME = "rlaalwls1220@hufs.ac.kr"

password = getpass("PhysioNet password: ")

METADATA_URL = (
    "https://physionet.org/files/mimic-cxr-jpg/2.1.0/"
    "mimic-cxr-2.0.0-metadata.csv.gz"
)

result = subprocess.run(
    [
        "wget",
        "-c",
        "--user", PHYSIONET_USERNAME,
        "--password", password,
        "-O", str(METADATA_PATH),
        METADATA_URL
    ]
)

print("종료 코드:", result.returncode)
print("파일 존재:", METADATA_PATH.exists())

if METADATA_PATH.exists():
    print("파일 크기:", METADATA_PATH.stat().st_size, "bytes")

# %% ORIGINAL CELL 58
import pandas as pd

metadata = pd.read_csv(METADATA_PATH)

print("metadata shape:", metadata.shape)
print(metadata.columns.tolist())
print(metadata.head())

# %% [markdown] ORIGINAL CELL 59
# PIL의 img.size 순서 = (width, height)
#
# metadata는 Columns = width, Rows = height
#
# ➡ 비교할땐 img.size == (Columns, Rows)

# %% ORIGINAL CELL 60
from PIL import Image

# dicom_id -> metadata 행
meta_small = metadata[["dicom_id", "Rows", "Columns"]].copy()

# cohort 1083장만 추리기
cohort_meta = cohort[["dicom_id"]].merge(
    meta_small,
    on="dicom_id",
    how="left"
)

# 실제 파일 경로 매핑
path_by_id = {p.stem: p for p in jpg_files}

mismatch = []
missing_meta = []

for i, row in cohort_meta.iterrows():
    dicom_id = str(row["dicom_id"])

    if pd.isna(row["Rows"]) or pd.isna(row["Columns"]):
        missing_meta.append(dicom_id)
        continue

    p = path_by_id[dicom_id]

    with Image.open(p) as img:
        jpg_w, jpg_h = img.size

    expected_w = int(row["Columns"])
    expected_h = int(row["Rows"])

    if (jpg_w, jpg_h) != (expected_w, expected_h):
        mismatch.append({
            "dicom_id": dicom_id,
            "jpg_size": (jpg_w, jpg_h),
            "metadata_size": (expected_w, expected_h)
        })

print("전체 cohort:", len(cohort_meta))
print("metadata 누락:", len(missing_meta))
print("해상도 불일치:", len(mismatch))

if mismatch:
    print("\n불일치 예시:")
    for x in mismatch[:10]:
        print(x)

# %% [markdown] ORIGINAL CELL 61
# 위의 결과:
#
# 전체 cohort: 1083
#
# metadata 누락: 0
#
# 해상도 불일치: 0
#
#
# ➡ MIMIC-CXR JPG가 원본 DICOM과 동일한 픽셀 크기를 유지하고 있다는 것을 cohort에서 직접 확인.

# %% [markdown] ORIGINAL CELL 62
# **좌표계 1차 검증**
#
#
# *   cohort 1,083장 전부 metadata 존재.
# *   JPG 크기와 원본 DICOM의 Columns x Rows가 1,083장 모두 일치.
#
# ➡ 따라서 X_ORIGINAL, Y_ORIGINAL을 JPG 위에 올려볼 근거가 생김.
#

# %% [markdown] ORIGINAL CELL 63
# fixation 좌표 범위가 이미지 크기 안에 들어오는지 확인 (heatmap X)

# %% ORIGINAL CELL 64
import pandas as pd
from pathlib import Path

FIXATION_PATH = EGD_DATA_ROOT / "fixations.csv"

fix = pd.read_csv(FIXATION_PATH)

print("fixations shape:", fix.shape)
print("컬럼 존재 확인:")
print("DICOM_ID" in fix.columns)
print("X_ORIGINAL" in fix.columns)
print("Y_ORIGINAL" in fix.columns)

# %% [markdown] ORIGINAL CELL 65
# cohort 1,083장에해당하는 fixation만 추리기.

# %% ORIGINAL CELL 66
cohort_ids = set(cohort["dicom_id"].astype(str))

fix_cohort = fix[
    fix["DICOM_ID"].astype(str).isin(cohort_ids)
].copy()

print("전체 fixation:", len(fix))
print("cohort fixation:", len(fix_cohort))
print("포함된 DICOM 수:", fix_cohort["DICOM_ID"].nunique())

# %% [markdown] ORIGINAL CELL 67
# 좌표 점위 검사

# %% ORIGINAL CELL 68
fix_check = fix_cohort.merge(
    metadata[["dicom_id", "Rows", "Columns"]],
    left_on="DICOM_ID",
    right_on="dicom_id",
    how="left"
)

valid_xy = (
    fix_check["X_ORIGINAL"].notna() &
    fix_check["Y_ORIGINAL"].notna()
)

out_of_bounds = fix_check[
    valid_xy &
    (
        (fix_check["X_ORIGINAL"] < 0) |
        (fix_check["Y_ORIGINAL"] < 0) |
        (fix_check["X_ORIGINAL"] >= fix_check["Columns"]) |
        (fix_check["Y_ORIGINAL"] >= fix_check["Rows"])
    )
]

print("좌표 있는 fixation:", valid_xy.sum())
print("이미지 범위 밖 fixation:", len(out_of_bounds))

# %% [markdown] ORIGINAL CELL 69
# ➡ 좌표 있는 fixation: 48747, 이미지 범위 밖 fixation: 481
#
# 즉, 약 99.0%의 fixation 좌표는 이미지 내부에 정상적으로 들어가고, 약 1%만 범위를 벗어남.

# %% [markdown] ORIGINAL CELL 70
# 유효 좌표만 남기기

# %% ORIGINAL CELL 71
valid_fix = fix_check[
    valid_xy &
    (fix_check["X_ORIGINAL"] >= 0) &
    (fix_check["Y_ORIGINAL"] >= 0) &
    (fix_check["X_ORIGINAL"] < fix_check["Columns"]) &
    (fix_check["Y_ORIGINAL"] < fix_check["Rows"])
].copy()

print("원래 좌표 있는 fixation:", valid_xy.sum())
print("유효 fixation:", len(valid_fix))
print("제외된 fixation:", valid_xy.sum() - len(valid_fix))
print("유효 DICOM 수:", valid_fix["DICOM_ID"].nunique())

# %% [markdown] ORIGINAL CELL 72
# 시각적 정합성 확인
#
# ➡ 시선점이 실제 흉부 X-ray의 폐, 심장, 종격동 같은 해부학적 영역에 자연스럽게 분포하는지 확인.
#
#
#
#
# *   점들이 이미지 바깥으로 대량으로 튀지 않음.
# *   좌우 반전/상하 반전처럼 보이지 않음.
# *   폐야, 종격동, 심장, 횡경막 주변에 자연스럽게 분포.
# *   일부 점ㅁ이 주변부에도 있지만, radiologist scan pattern으로 충분히 나올 수 있는 수준.
#

# %% ORIGINAL CELL 73
import matplotlib.pyplot as plt
from PIL import Image

# dicom_id -> 실제 JPG 경로
path_by_id = {p.stem: p for p in jpg_files}

# 각 클래스에서 fixation이 존재하는 첫 이미지 1장 선택
classes = ["Normal", "CHF", "pneumonia"]

for cls in classes:
    sample_id = cohort.loc[cohort[cls] == 1, "dicom_id"].iloc[0]

    img_path = path_by_id[sample_id]
    pts = valid_fix[valid_fix["DICOM_ID"] == sample_id]

    with Image.open(img_path) as img:
        plt.figure(figsize=(8, 8))
        plt.imshow(img, cmap="gray")

        plt.scatter(
            pts["X_ORIGINAL"],
            pts["Y_ORIGINAL"],
            s=20,
            alpha=0.6
        )

        plt.title(
            f"{cls} | DICOM: {sample_id}\n"
            f"Fixations: {len(pts)}"
        )
        plt.axis("off")
        plt.show()

# %% [markdown] ORIGINAL CELL 74
# fixation point ➡ heatmap
#
#
#
# ```
# heatmap 원칙
#
# *   입력 이미지 크기: 나중 학습용으로 통일된 크기 사용 → 우선 512 x 512 추천
# *   fixation 좌표: 원본 좌표를 512 x 512로 선형 스케일링
# *   가중치: fixation duration FPOGD 사용
# *   퍼짐 정도: Gaussian blur 사용
# *   최종값: 0~1 사이로 정규화
# ```
#
#
#
# ```
# heatmap 생성 방식
#
# *   heatmap이 비어 있지 않음
# *   fixation이 많았던 위치에 강한 hotspot이 생김
# *   overlay했을 때 흉부 구조 위에 자연스럽게 올라감
# *   좌우/상하 반전 문제 없어 보임
# ```
#
#
#

# %% [markdown] ORIGINAL CELL 75
# 전체 heatmap 생성해서 Drive에 저장

# %% ORIGINAL CELL 76
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter

# =========================
# 설정
# =========================
TARGET_SIZE = 512
SIGMA = 7

HEATMAP_ROOT = Path(
    "/content/drive/MyDrive/X-ray/gaze_calibration_research/heatmaps_512_sigma7"
)
HEATMAP_ROOT.mkdir(parents=True, exist_ok=True)

# =========================
# 함수
# =========================
def make_heatmap_for_dicom(dicom_id, valid_fix, metadata, target_size=512, sigma=7):
    pts = valid_fix[valid_fix["DICOM_ID"] == dicom_id].copy()

    meta_row = metadata[metadata["dicom_id"] == dicom_id].iloc[0]
    orig_h = int(meta_row["Rows"])
    orig_w = int(meta_row["Columns"])

    heat = np.zeros((target_size, target_size), dtype=np.float32)

    if "FPOGD" in pts.columns:
        weights = pts["FPOGD"].fillna(1.0).astype(float).values
    else:
        weights = np.ones(len(pts), dtype=np.float32)

    xs = pts["X_ORIGINAL"].values / orig_w * (target_size - 1)
    ys = pts["Y_ORIGINAL"].values / orig_h * (target_size - 1)

    for x, y, w in zip(xs, ys, weights):
        xi = int(round(x))
        yi = int(round(y))
        if 0 <= xi < target_size and 0 <= yi < target_size:
            heat[yi, xi] += float(w)

    heat = gaussian_filter(heat, sigma=sigma)

    if heat.max() > 0:
        heat = heat / heat.max()

    return heat.astype(np.float32)

# =========================
# 전체 생성
# =========================
records = []

for i, dicom_id in enumerate(cohort["dicom_id"].astype(str), 1):
    heat = make_heatmap_for_dicom(
        dicom_id=dicom_id,
        valid_fix=valid_fix,
        metadata=metadata,
        target_size=TARGET_SIZE,
        sigma=SIGMA
    )

    save_path = HEATMAP_ROOT / f"{dicom_id}.npy"
    np.save(save_path, heat)

    records.append({
        "dicom_id": dicom_id,
        "heatmap_path": str(save_path),
        "heatmap_max": float(heat.max()),
        "heatmap_sum": float(heat.sum())
    })

    if i % 100 == 0:
        print(f"{i}/{len(cohort)} 완료")

heatmap_df = pd.DataFrame(records)

print("\n전체 heatmap 생성 완료")
print("생성 수:", len(heatmap_df))
print("저장 폴더:", HEATMAP_ROOT)

# %% [markdown] ORIGINAL CELL 77
# 생성 결과를 cohort에 붙여서 저장.

# %% ORIGINAL CELL 78
COHORT_WITH_HEATMAP_PATH = Path(
    "/content/drive/MyDrive/X-ray/gaze_calibration_research/splits/"
    "egd_cxr_cohort_with_split_and_heatmap_seed42.csv"
)

cohort_with_heatmap = cohort.merge(
    heatmap_df[["dicom_id", "heatmap_path", "heatmap_max", "heatmap_sum"]],
    on="dicom_id",
    how="left"
)

cohort_with_heatmap.to_csv(COHORT_WITH_HEATMAP_PATH, index=False)

print("저장 완료:", COHORT_WITH_HEATMAP_PATH)
print(cohort_with_heatmap.head())

# %% [markdown] ORIGINAL CELL 79
# 생성이 잘 됐는지 확인

# %% ORIGINAL CELL 80
print("cohort 수:", len(cohort_with_heatmap))
print("heatmap 경로 누락:", cohort_with_heatmap["heatmap_path"].isna().sum())
print("heatmap max가 0인 이미지 수:", (cohort_with_heatmap["heatmap_max"] <= 0).sum())
print("heatmap sum이 0인 이미지 수:", (cohort_with_heatmap["heatmap_sum"] <= 0).sum())

# %% [markdown] ORIGINAL CELL 81
# 학습용 데이터셋 클래스 구성
#
#
#
# ```
# A. Baseline
# *   입력: X-ray 이미지
# *   출력: 3-class label
#
# B. Gaze-supervised model
# *   입력: X-ray 이미지
# *   출력: 3-class label
# *   추가 supervision: heatmap
#
# ➡ 같은 포맷의heatmap으로 만들기.
# ```
#

# %% [markdown] ORIGINAL CELL 82
# PyTorch Dataset이 이미지·라벨·heatmap을 정확히 한 샘플 단위로 묶어 반환하는지 검증

# %% ORIGINAL CELL 83
# cohort_with_heatmap에 실제 JPG 경로를 붙임.

from pathlib import Path

IMAGE_ROOT = Path(
    "/content/drive/MyDrive/X-ray/data/mimic-cxr-jpg-egdcxr/images"
)

# dicom_id -> 실제 jpg 경로
path_by_id = {
    p.stem: str(p)
    for p in IMAGE_ROOT.rglob("*.jpg")
}

cohort_with_heatmap["image_path"] = (
    cohort_with_heatmap["dicom_id"]
    .astype(str)
    .map(path_by_id)
)

print("전체:", len(cohort_with_heatmap))
print("image_path 누락:", cohort_with_heatmap["image_path"].isna().sum())
print("heatmap_path 누락:", cohort_with_heatmap["heatmap_path"].isna().sum())

# %% ORIGINAL CELL 84
# 라벨을 숫자로 고정

label_map = {
    "Normal": 0,
    "CHF": 1,
    "Pneumonia": 2
}

cohort_with_heatmap["label"] = (
    cohort_with_heatmap["target"]
    .map(label_map)
)

print(
    cohort_with_heatmap[
        ["target", "label"]
    ]
    .drop_duplicates()
    .sort_values("label")
)

# %% [markdown] ORIGINAL CELL 85
# Datast 클래스 생성

# %% ORIGINAL CELL 86
label_map = {
    "Normal": 0,
    "CHF": 1,
    "Pneumonia": 2
}

cohort_with_heatmap["label"] = (
    cohort_with_heatmap["target"]
    .map(label_map)
)

print(
    cohort_with_heatmap[
        ["target", "label"]
    ]
    .drop_duplicates()
    .sort_values("label")
)

# %% ORIGINAL CELL 87
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

image_transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class EGDCXRDataset(Dataset):
    def __init__(self, df, image_transform=None):
        self.df = df.reset_index(drop=True)
        self.image_transform = image_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # X-ray
        img = Image.open(row["image_path"]).convert("RGB")

        if self.image_transform is not None:
            img = self.image_transform(img)

        # gaze heatmap
        heatmap = np.load(row["heatmap_path"]).astype(np.float32)
        heatmap = torch.from_numpy(heatmap).unsqueeze(0)

        # class label
        label = torch.tensor(
            int(row["label"]),
            dtype=torch.long
        )

        return {
            "image": img,
            "heatmap": heatmap,
            "label": label,
            "dicom_id": row["dicom_id"]
        }

print("EGDCXRDataset 정의 완료")

# %% ORIGINAL CELL 88
train_df = cohort_with_heatmap[
    cohort_with_heatmap["split"] == "train"
].copy()

train_dataset = EGDCXRDataset(
    train_df,
    image_transform=image_transform
)

sample = train_dataset[0]

print("image shape:", sample["image"].shape)
print("heatmap shape:", sample["heatmap"].shape)
print("label:", sample["label"])
print("dicom_id:", sample["dicom_id"])
print("heatmap min/max:",
      sample["heatmap"].min().item(),
      sample["heatmap"].max().item())

# %% ORIGINAL CELL 89
from torch.utils.data import DataLoader

train_df = cohort_with_heatmap[
    cohort_with_heatmap["split"] == "train"
].copy()

val_df = cohort_with_heatmap[
    cohort_with_heatmap["split"] == "val"
].copy()

test_df = cohort_with_heatmap[
    cohort_with_heatmap["split"] == "test"
].copy()

train_dataset = EGDCXRDataset(train_df, image_transform=image_transform)
val_dataset = EGDCXRDataset(val_df, image_transform=image_transform)
test_dataset = EGDCXRDataset(test_df, image_transform=image_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print("train:", len(train_dataset))
print("val:", len(val_dataset))
print("test:", len(test_dataset))

# %% [markdown] ORIGINAL CELL 90
# 실제 배치 확인

# %% ORIGINAL CELL 91
batch = next(iter(train_loader))

print("image batch:", batch["image"].shape)
print("heatmap batch:", batch["heatmap"].shape)
print("label batch:", batch["label"].shape)
print("labels:", batch["label"])

# %% [markdown] ORIGINAL CELL 92
# Baseline DenseNet-121 정의
#
# ➡ 아직 gaze loss는 넣지말고, 먼저 이미지 → 3-class 분류만 하는 기준 모델을 만듦

# %% ORIGINAL CELL 93
import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("device:", device)

model = densenet121(
    weights=DenseNet121_Weights.IMAGENET1K_V1
)

# 마지막 classifier를 3-class로 변경
in_features = model.classifier.in_features
model.classifier = nn.Linear(in_features, 3)

model = model.to(device)

print(model.classifier)

# %% [markdown] ORIGINAL CELL 94
# loss와 optimizer 정의

# %% ORIGINAL CELL 95
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

# %% [markdown] ORIGINAL CELL 96
# 학습 전체를 돌리기 전에, 배치 forward가 정상인지 먼저 확인.
#
#
#
# ```
# logits shape: torch.Size([8, 3])
# labels shape: torch.Size([8])
# loss: 1.0799726247787476
#
# ➡ 통과
# ```
#

# %% ORIGINAL CELL 97
batch = next(iter(train_loader))

images = batch["image"].to(device)
labels = batch["label"].to(device)

with torch.no_grad():
    logits = model(images)

print("logits shape:", logits.shape)
print("labels shape:", labels.shape)

loss = criterion(logits, labels)
print("loss:", loss.item())

# %% [markdown] ORIGINAL CELL 98
# Baseline DenseNet-121 학습 루프 + validation + best model 저장

# %% [markdown] ORIGINAL CELL 99
# 학습 코드로 학습 함수부터 정의

# %% ORIGINAL CELL 100
import copy
import torch

def train_baseline(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=10
):
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):

        # =====================
        # Train
        # =====================
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # =====================
        # Validation
        # =====================
        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                logits = model(images)
                loss = criterion(logits, labels)

                val_loss += loss.item() * images.size(0)

                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"val acc {val_acc:.4f}"
        )

        # best validation loss 기준 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

    # 가장 좋았던 모델 복원
    model.load_state_dict(best_state)

    print("\nBest val loss:", best_val_loss)

    return model

# %% [markdown] ORIGINAL CELL 101
# GPU 사용중인지 확인

# %% ORIGINAL CELL 102
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("device:", device)

# %% [markdown] ORIGINAL CELL 103
# Baseline 학습 + Drive checkpoint 저장

# %% ORIGINAL CELL 104
# checkpoint 폴더 생성

from pathlib import Path

CHECKPOINT_DIR = Path(
    "/content/drive/MyDrive/X-ray/gaze_calibration_research/checkpoints/baseline_densenet121"
)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pt"
LAST_CHECKPOINT_PATH = CHECKPOINT_DIR / "last_checkpoint.pt"

print("checkpoint 폴더:", CHECKPOINT_DIR)

# %% [markdown] ORIGINAL CELL 105
# 기존 tain_baseline() 함수를 아래 코드로 변경

# %% ORIGINAL CELL 106
import copy
import torch

def train_baseline(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=10,
    start_epoch=1,
    best_val_loss=float("inf")
):
    for epoch in range(start_epoch, epochs + 1):

        # =====================
        # Train
        # =====================
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # =====================
        # Validation
        # =====================
        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                logits = model(images)
                loss = criterion(logits, labels)

                val_loss += loss.item() * images.size(0)

                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"val acc {val_acc:.4f}"
        )

        # =====================
        # 마지막 상태 저장
        # =====================
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "train_loss": train_loss,
                "train_acc": train_acc
            },
            LAST_CHECKPOINT_PATH
        )

        # =====================
        # Best model 저장
        # =====================
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc
                },
                BEST_MODEL_PATH
            )

            print("  → Best model 저장")

    print("\n학습 완료")
    print("Best val loss:", best_val_loss)

    return model

# %% [markdown] ORIGINAL CELL 107
# 실제 학습
#
# **train_baseline()**
#
# ➡ 매 epoch마다 Drive에 "last_checkpoint.pt"가 저장됨.
#
# ➡ validation loss가 개선될 때마다 "best_model.pt"도 갱신됨.
#
#
#
# ```
# *   last_checkpoint.pt → 학습 이어하기용
# *   best_model.pt → 최종 평가용
# ```
#

# %% ORIGINAL CELL 108
baseline_model = train_baseline(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    epochs=10
)

# %% [markdown] ORIGINAL CELL 109
# 학습 도중 런타임이 끊기면 아래 코드 실행하고, 그 이후에 위 학습코드 start_ephoch 변경해서 돌리기

# %% ORIGINAL CELL 110
checkpoint = torch.load(
    LAST_CHECKPOINT_PATH,
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

start_epoch = checkpoint["epoch"] + 1
best_val_loss = checkpoint["best_val_loss"]

print("마지막 완료 epoch:", checkpoint["epoch"])
print("다음 시작 epoch:", start_epoch)

# %% [markdown] ORIGINAL CELL 111


# %% ORIGINAL CELL 112



# ============================================================================
# PART B. 원본 Colab 이후 IWAIT 2027 초록 제출 직전까지 실제 추가/수정/보완된 실험 코드
# ============================================================================

# %% [markdown]
# ⭐⭐공통 import / seed / 최종 실험 설정⭐⭐

# %%
import os
import copy
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import densenet121, DenseNet121_Weights

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 3
CLASS_NAMES = ["Normal", "CHF", "Pneumonia"]

BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4

GAZE_LAMBDA = 0.1
LESION_LAMBDA = 0.1

TARGET_SIZE = 512
SIGMA = 7
N_BINS = 15

print("device:", device)

RESEARCH_ROOT = (
    PROJECT_ROOT
    / "gaze_calibration_research"
)

RESEARCH_ROOT.mkdir(
    parents=True,
    exist_ok=True
)



# %% [markdown]
# ⭐⭐최종 cohort / DataLoader 재확인⭐⭐

# %%
# 원본 notebook에서 생성된 cohort_with_heatmap, train_df, val_df, test_df,
# image_transform, EGDCXRDataset을 그대로 사용한다.

assert "cohort_with_heatmap" in globals()
assert "train_df" in globals()
assert "val_df" in globals()
assert "test_df" in globals()

print("train:", len(train_df))
print("val:", len(val_df))
print("test:", len(test_df))
print(train_df["target"].value_counts())
print(val_df["target"].value_counts())
print(test_df["target"].value_counts())


# ============================================================================
# 1. 공통 평가 함수
# ============================================================================

# %% [markdown]
# ⭐⭐logits / labels 수집⭐⭐

# %%
def get_class_logits(model, images):
    out = model(images)

    if isinstance(out, (tuple, list)):
        return out[0]

    if isinstance(out, dict):
        if "logits" in out:
            return out["logits"]
        if "class_logits" in out:
            return out["class_logits"]

    return out


@torch.no_grad()
def collect_logits_labels(model, loader, device):
    model.eval()

    all_logits = []
    all_labels = []
    all_dicom_ids = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits = get_class_logits(model, images)

        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

        if "dicom_id" in batch:
            all_dicom_ids.extend(list(batch["dicom_id"]))

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return logits, labels, all_dicom_ids


# %% [markdown]
# ⭐⭐Calibration / Classification metric 함수⭐⭐

# %%
def expected_calibration_error(probs, labels, n_bins=15):
    probs = np.asarray(probs)
    labels = np.asarray(labels)

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == 0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences > lower) & (confidences <= upper)

        if mask.sum() == 0:
            continue

        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()

        ece += (mask.sum() / len(labels)) * abs(bin_acc - bin_conf)

    return float(ece)


def multiclass_brier_score(probs, labels, num_classes=3):
    probs = np.asarray(probs)
    labels = np.asarray(labels)

    onehot = np.eye(num_classes)[labels]

    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def safe_classwise_auroc(labels, probs, class_names=CLASS_NAMES):
    labels = np.asarray(labels)
    probs = np.asarray(probs)

    out = {}

    for c, name in enumerate(class_names):
        y = (labels == c).astype(int)

        if len(np.unique(y)) < 2:
            out[name] = np.nan
        else:
            out[name] = roc_auc_score(y, probs[:, c])

    return out


def evaluate_probs(labels, probs, n_bins=15, class_names=CLASS_NAMES):
    labels = np.asarray(labels)
    probs = np.asarray(probs)

    preds = probs.argmax(axis=1)

    accuracy = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")

    auroc_macro = roc_auc_score(
        labels,
        probs,
        multi_class="ovr",
        average="macro"
    )

    ece = expected_calibration_error(
        probs,
        labels,
        n_bins=n_bins
    )

    brier = multiclass_brier_score(
        probs,
        labels,
        num_classes=probs.shape[1]
    )

    nll = log_loss(
        labels,
        probs,
        labels=list(range(probs.shape[1]))
    )

    class_auc = safe_classwise_auroc(
        labels,
        probs,
        class_names=class_names
    )

    return {
        "Accuracy": float(accuracy),
        "F1_macro": float(f1_macro),
        "F1_weighted": float(f1_weighted),
        "AUROC_macro": float(auroc_macro),
        "ECE": float(ece),
        "Brier": float(brier),
        "NLL": float(nll),
        **{
            f"AUROC_{name}": float(class_auc[name])
            for name in class_names
        }
    }


def print_metric_report(name, metrics):
    print("=" * 70)
    print(name)
    print("=" * 70)

    for key, value in metrics.items():
        if isinstance(value, (float, np.floating)):
            print(f"{key:<20}: {value:.4f}")
        else:
            print(f"{key:<20}: {value}")


# ============================================================================
# 2. Baseline best checkpoint 평가
# ============================================================================

# %% [markdown]
# ⭐⭐Baseline best model 다시 불러오기⭐⭐

# %%
baseline_model = densenet121(
    weights=DenseNet121_Weights.IMAGENET1K_V1
)

baseline_in_features = baseline_model.classifier.in_features
baseline_model.classifier = nn.Linear(
    baseline_in_features,
    NUM_CLASSES
)

baseline_model = baseline_model.to(device)

baseline_checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=device
)

baseline_model.load_state_dict(
    baseline_checkpoint["model_state_dict"]
)

baseline_model.eval()

print(
    "Baseline best epoch:",
    baseline_checkpoint.get("epoch", None)
)


# %% [markdown]
# ⭐⭐Baseline validation / test logits 저장⭐⭐

# %%
baseline_val_logits, baseline_val_labels_t, baseline_val_ids = collect_logits_labels(
    baseline_model,
    val_loader,
    device
)

baseline_test_logits, baseline_test_labels_t, baseline_test_ids = collect_logits_labels(
    baseline_model,
    test_loader,
    device
)

baseline_val_labels = baseline_val_labels_t.numpy()
baseline_test_labels = baseline_test_labels_t.numpy()

baseline_val_probs_raw = torch.softmax(
    baseline_val_logits,
    dim=1
).numpy()

baseline_test_probs_raw = torch.softmax(
    baseline_test_logits,
    dim=1
).numpy()


# %% [markdown]
# ⭐⭐Baseline Uncalibrated 평가⭐⭐

# %%
baseline_raw_metrics = evaluate_probs(
    baseline_test_labels,
    baseline_test_probs_raw,
    n_bins=N_BINS
)

print_metric_report(
    "Baseline - Uncalibrated",
    baseline_raw_metrics
)


# ============================================================================
# 3. Temperature scaling
# ============================================================================

# %% [markdown]
# ⭐⭐TemperatureScaler⭐⭐

# %%
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()

        self.temperature = nn.Parameter(
            torch.ones(1)
        )

    def forward(self, logits):
        return logits / self.temperature

    def fit(
        self,
        logits,
        labels,
        lr=0.1,
        max_iter=100
    ):
        logits = logits.to(device)
        labels = labels.to(device)

        self.to(device)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.LBFGS(
            [self.temperature],
            lr=lr,
            max_iter=max_iter
        )

        def closure():
            optimizer.zero_grad()

            loss = criterion(
                self(logits),
                labels
            )

            loss.backward()

            return loss

        optimizer.step(closure)

        print(
            "Temperature:",
            self.temperature.item()
        )

        return self


def fit_temperature(val_logits, val_labels):
    scaler = TemperatureScaler()

    scaler.fit(
        val_logits,
        val_labels,
        lr=0.1,
        max_iter=100
    )

    return scaler


def apply_temperature(scaler, logits):
    scaler.eval()

    with torch.no_grad():
        calibrated_logits = scaler(
            logits.to(device)
        ).cpu()

    calibrated_probs = torch.softmax(
        calibrated_logits,
        dim=1
    ).numpy()

    return calibrated_probs


# %% [markdown]
# ⭐⭐Baseline Temperature Scaling⭐⭐

# %%
baseline_temperature = fit_temperature(
    baseline_val_logits,
    baseline_val_labels_t
)

baseline_val_probs_cal = apply_temperature(
    baseline_temperature,
    baseline_val_logits
)

baseline_test_probs_cal = apply_temperature(
    baseline_temperature,
    baseline_test_logits
)

baseline_cal_metrics = evaluate_probs(
    baseline_test_labels,
    baseline_test_probs_cal,
    n_bins=N_BINS
)

print_metric_report(
    "Baseline - Temperature Scaled",
    baseline_cal_metrics
)


# ============================================================================
# 4. Gaze-Original
# ============================================================================

# %% [markdown]
# ⭐⭐DenseNet121 + Gaze auxiliary head⭐⭐

# %%
class DenseNet121Gaze(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        backbone = densenet121(
            weights=DenseNet121_Weights.IMAGENET1K_V1
        )

        self.features = backbone.features

        feature_dim = backbone.classifier.in_features

        self.classifier = nn.Linear(
            feature_dim,
            num_classes
        )

        self.gaze_head = nn.Conv2d(
            feature_dim,
            1,
            kernel_size=1
        )

    def forward(self, x):
        features = self.features(x)
        features = F.relu(
            features,
            inplace=False
        )

        pooled = F.adaptive_avg_pool2d(
            features,
            output_size=(1, 1)
        ).flatten(1)

        class_logits = self.classifier(pooled)

        gaze_logits = self.gaze_head(
            features
        )

        gaze_logits = F.interpolate(
            gaze_logits,
            size=(TARGET_SIZE, TARGET_SIZE),
            mode="bilinear",
            align_corners=False
        )

        return class_logits, gaze_logits


# %% [markdown]
# ⭐⭐Gaze KL loss⭐⭐
# fixation이 없는 샘플은 gaze auxiliary loss에서 제외.
# target heatmap은 spatial probability distribution으로 normalize.

# %%
def gaze_kl_loss(
    gaze_logits,
    target_heatmap
):
    batch_size = gaze_logits.size(0)

    pred = gaze_logits.reshape(
        batch_size,
        -1
    )

    target = target_heatmap.reshape(
        batch_size,
        -1
    )

    target_sum = target.sum(
        dim=1
    )

    valid = target_sum > 0

    if valid.sum() == 0:
        return gaze_logits.sum() * 0.0

    pred = pred[valid]
    target = target[valid]

    target = target / (
        target.sum(
            dim=1,
            keepdim=True
        ) + 1e-12
    )

    log_pred = F.log_softmax(
        pred,
        dim=1
    )

    loss = F.kl_div(
        log_pred,
        target,
        reduction="batchmean"
    )

    return loss


# %% [markdown]
# ⭐⭐Gaze model 학습 함수⭐⭐
# 최종 checkpoint 선택은 validation classification loss 기준.

# %%
def train_gaze_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    checkpoint_dir,
    epochs=10,
    gaze_lambda=0.1
):
    checkpoint_dir = Path(
        checkpoint_dir
    )
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    best_model_path = (
        checkpoint_dir / "best_model.pt"
    )

    last_checkpoint_path = (
        checkpoint_dir / "last_checkpoint.pt"
    )

    cls_criterion = nn.CrossEntropyLoss()

    best_val_cls_loss = float("inf")

    history = []

    for epoch in range(
        1,
        epochs + 1
    ):
        print(
            f"\n===== Epoch {epoch}/{epochs} ====="
        )

        # ---------------------
        # Train
        # ---------------------
        model.train()

        train_total_loss = 0.0
        train_cls_loss = 0.0
        train_gaze_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            heatmaps = batch["heatmap"].to(device)

            optimizer.zero_grad()

            class_logits, gaze_logits = model(
                images
            )

            cls_loss = cls_criterion(
                class_logits,
                labels
            )

            aux_loss = gaze_kl_loss(
                gaze_logits,
                heatmaps
            )

            loss = (
                cls_loss
                + gaze_lambda * aux_loss
            )

            loss.backward()
            optimizer.step()

            bs = images.size(0)

            train_total_loss += (
                loss.item() * bs
            )
            train_cls_loss += (
                cls_loss.item() * bs
            )
            train_gaze_loss += (
                aux_loss.item() * bs
            )

            pred = class_logits.argmax(
                dim=1
            )

            train_correct += (
                pred == labels
            ).sum().item()

            train_total += bs

        train_total_loss /= train_total
        train_cls_loss /= train_total
        train_gaze_loss /= train_total
        train_acc = (
            train_correct / train_total
        )

        # ---------------------
        # Validation
        # ---------------------
        model.eval()

        val_total_loss = 0.0
        val_cls_loss = 0.0
        val_gaze_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                heatmaps = batch["heatmap"].to(device)

                class_logits, gaze_logits = model(
                    images
                )

                cls_loss = cls_criterion(
                    class_logits,
                    labels
                )

                aux_loss = gaze_kl_loss(
                    gaze_logits,
                    heatmaps
                )

                loss = (
                    cls_loss
                    + gaze_lambda * aux_loss
                )

                bs = images.size(0)

                val_total_loss += (
                    loss.item() * bs
                )
                val_cls_loss += (
                    cls_loss.item() * bs
                )
                val_gaze_loss += (
                    aux_loss.item() * bs
                )

                pred = class_logits.argmax(
                    dim=1
                )

                val_correct += (
                    pred == labels
                ).sum().item()

                val_total += bs

        val_total_loss /= val_total
        val_cls_loss /= val_total
        val_gaze_loss /= val_total
        val_acc = (
            val_correct / val_total
        )

        row = {
            "epoch": epoch,
            "train_total_loss": train_total_loss,
            "train_cls_loss": train_cls_loss,
            "train_gaze_loss": train_gaze_loss,
            "train_acc": train_acc,
            "val_total_loss": val_total_loss,
            "val_cls_loss": val_cls_loss,
            "val_gaze_loss": val_gaze_loss,
            "val_acc": val_acc
        }

        history.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"train cls {train_cls_loss:.4f} | "
            f"train gaze {train_gaze_loss:.4f} | "
            f"train acc {train_acc:.4f} | "
            f"val cls {val_cls_loss:.4f} | "
            f"val gaze {val_gaze_loss:.4f} | "
            f"val acc {val_acc:.4f}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_cls_loss": val_cls_loss,
                "val_total_loss": val_total_loss,
                "val_acc": val_acc
            },
            last_checkpoint_path
        )

        if val_cls_loss < best_val_cls_loss:
            best_val_cls_loss = val_cls_loss

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_cls_loss": val_cls_loss,
                    "val_total_loss": val_total_loss,
                    "val_acc": val_acc
                },
                best_model_path
            )

            print(
                "  → Best model 저장"
            )

    history = pd.DataFrame(history)

    history.to_csv(
        checkpoint_dir / "history.csv",
        index=False
    )

    best_checkpoint = torch.load(
        best_model_path,
        map_location=device
    )

    model.load_state_dict(
        best_checkpoint["model_state_dict"]
    )

    return model, history


# %% [markdown]
# ⭐⭐Gaze-Original train / val / test loader⭐⭐

# %%
gaze_original_train_dataset = EGDCXRDataset(
    train_df,
    image_transform=image_transform
)

gaze_original_val_dataset = EGDCXRDataset(
    val_df,
    image_transform=image_transform
)

gaze_original_test_dataset = EGDCXRDataset(
    test_df,
    image_transform=image_transform
)

gaze_original_train_loader = DataLoader(
    gaze_original_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

gaze_original_val_loader = DataLoader(
    gaze_original_val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

gaze_original_test_loader = DataLoader(
    gaze_original_test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)



# %%
GAZE_ORIGINAL_DIR = (
    RESEARCH_ROOT
    / "checkpoints"
    / "gaze_original_densenet121"
)

GAZE_ORIGINAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

GAZE_ORIGINAL_BEST_PATH = (
    GAZE_ORIGINAL_DIR
    / "best_model.pt"
)

GAZE_ORIGINAL_LAST_PATH = (
    GAZE_ORIGINAL_DIR
    / "last_checkpoint.pt"
)

GAZE_ORIGINAL_HISTORY_PATH = (
    GAZE_ORIGINAL_DIR
    / "history.pt"
)

GAZE_ORIGINAL_EPOCHS = 10
GAZE_ORIGINAL_LAMBDA_GAZE = 0.1


# %% [markdown]
# ⭐⭐Gaze-Original 학습⭐⭐

# %%
gaze_original_model = DenseNet121Gaze(
    num_classes=NUM_CLASSES
).to(device)

gaze_original_optimizer = torch.optim.AdamW(
    gaze_original_model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

GAZE_ORIGINAL_CKPT_DIR = GAZE_ORIGINAL_DIR

gaze_original_model, gaze_original_history = train_gaze_model(
    model=gaze_original_model,
    train_loader=gaze_original_train_loader,
    val_loader=gaze_original_val_loader,
    optimizer=gaze_original_optimizer,
    device=device,
    checkpoint_dir=GAZE_ORIGINAL_CKPT_DIR,
    epochs=EPOCHS,
    gaze_lambda=GAZE_LAMBDA
)


# %% [markdown]
# ⭐⭐Gaze-Original 평가 / temperature scaling⭐⭐

# %%
gaze_original_val_logits, gaze_original_val_labels_t, gaze_original_val_ids = collect_logits_labels(
    gaze_original_model,
    gaze_original_val_loader,
    device
)

gaze_original_test_logits, gaze_original_test_labels_t, gaze_original_test_ids = collect_logits_labels(
    gaze_original_model,
    gaze_original_test_loader,
    device
)

gaze_original_val_labels = (
    gaze_original_val_labels_t.numpy()
)
gaze_original_test_labels = (
    gaze_original_test_labels_t.numpy()
)

gaze_original_val_probs_raw = torch.softmax(
    gaze_original_val_logits,
    dim=1
).numpy()

gaze_original_test_probs_raw = torch.softmax(
    gaze_original_test_logits,
    dim=1
).numpy()

gaze_original_raw_metrics = evaluate_probs(
    gaze_original_test_labels,
    gaze_original_test_probs_raw,
    n_bins=N_BINS
)

print_metric_report(
    "Gaze-Original - Uncalibrated",
    gaze_original_raw_metrics
)

gaze_original_temperature = fit_temperature(
    gaze_original_val_logits,
    gaze_original_val_labels_t
)

gaze_original_val_probs_cal = apply_temperature(
    gaze_original_temperature,
    gaze_original_val_logits
)

gaze_original_test_probs_cal = apply_temperature(
    gaze_original_temperature,
    gaze_original_test_logits
)

gaze_original_cal_metrics = evaluate_probs(
    gaze_original_test_labels,
    gaze_original_test_probs_cal,
    n_bins=N_BINS
)

print_metric_report(
    "Gaze-Original - Temperature Scaled",
    gaze_original_cal_metrics
)



# %% [markdown]
# ⭐⭐당시 평가 변수명 alias⭐⭐

# %%
gaze_val_logits = gaze_original_val_logits
gaze_val_labels = gaze_original_val_labels_t
gaze_test_logits = gaze_original_test_logits
gaze_test_labels = gaze_original_test_labels_t

gaze_temperature = gaze_original_temperature
gaze_raw_metrics = gaze_original_raw_metrics
gaze_cal_metrics = gaze_original_cal_metrics


# ============================================================================
# 5. Patient-level Shuffled-Gaze
# ============================================================================


# %% [markdown]
# ⭐⭐Shuffled-Gaze 초기 버전⭐⭐
# 2026-08-15 첫 구현.
# 같은 image 자체가 다시 매칭되지 않도록 derangement를 만들었지만,
# 이후 같은 patient의 다른 image가 donor가 될 수 있다는 점을 발견해
# 아래 PatientShuffledGazeDataset으로 교체하고 새 checkpoint에서 재학습했다.

# %%
class ShuffledGazeDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        seed=42
    ):
        self.base_dataset = base_dataset
        self.seed = seed

        n = len(base_dataset)
        indices = np.arange(n)

        rng = np.random.default_rng(
            seed
        )

        shuffled = rng.permutation(
            n
        )

        while not np.all(
            shuffled != indices
        ):
            shuffled = rng.permutation(
                n
            )

        self.shuffled_indices = (
            shuffled
        )

    def __len__(self):
        return len(
            self.base_dataset
        )

    def __getitem__(self, idx):
        sample = self.base_dataset[
            idx
        ]

        shuffled_idx = int(
            self.shuffled_indices[
                idx
            ]
        )

        shuffled_sample = (
            self.base_dataset[
                shuffled_idx
            ]
        )

        return {
            "image": sample["image"],
            "heatmap": shuffled_sample[
                "heatmap"
            ],
            "label": sample["label"],
            "dicom_id": sample[
                "dicom_id"
            ],
            "gaze_source_id": shuffled_sample[
                "dicom_id"
            ]
        }


# %% [markdown]
# ⭐⭐초기 shuffled-gaze checkpoint 설정⭐⭐
# 이 결과는 preliminary/debug로만 두고,
# patient-level 수정본을 최종 Shuffled-Gaze로 사용.

# %%
SHUFFLED_GAZE_DIR = (
    RESEARCH_ROOT
    / "checkpoints"
    / "gaze_shuffled_densenet121_preliminary"
)

SHUFFLED_GAZE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SHUFFLED_BEST_PATH = (
    SHUFFLED_GAZE_DIR
    / "best_model.pt"
)

SHUFFLED_LAST_PATH = (
    SHUFFLED_GAZE_DIR
    / "last_checkpoint.pt"
)

SHUFFLED_HISTORY_PATH = (
    SHUFFLED_GAZE_DIR
    / "history.pt"
)

SHUFFLED_EPOCHS = 10
SHUFFLED_LAMBDA_GAZE = 0.1


# %% [markdown]
# ⭐⭐PatientShuffledGazeDataset⭐⭐
# 같은 환자의 gaze가 자기 영상에 다시 배정되지 않도록 patient_id가 다른 donor만 선택.
# mapping은 seed=42로 고정되어 epoch마다 바뀌지 않음.
# gaze_source_id / gaze_source_patient_id를 보존해서 검증 가능하게 함.

# %%
class PatientShuffledGazeDataset(Dataset):
    def __init__(
        self,
        df,
        image_transform=None,
        seed=42
    ):
        self.df = (
            df.reset_index(drop=True)
            .copy()
        )

        self.image_transform = image_transform
        self.seed = seed

        rng = np.random.default_rng(
            seed
        )

        patient_ids = (
            self.df["patient_id"]
            .astype(str)
            .values
        )

        self.gaze_source_indices = []

        all_indices = np.arange(
            len(self.df)
        )

        for i in range(
            len(self.df)
        ):
            candidate_indices = all_indices[
                patient_ids
                != patient_ids[i]
            ]

            donor = int(
                rng.choice(
                    candidate_indices
                )
            )

            self.gaze_source_indices.append(
                donor
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        gaze_idx = (
            self.gaze_source_indices[idx]
        )

        gaze_row = self.df.iloc[
            gaze_idx
        ]

        img = Image.open(
            row["image_path"]
        ).convert("RGB")

        if self.image_transform is not None:
            img = self.image_transform(
                img
            )

        heatmap = np.load(
            gaze_row["heatmap_path"]
        ).astype(np.float32)

        heatmap = torch.from_numpy(
            heatmap
        ).unsqueeze(0)

        label = torch.tensor(
            int(row["label"]),
            dtype=torch.long
        )

        return {
            "image": img,
            "heatmap": heatmap,
            "label": label,
            "dicom_id": row["dicom_id"],
            "patient_id": str(row["patient_id"]),
            "gaze_source_id": gaze_row["dicom_id"],
            "gaze_source_patient_id": str(
                gaze_row["patient_id"]
            )
        }


# %% [markdown]
# ⭐⭐Shuffled mapping 검증⭐⭐

# %%
shuffled_check_dataset = PatientShuffledGazeDataset(
    train_df,
    image_transform=image_transform,
    seed=SEED
)

self_patient_count = 0

for i in range(
    len(shuffled_check_dataset)
):
    sample = shuffled_check_dataset[i]

    if (
        sample["patient_id"]
        == sample["gaze_source_patient_id"]
    ):
        self_patient_count += 1

print(
    "same-patient shuffled gaze:",
    self_patient_count
)

assert self_patient_count == 0


# %% [markdown]
# ⭐⭐Shuffled-Gaze loaders⭐⭐

# %%
shuffled_train_dataset = PatientShuffledGazeDataset(
    train_df,
    image_transform=image_transform,
    seed=SEED
)

shuffled_val_dataset = PatientShuffledGazeDataset(
    val_df,
    image_transform=image_transform,
    seed=SEED
)

shuffled_test_dataset = PatientShuffledGazeDataset(
    test_df,
    image_transform=image_transform,
    seed=SEED
)

shuffled_train_loader = DataLoader(
    shuffled_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

shuffled_val_loader = DataLoader(
    shuffled_val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

shuffled_test_loader = DataLoader(
    shuffled_test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


# %% [markdown]
# ⭐⭐Shuffled-Gaze 학습 / 평가⭐⭐

# %%
shuffled_gaze_model = DenseNet121Gaze(
    num_classes=NUM_CLASSES
).to(device)

shuffled_gaze_optimizer = torch.optim.AdamW(
    shuffled_gaze_model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

SHUFFLED_GAZE_CKPT_DIR = Path(
    str(RESEARCH_ROOT / "checkpoints" / "gaze_shuffled_densenet121")
)

shuffled_gaze_model, shuffled_gaze_history = train_gaze_model(
    model=shuffled_gaze_model,
    train_loader=shuffled_train_loader,
    val_loader=shuffled_val_loader,
    optimizer=shuffled_gaze_optimizer,
    device=device,
    checkpoint_dir=SHUFFLED_GAZE_CKPT_DIR,
    epochs=EPOCHS,
    gaze_lambda=GAZE_LAMBDA
)

shuffled_val_logits, shuffled_val_labels_t, shuffled_val_ids = collect_logits_labels(
    shuffled_gaze_model,
    shuffled_val_loader,
    device
)

shuffled_test_logits, shuffled_test_labels_t, shuffled_test_ids = collect_logits_labels(
    shuffled_gaze_model,
    shuffled_test_loader,
    device
)

shuffled_val_labels = (
    shuffled_val_labels_t.numpy()
)
shuffled_test_labels = (
    shuffled_test_labels_t.numpy()
)

shuffled_val_probs_raw = torch.softmax(
    shuffled_val_logits,
    dim=1
).numpy()

shuffled_test_probs_raw = torch.softmax(
    shuffled_test_logits,
    dim=1
).numpy()

shuffled_raw_metrics = evaluate_probs(
    shuffled_test_labels,
    shuffled_test_probs_raw,
    n_bins=N_BINS
)

print_metric_report(
    "Shuffled-Gaze - Uncalibrated",
    shuffled_raw_metrics
)

shuffled_temperature = fit_temperature(
    shuffled_val_logits,
    shuffled_val_labels_t
)

shuffled_val_probs_cal = apply_temperature(
    shuffled_temperature,
    shuffled_val_logits
)

shuffled_test_probs_cal = apply_temperature(
    shuffled_temperature,
    shuffled_test_logits
)

shuffled_cal_metrics = evaluate_probs(
    shuffled_test_labels,
    shuffled_test_probs_cal,
    n_bins=N_BINS
)

print_metric_report(
    "Shuffled-Gaze - Temperature Scaled",
    shuffled_cal_metrics
)


# ============================================================================
# 6. Random-Gaze
# ============================================================================

# %% [markdown]
# ⭐⭐RandomGazeDataset⭐⭐
# 당시 Random-Gaze:
# - seed=42
# - target_size=512
# - sigma=7
# - original heatmap의 positive_count를 사용
# - n_points = max(1, min(positive_count, 30))
# - random points를 gaussian_filter 후 max-normalize
# - sample별 map을 미리 만들어 cache → epoch마다 고정

# %%
class RandomGazeDataset(Dataset):
    def __init__(
        self,
        df,
        image_transform=None,
        seed=42,
        target_size=512,
        sigma=7
    ):
        self.df = (
            df.reset_index(drop=True)
            .copy()
        )

        self.image_transform = (
            image_transform
        )

        self.seed = seed
        self.target_size = target_size
        self.sigma = sigma

        rng = np.random.default_rng(
            seed
        )

        self.random_heatmaps = []

        for idx in range(
            len(self.df)
        ):
            row = self.df.iloc[idx]

            original_heatmap = np.load(
                row["heatmap_path"]
            ).astype(np.float32)

            positive_count = int(
                np.count_nonzero(
                    original_heatmap > 0
                )
            )

            n_points = max(
                1,
                min(
                    positive_count,
                    30
                )
            )

            random_heatmap = np.zeros(
                (
                    target_size,
                    target_size
                ),
                dtype=np.float32
            )

            xs = rng.integers(
                0,
                target_size,
                size=n_points
            )

            ys = rng.integers(
                0,
                target_size,
                size=n_points
            )

            for x, y in zip(
                xs,
                ys
            ):
                random_heatmap[
                    y,
                    x
                ] += 1.0

            random_heatmap = gaussian_filter(
                random_heatmap,
                sigma=sigma
            )

            if (
                random_heatmap.max()
                > 0
            ):
                random_heatmap = (
                    random_heatmap
                    / random_heatmap.max()
                )

            self.random_heatmaps.append(
                random_heatmap.astype(
                    np.float32
                )
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(
            row["image_path"]
        ).convert("RGB")

        if self.image_transform is not None:
            img = self.image_transform(
                img
            )

        heatmap = torch.from_numpy(
            self.random_heatmaps[idx]
        ).unsqueeze(0)

        label = torch.tensor(
            int(row["label"]),
            dtype=torch.long
        )

        return {
            "image": img,
            "heatmap": heatmap,
            "label": label,
            "dicom_id": row["dicom_id"]
        }



# %%
RANDOM_GAZE_DIR = (
    RESEARCH_ROOT
    / "checkpoints"
    / "gaze_random_densenet121"
)

RANDOM_GAZE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RANDOM_BEST_PATH = (
    RANDOM_GAZE_DIR
    / "best_model.pt"
)

RANDOM_LAST_PATH = (
    RANDOM_GAZE_DIR
    / "last_checkpoint.pt"
)

RANDOM_HISTORY_PATH = (
    RANDOM_GAZE_DIR
    / "history.pt"
)

RANDOM_EPOCHS = 10
RANDOM_LAMBDA_GAZE = 0.1


# %% [markdown]
# ⭐⭐Random-Gaze loaders / 학습 / 평가⭐⭐

# %%
random_train_dataset = RandomGazeDataset(
    train_df,
    image_transform=image_transform,
    seed=SEED,
    target_size=TARGET_SIZE,
    sigma=SIGMA
)

random_val_dataset = RandomGazeDataset(
    val_df,
    image_transform=image_transform,
    seed=SEED,
    target_size=TARGET_SIZE,
    sigma=SIGMA
)

random_test_dataset = RandomGazeDataset(
    test_df,
    image_transform=image_transform,
    seed=SEED,
    target_size=TARGET_SIZE,
    sigma=SIGMA
)

random_train_loader = DataLoader(
    random_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

random_val_loader = DataLoader(
    random_val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

random_test_loader = DataLoader(
    random_test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

random_gaze_model = DenseNet121Gaze(
    num_classes=NUM_CLASSES
).to(device)

random_gaze_optimizer = torch.optim.AdamW(
    random_gaze_model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

RANDOM_GAZE_CKPT_DIR = RANDOM_GAZE_DIR

random_gaze_model, random_gaze_history = train_gaze_model(
    model=random_gaze_model,
    train_loader=random_train_loader,
    val_loader=random_val_loader,
    optimizer=random_gaze_optimizer,
    device=device,
    checkpoint_dir=RANDOM_GAZE_CKPT_DIR,
    epochs=EPOCHS,
    gaze_lambda=GAZE_LAMBDA
)

random_val_logits, random_val_labels_t, random_val_ids = collect_logits_labels(
    random_gaze_model,
    random_val_loader,
    device
)

random_test_logits, random_test_labels_t, random_test_ids = collect_logits_labels(
    random_gaze_model,
    random_test_loader,
    device
)

random_val_labels = (
    random_val_labels_t.numpy()
)
random_test_labels = (
    random_test_labels_t.numpy()
)

random_val_probs_raw = torch.softmax(
    random_val_logits,
    dim=1
).numpy()

random_test_probs_raw = torch.softmax(
    random_test_logits,
    dim=1
).numpy()

random_raw_metrics = evaluate_probs(
    random_test_labels,
    random_test_probs_raw,
    n_bins=N_BINS
)

print_metric_report(
    "Random-Gaze - Uncalibrated",
    random_raw_metrics
)

random_temperature = fit_temperature(
    random_val_logits,
    random_val_labels_t
)

random_val_probs_cal = apply_temperature(
    random_temperature,
    random_val_logits
)

random_test_probs_cal = apply_temperature(
    random_temperature,
    random_test_logits
)

random_cal_metrics = evaluate_probs(
    random_test_labels,
    random_test_probs_cal,
    n_bins=N_BINS
)

print_metric_report(
    "Random-Gaze - Temperature Scaled",
    random_cal_metrics
)


# ============================================================================
# 7. REFLACX overlap 확인
# ============================================================================

# %% [markdown]
# ⭐⭐REFLACX ROI overlap 검토⭐⭐
# 최종 확인 결과:
# EGD-CXR 1,083개 중 overlap 10개(0.92%) → lesion control로 사용하지 않음.

# %%
def check_ref_lacx_overlap(
    cohort_df,
    reflacx_df,
    egd_dicom_col="dicom_id",
    reflacx_dicom_col="dicom_id"
):
    egd_ids = set(
        cohort_df[
            egd_dicom_col
        ].astype(str)
    )

    reflacx_ids = set(
        reflacx_df[
            reflacx_dicom_col
        ].astype(str)
    )

    overlap = (
        egd_ids
        & reflacx_ids
    )

    print(
        "EGD-CXR:",
        len(egd_ids)
    )
    print(
        "REFLACX:",
        len(reflacx_ids)
    )
    print(
        "overlap:",
        len(overlap)
    )
    print(
        "overlap rate:",
        len(overlap)
        / len(egd_ids)
    )

    return overlap


# ============================================================================
# 8. MIMIC-CXR-Ext-ILS lesion mask preprocessing
# ============================================================================
#
# RECONSTRUCTION STATUS:
# - mask loading / resizing / union / saving 함수는 당시 기록에서 복구됨.
# - 그러나 현재 확보된 대화 기반 자료에서는 아래
#   `save_ext_ils_union_masks(...)`를 실제 metadata dataframe에 호출한
#   실행 셀을 확인하지 못함.
# - 따라서 이 복구본은 존재하지 않는 호출을 임의로 생성하지 않는다.
# - 즉, 이 section은 "전처리 함수 정의 보존"이며,
#   원본 Ext-ILS metadata -> union .npy 생성까지 end-to-end 연결된
#   독립 실행 pipeline이라고 주장하지 않는다.
#
# %% [markdown]
# ⭐⭐MIMIC-CXR-Ext-ILS mask union 생성⭐⭐
# 당시 확인:
# - positive mask annotation: 406
# - unique DICOM: 270
# - 복구 후 valid PNG mask: 328 unique mask
# - lesion type:
#   cardiomegaly 219 / edema 35 / opacity 25 / pneumonia 16 /
#   effusion 15 / atelectasis 10 / consolidation 8
#
# mask:
# Image.open(...).convert("L")
# Resize 512x512 = NEAREST
# arr > 0
# 같은 dicom_id의 여러 lesion mask는 np.maximum으로 union.
#
# 서버에서 사용한 root 기록:
# EXT_ILS_MASK_ROOT = /data/mimic-cxr-ext-ils/1.0.0/lesion_mask
# Colab/Drive에서는 실제 저장 위치에 맞게 변경.

# %%
EXT_ILS_MASK_ROOT = Path(
    "/data/mimic-cxr-ext-ils/1.0.0/lesion_mask"
)

LESION_UNION_ROOT = Path(
    "/content/drive/MyDrive/X-ray/"
    "gaze_calibration_research/"
    "lesion_masks_ext_ils_512"
)

LESION_UNION_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# %%
def load_binary_lesion_mask(
    mask_path,
    target_size=512
):
    mask = Image.open(
        mask_path
    ).convert("L")

    mask = mask.resize(
        (
            target_size,
            target_size
        ),
        resample=Image.Resampling.NEAREST
    )

    arr = np.asarray(
        mask
    )

    arr = (
        arr > 0
    ).astype(
        np.float32
    )

    return arr


def build_union_mask(
    mask_paths,
    target_size=512
):
    union = np.zeros(
        (
            target_size,
            target_size
        ),
        dtype=np.float32
    )

    for mask_path in mask_paths:
        arr = load_binary_lesion_mask(
            mask_path,
            target_size=target_size
        )

        union = np.maximum(
            union,
            arr
        )

    return union


# %% [markdown]
# ⭐⭐Ext-ILS metadata가 dicom_id / mask_path 형태로 정리되어 있을 때 union 저장⭐⭐

# %%
def save_ext_ils_union_masks(
    ext_ils_mask_df,
    dicom_col="dicom_id",
    mask_path_col="mask_path",
    output_root=LESION_UNION_ROOT,
    target_size=512
):
    output_root = Path(
        output_root
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True
    )

    records = []

    for dicom_id, group in ext_ils_mask_df.groupby(
        dicom_col
    ):
        mask_paths = [
            Path(p)
            for p in group[
                mask_path_col
            ].tolist()
        ]

        union = build_union_mask(
            mask_paths,
            target_size=target_size
        )

        save_path = (
            output_root
            / f"{dicom_id}.npy"
        )

        np.save(
            save_path,
            union
        )

        records.append({
            "dicom_id": str(dicom_id),
            "lesion_mask_path": str(save_path),
            "lesion_positive_pixels": int(
                union.sum()
            )
        })

    return pd.DataFrame(
        records
    )


# ============================================================================
# 9. Lesion-Supervised dataset / model
# ============================================================================

# %% [markdown]
# ⭐⭐LesionSupervisedDataset⭐⭐
# mask 없는 샘플:
# - classification에는 포함
# - lesion mask는 zero mask
# - lesion_valid=False
# - auxiliary lesion loss에서는 제외

# %%
class LesionSupervisedDataset(Dataset):
    def __init__(
        self,
        df,
        image_transform=None
    ):
        self.df = (
            df.reset_index(drop=True)
            .copy()
        )

        self.image_transform = image_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(
            row["image_path"]
        ).convert("RGB")

        if self.image_transform is not None:
            img = self.image_transform(
                img
            )

        lesion_path = row.get(
            "lesion_mask_path",
            np.nan
        )

        lesion_valid = (
            pd.notna(lesion_path)
            and Path(
                lesion_path
            ).exists()
        )

        if lesion_valid:
            lesion = np.load(
                lesion_path
            ).astype(
                np.float32
            )
        else:
            lesion = np.zeros(
                (
                    TARGET_SIZE,
                    TARGET_SIZE
                ),
                dtype=np.float32
            )

        lesion = torch.from_numpy(
            lesion
        ).unsqueeze(0)

        label = torch.tensor(
            int(row["label"]),
            dtype=torch.long
        )

        return {
            "image": img,
            "label": label,
            "lesion_mask": lesion,
            "lesion_valid": torch.tensor(
                lesion_valid,
                dtype=torch.bool
            ),
            "dicom_id": row["dicom_id"]
        }


# %% [markdown]
# ⭐⭐DenseNet121Lesion⭐⭐

# %%
class DenseNet121Lesion(nn.Module):
    def __init__(
        self,
        num_classes=3
    ):
        super().__init__()

        backbone = densenet121(
            weights=DenseNet121_Weights.IMAGENET1K_V1
        )

        self.features = backbone.features

        feature_dim = (
            backbone.classifier.in_features
        )

        self.classifier = nn.Linear(
            feature_dim,
            num_classes
        )

        self.lesion_head = nn.Conv2d(
            feature_dim,
            1,
            kernel_size=1
        )

    def forward(self, x):
        features = self.features(x)
        features = F.relu(
            features,
            inplace=False
        )

        pooled = F.adaptive_avg_pool2d(
            features,
            output_size=(1, 1)
        ).flatten(1)

        class_logits = self.classifier(
            pooled
        )

        lesion_logits = self.lesion_head(
            features
        )

        return (
            class_logits,
            lesion_logits
        )


# %% [markdown]
# ⭐⭐BCEWithLogits + Soft Dice⭐⭐

# %%
def soft_dice_loss_from_logits(
    logits,
    targets,
    eps=1e-6
):
    probs = torch.sigmoid(
        logits
    )

    probs = probs.flatten(1)
    targets = targets.flatten(1)

    intersection = (
        probs * targets
    ).sum(dim=1)

    denominator = (
        probs.sum(dim=1)
        + targets.sum(dim=1)
    )

    dice = (
        2.0 * intersection
        + eps
    ) / (
        denominator
        + eps
    )

    return (
        1.0 - dice
    ).mean()


def lesion_aux_loss(
    lesion_logits,
    lesion_target,
    lesion_valid
):
    lesion_valid = (
        lesion_valid.bool()
    )

    if lesion_valid.sum() == 0:
        return (
            lesion_logits.sum()
            * 0.0
        )

    pred = lesion_logits[
        lesion_valid
    ]

    target = lesion_target[
        lesion_valid
    ]

    # 당시 구현: feature-map 크기에 맞춰 target을 area mode로 downsample
    target = F.interpolate(
        target,
        size=pred.shape[-2:],
        mode="area"
    )

    bce = F.binary_cross_entropy_with_logits(
        pred,
        target
    )

    dice = soft_dice_loss_from_logits(
        pred,
        target
    )

    return bce + dice


# %% [markdown]
# ⭐⭐Lesion-Supervised 학습 함수⭐⭐
# selection criterion = validation classification loss.

# %%
def train_lesion_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    checkpoint_dir,
    epochs=10,
    lesion_lambda=0.1
):
    checkpoint_dir = Path(
        checkpoint_dir
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    best_model_path = (
        checkpoint_dir
        / "best_model.pt"
    )

    last_checkpoint_path = (
        checkpoint_dir
        / "last_checkpoint.pt"
    )

    cls_criterion = nn.CrossEntropyLoss()

    best_val_cls_loss = float(
        "inf"
    )

    history = []

    for epoch in range(
        1,
        epochs + 1
    ):
        print(
            f"\n===== Epoch {epoch}/{epochs} ====="
        )

        model.train()

        train_cls_sum = 0.0
        train_lesion_sum = 0.0
        train_total_sum = 0.0
        train_correct = 0
        train_n = 0

        for batch in train_loader:
            images = batch["image"].to(
                device
            )

            labels = batch["label"].to(
                device
            )

            lesion_target = batch[
                "lesion_mask"
            ].to(device)

            lesion_valid = batch[
                "lesion_valid"
            ].to(device)

            optimizer.zero_grad()

            class_logits, lesion_logits = model(
                images
            )

            cls_loss = cls_criterion(
                class_logits,
                labels
            )

            aux_loss = lesion_aux_loss(
                lesion_logits,
                lesion_target,
                lesion_valid
            )

            total_loss = (
                cls_loss
                + lesion_lambda
                * aux_loss
            )

            total_loss.backward()
            optimizer.step()

            bs = images.size(0)

            train_cls_sum += (
                cls_loss.item() * bs
            )
            train_lesion_sum += (
                aux_loss.item() * bs
            )
            train_total_sum += (
                total_loss.item() * bs
            )

            train_correct += (
                class_logits.argmax(1)
                == labels
            ).sum().item()

            train_n += bs

        train_cls = (
            train_cls_sum / train_n
        )
        train_lesion = (
            train_lesion_sum / train_n
        )
        train_total = (
            train_total_sum / train_n
        )
        train_acc = (
            train_correct / train_n
        )

        model.eval()

        val_cls_sum = 0.0
        val_lesion_sum = 0.0
        val_total_sum = 0.0
        val_correct = 0
        val_n = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(
                    device
                )

                labels = batch["label"].to(
                    device
                )

                lesion_target = batch[
                    "lesion_mask"
                ].to(device)

                lesion_valid = batch[
                    "lesion_valid"
                ].to(device)

                class_logits, lesion_logits = model(
                    images
                )

                cls_loss = cls_criterion(
                    class_logits,
                    labels
                )

                aux_loss = lesion_aux_loss(
                    lesion_logits,
                    lesion_target,
                    lesion_valid
                )

                total_loss = (
                    cls_loss
                    + lesion_lambda
                    * aux_loss
                )

                bs = images.size(0)

                val_cls_sum += (
                    cls_loss.item() * bs
                )
                val_lesion_sum += (
                    aux_loss.item() * bs
                )
                val_total_sum += (
                    total_loss.item() * bs
                )

                val_correct += (
                    class_logits.argmax(1)
                    == labels
                ).sum().item()

                val_n += bs

        val_cls = (
            val_cls_sum / val_n
        )
        val_lesion = (
            val_lesion_sum / val_n
        )
        val_total = (
            val_total_sum / val_n
        )
        val_acc = (
            val_correct / val_n
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train cls {train_cls:.4f} | "
            f"train lesion {train_lesion:.4f} | "
            f"train acc {train_acc:.4f} | "
            f"val cls {val_cls:.4f} | "
            f"val lesion {val_lesion:.4f} | "
            f"val acc {val_acc:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_cls_loss": train_cls,
            "train_lesion_loss": train_lesion,
            "train_total_loss": train_total,
            "train_acc": train_acc,
            "val_cls_loss": val_cls,
            "val_lesion_loss": val_lesion,
            "val_total_loss": val_total,
            "val_acc": val_acc
        })

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_cls_loss": val_cls,
                "val_total_loss": val_total,
                "val_acc": val_acc
            },
            last_checkpoint_path
        )

        if (
            val_cls
            < best_val_cls_loss
        ):
            best_val_cls_loss = val_cls

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_cls_loss": val_cls,
                    "val_total_loss": val_total,
                    "val_acc": val_acc
                },
                best_model_path
            )

            print(
                "  → Best model 저장"
            )

    history = pd.DataFrame(
        history
    )

    history.to_csv(
        checkpoint_dir
        / "history.csv",
        index=False
    )

    best_checkpoint = torch.load(
        best_model_path,
        map_location=device
    )

    model.load_state_dict(
        best_checkpoint[
            "model_state_dict"
        ]
    )

    return model, history


# %% [markdown]
# ⭐⭐lesion path를 cohort에 붙이기⭐⭐

# %%
def attach_lesion_paths(
    df,
    lesion_root=LESION_UNION_ROOT
):
    df = df.copy()

    lesion_root = Path(
        lesion_root
    )

    df["lesion_mask_path"] = (
        df["dicom_id"]
        .astype(str)
        .map(
            lambda x:
            str(
                lesion_root
                / f"{x}.npy"
            )
            if (
                lesion_root
                / f"{x}.npy"
            ).exists()
            else np.nan
        )
    )

    return df


lesion_train_df = attach_lesion_paths(
    train_df
)

lesion_val_df = attach_lesion_paths(
    val_df
)

lesion_test_df = attach_lesion_paths(
    test_df
)

print(
    "train lesion-valid:",
    lesion_train_df[
        "lesion_mask_path"
    ].notna().sum()
)

print(
    "val lesion-valid:",
    lesion_val_df[
        "lesion_mask_path"
    ].notna().sum()
)

print(
    "test lesion-valid:",
    lesion_test_df[
        "lesion_mask_path"
    ].notna().sum()
)



# %%
LESION_EXTILS_DIR = (
    RESEARCH_ROOT
    / "checkpoints"
    / "lesion_extils_densenet121"
)

LESION_EXTILS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LESION_BEST_PATH = (
    LESION_EXTILS_DIR
    / "best_model.pt"
)

LESION_LAST_PATH = (
    LESION_EXTILS_DIR
    / "last_checkpoint.pt"
)

LESION_HISTORY_PATH = (
    LESION_EXTILS_DIR
    / "history.pt"
)

LESION_EPOCHS = 10
LESION_LAMBDA = 0.1


# %% [markdown]
# ⭐⭐Lesion-Supervised loaders / 학습 / 평가⭐⭐

# %%
lesion_train_dataset = LesionSupervisedDataset(
    lesion_train_df,
    image_transform=image_transform
)

lesion_val_dataset = LesionSupervisedDataset(
    lesion_val_df,
    image_transform=image_transform
)

lesion_test_dataset = LesionSupervisedDataset(
    lesion_test_df,
    image_transform=image_transform
)

lesion_train_loader = DataLoader(
    lesion_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

lesion_val_loader = DataLoader(
    lesion_val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

lesion_test_loader = DataLoader(
    lesion_test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

lesion_model = DenseNet121Lesion(
    num_classes=NUM_CLASSES
).to(device)

lesion_optimizer = torch.optim.AdamW(
    lesion_model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

LESION_CKPT_DIR = LESION_EXTILS_DIR

lesion_model, lesion_history = train_lesion_model(
    model=lesion_model,
    train_loader=lesion_train_loader,
    val_loader=lesion_val_loader,
    optimizer=lesion_optimizer,
    device=device,
    checkpoint_dir=LESION_CKPT_DIR,
    epochs=EPOCHS,
    lesion_lambda=LESION_LAMBDA
)

lesion_val_logits, lesion_val_labels_t, lesion_val_ids = collect_logits_labels(
    lesion_model,
    lesion_val_loader,
    device
)

lesion_test_logits, lesion_test_labels_t, lesion_test_ids = collect_logits_labels(
    lesion_model,
    lesion_test_loader,
    device
)

lesion_val_labels = (
    lesion_val_labels_t.numpy()
)

lesion_test_labels = (
    lesion_test_labels_t.numpy()
)

lesion_val_probs_raw = torch.softmax(
    lesion_val_logits,
    dim=1
).numpy()

lesion_test_probs_raw = torch.softmax(
    lesion_test_logits,
    dim=1
).numpy()

lesion_raw_metrics = evaluate_probs(
    lesion_test_labels,
    lesion_test_probs_raw,
    n_bins=N_BINS
)

print_metric_report(
    "Lesion-Supervised - Uncalibrated",
    lesion_raw_metrics
)

lesion_temperature = fit_temperature(
    lesion_val_logits,
    lesion_val_labels_t
)

lesion_val_probs_cal = apply_temperature(
    lesion_temperature,
    lesion_val_logits
)

lesion_test_probs_cal = apply_temperature(
    lesion_temperature,
    lesion_test_logits
)

lesion_cal_metrics = evaluate_probs(
    lesion_test_labels,
    lesion_test_probs_cal,
    n_bins=N_BINS
)

print_metric_report(
    "Lesion-Supervised - Temperature Scaled",
    lesion_cal_metrics
)


# ============================================================================
# 10. 5-arm 최종 결과 표
# ============================================================================

# %% [markdown]
# ⭐⭐Uncalibrated final comparison⭐⭐

# %%
raw_metric_dict = {
    "Baseline": baseline_raw_metrics,
    "Gaze-Original": gaze_original_raw_metrics,
    "Shuffled-Gaze": shuffled_raw_metrics,
    "Random-Gaze": random_raw_metrics,
    "Lesion-Supervised": lesion_raw_metrics
}

raw_results_df = pd.DataFrame(
    raw_metric_dict
).T

print(
    raw_results_df[
        [
            "Accuracy",
            "F1_macro",
            "AUROC_macro",
            "ECE",
            "Brier",
            "NLL"
        ]
    ]
)

RESULT_DIR = RESEARCH_ROOT / "results"

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

raw_results_df.to_csv(
    RESULT_DIR
    / "five_arm_uncalibrated_metrics.csv"
)


# %% [markdown]
# ⭐⭐Temperature-scaled final comparison⭐⭐

# %%
cal_metric_dict = {
    "Baseline": baseline_cal_metrics,
    "Gaze-Original": gaze_original_cal_metrics,
    "Shuffled-Gaze": shuffled_cal_metrics,
    "Random-Gaze": random_cal_metrics,
    "Lesion-Supervised": lesion_cal_metrics
}

cal_results_df = pd.DataFrame(
    cal_metric_dict
).T

cal_results_df.to_csv(
    RESULT_DIR
    / "five_arm_temperature_scaled_metrics.csv"
)

print(
    cal_results_df[
        [
            "AUROC_macro",
            "ECE",
            "Brier",
            "NLL"
        ]
    ]
)



# ============================================================================
# 10-1. 2026-08-19 arm 비교용 bootstrap dataframe 분석
# ============================================================================

# %% [markdown]
# ⭐⭐calc_ece / calc_brier / calc_nll / calc_metrics_from_df⭐⭐
# 당시 test prediction dataframe column:
# label, prob_Normal, prob_CHF, prob_Pneumonia
# bootstrap: 1,000회, class별 replacement resampling, 같은 index를 모든 arm에 적용.

# %%
PROB_COLS = [
    "prob_Normal",
    "prob_CHF",
    "prob_Pneumonia"
]


def calc_ece(
    labels,
    probs,
    n_bins=15
):
    labels = np.asarray(
        labels
    )

    probs = np.asarray(
        probs
    )

    confidence = probs.max(
        axis=1
    )

    pred = probs.argmax(
        axis=1
    )

    correct = (
        pred == labels
    ).astype(float)

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1
    )

    ece = 0.0

    for i in range(
        n_bins
    ):
        lo = edges[i]
        hi = edges[i + 1]

        if i == 0:
            mask = (
                (confidence >= lo)
                & (confidence <= hi)
            )
        else:
            mask = (
                (confidence > lo)
                & (confidence <= hi)
            )

        if mask.sum() == 0:
            continue

        ece += (
            mask.mean()
            * abs(
                correct[mask].mean()
                - confidence[mask].mean()
            )
        )

    return float(
        ece
    )


def calc_brier(
    labels,
    probs
):
    labels = np.asarray(
        labels
    )

    probs = np.asarray(
        probs
    )

    onehot = np.eye(
        probs.shape[1]
    )[labels]

    return float(
        np.mean(
            np.sum(
                (
                    probs
                    - onehot
                ) ** 2,
                axis=1
            )
        )
    )


def calc_nll(
    labels,
    probs
):
    labels = np.asarray(
        labels
    )

    probs = np.asarray(
        probs
    )

    probs = np.clip(
        probs,
        1e-12,
        1.0
    )

    true_prob = probs[
        np.arange(
            len(labels)
        ),
        labels
    ]

    return float(
        -np.mean(
            np.log(
                true_prob
            )
        )
    )


def calc_metrics_from_df(
    df
):
    labels = (
        df["label"]
        .to_numpy(
            dtype=int
        )
    )

    probs = (
        df[PROB_COLS]
        .to_numpy(
            dtype=float
        )
    )

    pred = probs.argmax(
        axis=1
    )

    pneumonia_binary = (
        labels == 2
    ).astype(int)

    return {
        "Accuracy": accuracy_score(
            labels,
            pred
        ),
        "Macro_F1": f1_score(
            labels,
            pred,
            average="macro"
        ),
        "Macro_AUROC": roc_auc_score(
            labels,
            probs,
            multi_class="ovr",
            average="macro"
        ),
        "ECE": calc_ece(
            labels,
            probs,
            n_bins=15
        ),
        "Brier": calc_brier(
            labels,
            probs
        ),
        "NLL": calc_nll(
            labels,
            probs
        ),
        "Pneumonia_AUROC": roc_auc_score(
            pneumonia_binary,
            probs[:, 2]
        )
    }


def bootstrap_df(
    prediction_dfs,
    n_bootstrap=1000,
    seed=42
):
    """
    prediction_dfs:
        dict[str, pd.DataFrame]
        모든 arm의 row order가 동일한 test sample 순서여야 함.

    paired + stratified:
        class별로 같은 bootstrap index를 뽑고,
        그 index를 모든 arm에 동일 적용.
    """

    arm_names = list(
        prediction_dfs.keys()
    )

    reference_df = prediction_dfs[
        arm_names[0]
    ].reset_index(
        drop=True
    )

    labels = reference_df[
        "label"
    ].to_numpy(
        dtype=int
    )

    rng = np.random.default_rng(
        seed
    )

    rows = []

    for bootstrap_id in range(
        n_bootstrap
    ):
        sampled_parts = []

        for cls in np.unique(
            labels
        ):
            cls_idx = np.where(
                labels == cls
            )[0]

            sampled = rng.choice(
                cls_idx,
                size=len(
                    cls_idx
                ),
                replace=True
            )

            sampled_parts.append(
                sampled
            )

        sampled_idx = np.concatenate(
            sampled_parts
        )

        for arm in arm_names:
            sampled_df = (
                prediction_dfs[
                    arm
                ]
                .reset_index(
                    drop=True
                )
                .iloc[
                    sampled_idx
                ]
                .reset_index(
                    drop=True
                )
            )

            metrics = calc_metrics_from_df(
                sampled_df
            )

            rows.append({
                "bootstrap_id": bootstrap_id,
                "Arm": arm,
                **metrics
            })

    return pd.DataFrame(
        rows
    )


def bootstrap_ci(
    bootstrap_df_,
    metric
):
    rows = []

    for arm, group in bootstrap_df_.groupby(
        "Arm"
    ):
        values = group[
            metric
        ].to_numpy()

        rows.append({
            "Arm": arm,
            "Metric": metric,
            "CI_Low": np.percentile(
                values,
                2.5
            ),
            "CI_High": np.percentile(
                values,
                97.5
            )
        })

    return pd.DataFrame(
        rows
    )


def difference_ci(
    bootstrap_df_,
    arm_a,
    arm_b,
    metric
):
    a = (
        bootstrap_df_[
            bootstrap_df_[
                "Arm"
            ] == arm_a
        ]
        .sort_values(
            "bootstrap_id"
        )[
            metric
        ]
        .to_numpy()
    )

    b = (
        bootstrap_df_[
            bootstrap_df_[
                "Arm"
            ] == arm_b
        ]
        .sort_values(
            "bootstrap_id"
        )[
            metric
        ]
        .to_numpy()
    )

    diff = (
        a
        - b
    )

    return {
        "Arm_A": arm_a,
        "Arm_B": arm_b,
        "Metric": metric,
        "Difference_Mean": diff.mean(),
        "CI_Low": np.percentile(
            diff,
            2.5
        ),
        "CI_High": np.percentile(
            diff,
            97.5
        )
    }


def ci_table(
    point_metric_df,
    bootstrap_df_
):
    rows = []

    for arm in point_metric_df.index:
        for metric in [
            "Accuracy",
            "Macro_F1",
            "Macro_AUROC",
            "ECE",
            "Brier",
            "NLL"
        ]:
            boot = bootstrap_df_[
                bootstrap_df_[
                    "Arm"
                ] == arm
            ][metric]

            lo, hi = np.percentile(
                boot,
                [
                    2.5,
                    97.5
                ]
            )

            estimate = point_metric_df.loc[
                arm,
                metric
            ]

            rows.append({
                "Arm": arm,
                "Metric": metric,
                "Estimate": estimate,
                "CI_Low": lo,
                "CI_High": hi,
                "Estimate_95CI": (
                    f"{estimate:.4f} "
                    f"[{lo:.4f}, {hi:.4f}]"
                )
            })

    return pd.DataFrame(
        rows
    )


# %% [markdown]
# ⭐⭐prediction dataframe 만들기⭐⭐

# %%
def make_prediction_df(
    labels,
    probs
):
    return pd.DataFrame({
        "label": np.asarray(
            labels,
            dtype=int
        ),
        "prob_Normal": probs[
            :,
            0
        ],
        "prob_CHF": probs[
            :,
            1
        ],
        "prob_Pneumonia": probs[
            :,
            2
        ]
    })


prediction_dfs_raw = {
    "Baseline": make_prediction_df(
        baseline_test_labels,
        baseline_test_probs_raw
    ),
    "Gaze-Original": make_prediction_df(
        gaze_original_test_labels,
        gaze_original_test_probs_raw
    ),
    "Shuffled-Gaze": make_prediction_df(
        shuffled_test_labels,
        shuffled_test_probs_raw
    ),
    "Random-Gaze": make_prediction_df(
        random_test_labels,
        random_test_probs_raw
    ),
    "Lesion-Supervised": make_prediction_df(
        lesion_test_labels,
        lesion_test_probs_raw
    )
}


# %% [markdown]
# ⭐⭐1,000 paired stratified bootstrap⭐⭐

# %%
bootstrap_df_result = bootstrap_df(
    prediction_dfs_raw,
    n_bootstrap=1000,
    seed=42
)

bootstrap_df_result.to_csv(
    RESULT_DIR
    / "five_arm_bootstrap_1000.csv",
    index=False
)


# %%
auroc_difference = difference_ci(
    bootstrap_df_result,
    arm_a="Gaze-Original",
    arm_b="Baseline",
    metric="Macro_AUROC"
)

ece_difference = difference_ci(
    bootstrap_df_result,
    arm_a="Gaze-Original",
    arm_b="Shuffled-Gaze",
    metric="ECE"
)

print(
    "Original - Baseline Macro AUROC"
)
print(
    auroc_difference
)

print(
    "Original - Shuffled ECE"
)
print(
    ece_difference
)



# ============================================================================
# 11. Risk-Coverage / AURC / E-AURC
# ============================================================================

# %% [markdown]
# ⭐⭐7번 AURC 계산⭐⭐
# 최종 실험에서 selective prediction은 temperature-calibrated probabilities 사용.

# %%
def risk_coverage_curve(
    labels,
    probs
):
    labels = np.asarray(
        labels
    )

    probs = np.asarray(
        probs
    )

    confidence = probs.max(
        axis=1
    )

    pred = probs.argmax(
        axis=1
    )

    error = (
        pred != labels
    ).astype(
        np.float64
    )

    order = np.argsort(
        -confidence
    )

    sorted_error = error[
        order
    ]

    cumulative_error = np.cumsum(
        sorted_error
    )

    n = len(
        labels
    )

    retained = np.arange(
        1,
        n + 1
    )

    coverage = (
        retained
        / n
    )

    risk = (
        cumulative_error
        / retained
    )

    aurc = float(
        risk.mean()
    )

    return {
        "coverage": coverage,
        "risk": risk,
        "aurc": aurc,
        "confidence": confidence,
        "error": error,
        "order": order
    }


def ideal_aurc(
    labels,
    probs
):
    labels = np.asarray(
        labels
    )

    probs = np.asarray(
        probs
    )

    pred = probs.argmax(
        axis=1
    )

    error = (
        pred != labels
    ).astype(
        np.float64
    )

    # correct samples first, errors last
    ideal_sorted_error = np.sort(
        error
    )

    retained = np.arange(
        1,
        len(error) + 1
    )

    ideal_risk = np.cumsum(
        ideal_sorted_error
    ) / retained

    return float(
        ideal_risk.mean()
    )


def selective_metrics(
    labels,
    probs
):
    rc = risk_coverage_curve(
        labels,
        probs
    )

    ideal = ideal_aurc(
        labels,
        probs
    )

    eaurc = (
        rc["aurc"]
        - ideal
    )

    uncertainty = (
        1.0
        - rc["confidence"]
    )

    error = rc[
        "error"
    ]

    if len(
        np.unique(error)
    ) >= 2:
        error_detection_auroc = roc_auc_score(
            error,
            uncertainty
        )
    else:
        error_detection_auroc = np.nan

    return {
        "AURC": rc["aurc"],
        "Ideal_AURC": ideal,
        "E_AURC": float(eaurc),
        "Error_Detection_AUROC": float(
            error_detection_auroc
        ),
        **rc
    }


# %% [markdown]
# ⭐⭐모든 arm validation/test AURC⭐⭐

# %%
arm_prob_store = {
    "Baseline": {
        "val_labels": baseline_val_labels,
        "val_probs": baseline_val_probs_cal,
        "test_labels": baseline_test_labels,
        "test_probs": baseline_test_probs_cal
    },
    "Gaze-Original": {
        "val_labels": gaze_original_val_labels,
        "val_probs": gaze_original_val_probs_cal,
        "test_labels": gaze_original_test_labels,
        "test_probs": gaze_original_test_probs_cal
    },
    "Shuffled-Gaze": {
        "val_labels": shuffled_val_labels,
        "val_probs": shuffled_val_probs_cal,
        "test_labels": shuffled_test_labels,
        "test_probs": shuffled_test_probs_cal
    },
    "Random-Gaze": {
        "val_labels": random_val_labels,
        "val_probs": random_val_probs_cal,
        "test_labels": random_test_labels,
        "test_probs": random_test_probs_cal
    },
    "Lesion-Supervised": {
        "val_labels": lesion_val_labels,
        "val_probs": lesion_val_probs_cal,
        "test_labels": lesion_test_labels,
        "test_probs": lesion_test_probs_cal
    }
}

selective_summary_rows = []

selective_detail = {}

for arm, data in arm_prob_store.items():
    val_sel = selective_metrics(
        data["val_labels"],
        data["val_probs"]
    )

    test_sel = selective_metrics(
        data["test_labels"],
        data["test_probs"]
    )

    selective_detail[arm] = {
        "val": val_sel,
        "test": test_sel
    }

    selective_summary_rows.append({
        "Arm": arm,
        "Val_AURC": val_sel["AURC"],
        "Val_E_AURC": val_sel["E_AURC"],
        "Val_Error_AUROC": val_sel[
            "Error_Detection_AUROC"
        ],
        "Test_AURC": test_sel["AURC"],
        "Test_E_AURC": test_sel["E_AURC"],
        "Test_Error_AUROC": test_sel[
            "Error_Detection_AUROC"
        ]
    })

selective_summary_df = pd.DataFrame(
    selective_summary_rows
)

print(
    selective_summary_df
)

selective_summary_df.to_csv(
    RESULT_DIR
    / "selective_prediction_summary.csv",
    index=False
)


# ============================================================================
# 12. Deployable candidate selection
# ============================================================================

# %% [markdown]
# ⭐⭐Deployable Decision-support Candidate⭐⭐
# Shuffled/Random은 실험 control이므로 deployment 후보에서 제외.
# Baseline / Gaze-Original / Lesion-Supervised 중 validation AURC 최소 arm 선택.

# %%
DEPLOYABLE_ARMS = [
    "Baseline",
    "Gaze-Original",
    "Lesion-Supervised"
]

deployable_df = (
    selective_summary_df[
        selective_summary_df[
            "Arm"
        ].isin(
            DEPLOYABLE_ARMS
        )
    ]
    .sort_values(
        "Val_AURC"
    )
    .reset_index(
        drop=True
    )
)

deployable_df.insert(
    0,
    "Rank",
    np.arange(
        1,
        len(deployable_df) + 1
    )
)

selected_arm = (
    deployable_df
    .iloc[0]["Arm"]
)

print(
    "=" * 42
)
print(
    "Deployable Decision-support Candidate"
)
print(
    "=" * 42
)
print(
    "Selected Arm    :",
    selected_arm
)
print(
    "Validation AURC :",
    f"{deployable_df.iloc[0]['Val_AURC']:.4f}"
)
print(
    "Test AURC       :",
    f"{deployable_df.iloc[0]['Test_AURC']:.4f}"
)

print(
    deployable_df[
        [
            "Rank",
            "Arm",
            "Val_AURC",
            "Test_AURC"
        ]
    ]
)


# ============================================================================
# 13. Validation-derived operating point / referral table
# ============================================================================

# %% [markdown]
# ⭐⭐confidence threshold⭐⭐
# validation에서 target coverage에 해당하는 threshold를 결정하고
# 같은 threshold를 test에 고정 적용.
#
# RECONSTRUCTION NOTE:
# 본 복구본은 이 구간의 np.quantile() 구현을 그대로 보존한다.
# 별도 대화 기록에는 ceil(n * coverage) -> confidence 내림차순 정렬 ->
# 해당 순위값 선택 방식도 존재한다.
# 현재 확보된 자료만으로 당시 최종 실행에 어느 구현이 사용됐는지는 확정하지 않는다.
# 따라서 계산 코드는 변경하지 않고, 구현 이력의 불확실성만 명시한다.

# %%
def threshold_for_coverage(
    probs,
    target_coverage
):
    # 2026-08-19 8-4 수정:
    # 100% coverage는 validation minimum confidence가 아니라 -inf로 처리해야
    # test에서도 반드시 100%가 유지됨.
    confidence = np.max(
        probs,
        axis=1
    )

    if target_coverage >= 1.0:
        return -np.inf

    threshold = np.quantile(
        confidence,
        1.0 - target_coverage
    )

    return float(
        threshold
    )


def evaluate_operating_point(
    labels,
    probs,
    confidence_threshold
):
    labels = np.asarray(
        labels
    )

    probs = np.asarray(
        probs
    )

    confidence = probs.max(
        axis=1
    )

    pred = probs.argmax(
        axis=1
    )

    error = (
        pred != labels
    )

    retained = (
        confidence
        >= confidence_threshold
    )

    deferred = ~retained

    actual_coverage = (
        retained.mean()
    )

    defer_rate = (
        deferred.mean()
    )

    if retained.sum() > 0:
        selective_accuracy = (
            pred[retained]
            == labels[retained]
        ).mean()
    else:
        selective_accuracy = np.nan

    errors_total = int(
        error.sum()
    )

    errors_captured = int(
        error[
            deferred
        ].sum()
    )

    if errors_total > 0:
        error_capture_rate = (
            errors_captured
            / errors_total
        )
    else:
        error_capture_rate = np.nan

    if deferred.sum() > 0:
        referral_error_precision = (
            error[
                deferred
            ].mean()
        )
    else:
        referral_error_precision = np.nan

    return {
        "Confidence_Threshold": float(
            confidence_threshold
        ),
        "Actual_Coverage": float(
            actual_coverage
        ),
        "Defer_Rate": float(
            defer_rate
        ),
        "AI_Handled": int(
            retained.sum()
        ),
        "Deferred": int(
            deferred.sum()
        ),
        "Selective_Accuracy": float(
            selective_accuracy
        ),
        "Errors_Total": errors_total,
        "Errors_Captured_By_Defer": errors_captured,
        "Error_Capture_Rate": float(
            error_capture_rate
        ),
        "Referral_Error_Precision": float(
            referral_error_precision
        )
    }


# %%
target_coverages = [
    1.0,
    0.9,
    0.8,
    0.7,
    0.6,
    0.5
]

operating_rows = []

for arm, data in arm_prob_store.items():
    for target_coverage in target_coverages:
        threshold = threshold_for_coverage(
            data["val_probs"],
            target_coverage
        )

        result = evaluate_operating_point(
            data["test_labels"],
            data["test_probs"],
            threshold
        )

        operating_rows.append({
            "Arm": arm,
            "Target_Coverage": target_coverage,
            **result
        })

operating_df = pd.DataFrame(
    operating_rows
)

print(
    operating_df
)

operating_df.to_csv(
    RESULT_DIR
    / "validation_derived_operating_points.csv",
    index=False
)


# %% [markdown]
# ⭐⭐대표 operating point: target coverage = 0.70⭐⭐
# IWAIT 2027 초록 제출 당시 최종 결과:
# test actual coverage = 0.75
# defer rate = 0.25
# selective accuracy = 0.7724
# total errors = 54
# captured errors = 26
# error capture rate = 0.4815

# %%
SELECTED_TARGET_COVERAGE = 0.70

selected_data = arm_prob_store[
    selected_arm
]

selected_threshold = threshold_for_coverage(
    selected_data["val_probs"],
    SELECTED_TARGET_COVERAGE
)

selected_operating_point = evaluate_operating_point(
    selected_data["test_labels"],
    selected_data["test_probs"],
    selected_threshold
)

print(
    selected_operating_point
)


# ============================================================================
# 14. Error enrichment
# ============================================================================

# %% [markdown]
# ⭐⭐random referral 대비 error enrichment⭐⭐
# random referral에서 기대되는 error capture rate = defer rate.
# enrichment = observed error capture rate / defer rate.

# %%
def error_enrichment(
    error_capture_rate,
    defer_rate
):
    if (
        defer_rate <= 0
        or np.isnan(
            error_capture_rate
        )
    ):
        return np.nan

    return (
        error_capture_rate
        / defer_rate
    )


selected_enrichment = error_enrichment(
    selected_operating_point[
        "Error_Capture_Rate"
    ],
    selected_operating_point[
        "Defer_Rate"
    ]
)

print(
    "Error enrichment:",
    selected_enrichment
)


# ============================================================================
# 15. Operating-point bootstrap (2000)
# ============================================================================

# %% [markdown]
# ⭐⭐8-4. fixed validation threshold + test bootstrap 2000⭐⭐
# validation-derived threshold는 bootstrap마다 다시 계산하지 않고 고정.
# test set만 replacement bootstrap.
# 95% percentile CI = 2.5 / 97.5 percentile.

# %%
def bootstrap_operating_point(
    labels,
    probs,
    confidence_threshold,
    n_bootstrap=2000,
    seed=42
):
    labels = np.asarray(
        labels
    )

    probs = np.asarray(
        probs
    )

    n = len(
        labels
    )

    rng = np.random.default_rng(
        seed
    )

    rows = []

    for b in range(
        n_bootstrap
    ):
        idx = rng.integers(
            0,
            n,
            size=n
        )

        y_b = labels[
            idx
        ]

        p_b = probs[
            idx
        ]

        op = evaluate_operating_point(
            y_b,
            p_b,
            confidence_threshold
        )

        enrichment = error_enrichment(
            op[
                "Error_Capture_Rate"
            ],
            op[
                "Defer_Rate"
            ]
        )

        rows.append({
            **op,
            "Error_Enrichment": enrichment
        })

    boot_df = pd.DataFrame(
        rows
    )

    def percentile_ci(
        column
    ):
        values = (
            boot_df[column]
            .dropna()
            .values
        )

        return (
            float(
                np.percentile(
                    values,
                    2.5
                )
            ),
            float(
                np.percentile(
                    values,
                    97.5
                )
            )
        )

    ci = {
        "Actual_Coverage_CI": percentile_ci(
            "Actual_Coverage"
        ),
        "Defer_Rate_CI": percentile_ci(
            "Defer_Rate"
        ),
        "Selective_Accuracy_CI": percentile_ci(
            "Selective_Accuracy"
        ),
        "Error_Capture_Rate_CI": percentile_ci(
            "Error_Capture_Rate"
        ),
        "Referral_Error_Precision_CI": percentile_ci(
            "Referral_Error_Precision"
        ),
        "Error_Enrichment_CI": percentile_ci(
            "Error_Enrichment"
        )
    }

    return boot_df, ci


operating_bootstrap_df, operating_bootstrap_ci = bootstrap_operating_point(
    selected_data["test_labels"],
    selected_data["test_probs"],
    selected_threshold,
    n_bootstrap=2000,
    seed=SEED
)

print(
    operating_bootstrap_ci
)

operating_bootstrap_df.to_csv(
    RESULT_DIR
    / "baseline_operating_point_bootstrap_2000.csv",
    index=False
)


# ============================================================================
# 16. Error-detection AUROC bootstrap
# ============================================================================

# %% [markdown]
# ⭐⭐전체 test error-detection AUROC + 95% CI⭐⭐

# %%
def error_detection_auroc(
    labels,
    probs
):
    labels = np.asarray(
        labels
    )

    probs = np.asarray(
        probs
    )

    pred = probs.argmax(
        axis=1
    )

    error = (
        pred != labels
    ).astype(int)

    uncertainty = (
        1.0
        - probs.max(
            axis=1
        )
    )

    if (
        len(
            np.unique(error)
        )
        < 2
    ):
        return np.nan

    return float(
        roc_auc_score(
            error,
            uncertainty
        )
    )


def bootstrap_error_detection_auroc(
    labels,
    probs,
    n_bootstrap=2000,
    seed=42
):
    labels = np.asarray(
        labels
    )

    probs = np.asarray(
        probs
    )

    n = len(
        labels
    )

    rng = np.random.default_rng(
        seed
    )

    values = []

    for _ in range(
        n_bootstrap
    ):
        idx = rng.integers(
            0,
            n,
            size=n
        )

        value = error_detection_auroc(
            labels[idx],
            probs[idx]
        )

        if not np.isnan(
            value
        ):
            values.append(
                value
            )

    point = error_detection_auroc(
        labels,
        probs
    )

    lo, hi = np.percentile(
        values,
        [
            2.5,
            97.5
        ]
    )

    return (
        float(point),
        float(lo),
        float(hi)
    )


baseline_error_auc_ci = bootstrap_error_detection_auroc(
    selected_data["test_labels"],
    selected_data["test_probs"],
    n_bootstrap=2000,
    seed=SEED
)

print(
    "Baseline test error-detection AUROC:",
    baseline_error_auc_ci
)



# ============================================================================
# 16-1. Error-detection AUROC / AUPRC / AURC bootstrap
# ============================================================================

# %%
from sklearn.metrics import average_precision_score


def error_detection_metrics(
    labels,
    probs
):
    labels = np.asarray(
        labels
    )

    probs = np.asarray(
        probs
    )

    pred = probs.argmax(
        axis=1
    )

    error = (
        pred != labels
    ).astype(int)

    confidence = probs.max(
        axis=1
    )

    uncertainty = (
        1.0
        - confidence
    )

    error_auroc = roc_auc_score(
        error,
        uncertainty
    )

    error_auprc = average_precision_score(
        error,
        uncertainty
    )

    rc = risk_coverage_curve(
        labels,
        probs
    )

    return {
        "Error_AUROC": float(
            error_auroc
        ),
        "Error_AUPRC": float(
            error_auprc
        ),
        "AURC": float(
            rc["aurc"]
        )
    }


def bootstrap_error_metrics(
    labels,
    probs,
    n_bootstrap=2000,
    seed=42
):
    labels = np.asarray(
        labels
    )

    probs = np.asarray(
        probs
    )

    rng = np.random.default_rng(
        seed
    )

    rows = []
    n = len(
        labels
    )

    for b in range(
        n_bootstrap
    ):
        idx = rng.integers(
            0,
            n,
            size=n
        )

        y_b = labels[
            idx
        ]

        p_b = probs[
            idx
        ]

        pred_b = p_b.argmax(
            axis=1
        )

        error_b = (
            pred_b != y_b
        ).astype(int)

        if len(
            np.unique(
                error_b
            )
        ) < 2:
            continue

        m = error_detection_metrics(
            y_b,
            p_b
        )

        rows.append({
            "bootstrap_id": b,
            **m
        })

    bootstrap_metrics_df = pd.DataFrame(
        rows
    )

    point = error_detection_metrics(
        labels,
        probs
    )

    ci = {}

    for metric in [
        "Error_AUROC",
        "Error_AUPRC",
        "AURC"
    ]:
        lo, hi = np.percentile(
            bootstrap_metrics_df[
                metric
            ],
            [
                2.5,
                97.5
            ]
        )

        ci[metric] = {
            "Estimate": point[
                metric
            ],
            "CI_Low": float(
                lo
            ),
            "CI_High": float(
                hi
            )
        }

    return (
        bootstrap_metrics_df,
        ci
    )


error_metric_bootstrap_df, error_metric_ci = bootstrap_error_metrics(
    selected_data["test_labels"],
    selected_data["test_probs"],
    n_bootstrap=2000,
    seed=42
)

print(
    error_metric_ci
)

error_metric_bootstrap_df.to_csv(
    RESULT_DIR
    / "baseline_error_detection_bootstrap_2000.csv",
    index=False
)



# ============================================================================
# 17. Class-wise error detection
# ============================================================================

# %% [markdown]
# ⭐⭐Normal / CHF / Pneumonia별 error-detection AUROC⭐⭐

# %%
def class_error_detection(
    labels,
    probs,
    class_names=CLASS_NAMES,
    n_bootstrap=2000,
    seed=42
):
    labels = np.asarray(
        labels
    )

    probs = np.asarray(
        probs
    )

    rows = []

    for class_idx, class_name in enumerate(
        class_names
    ):
        mask = (
            labels == class_idx
        )

        y = labels[
            mask
        ]

        p = probs[
            mask
        ]

        point = error_detection_auroc(
            y,
            p
        )

        rng = np.random.default_rng(
            seed + class_idx
        )

        values = []

        n = len(
            y
        )

        for _ in range(
            n_bootstrap
        ):
            idx = rng.integers(
                0,
                n,
                size=n
            )

            value = error_detection_auroc(
                y[idx],
                p[idx]
            )

            if not np.isnan(
                value
            ):
                values.append(
                    value
                )

        lo, hi = np.percentile(
            values,
            [
                2.5,
                97.5
            ]
        )

        rows.append({
            "Class": class_name,
            "n": n,
            "Error_Detection_AUROC": point,
            "CI_Low": float(lo),
            "CI_High": float(hi)
        })

    return pd.DataFrame(
        rows
    )



# %%
def class_error_bootstrap(
    labels,
    probs,
    class_idx,
    n_bootstrap=2000,
    seed=42
):
    labels = np.asarray(
        labels
    )
    probs = np.asarray(
        probs
    )

    mask = (
        labels == class_idx
    )

    y = labels[
        mask
    ]
    p = probs[
        mask
    ]

    rng = np.random.default_rng(
        seed + class_idx
    )

    values = []
    n = len(
        y
    )

    for _ in range(
        n_bootstrap
    ):
        idx = rng.integers(
            0,
            n,
            size=n
        )

        value = error_detection_auroc(
            y[idx],
            p[idx]
        )

        if not np.isnan(
            value
        ):
            values.append(
                value
            )

    return np.asarray(
        values
    )


def class_error_detection_ci(
    labels,
    probs,
    class_names=CLASS_NAMES,
    n_bootstrap=2000,
    seed=42
):
    rows = []

    for class_idx, class_name in enumerate(
        class_names
    ):
        mask = (
            np.asarray(labels)
            == class_idx
        )

        point = error_detection_auroc(
            np.asarray(labels)[mask],
            np.asarray(probs)[mask]
        )

        boot = class_error_bootstrap(
            labels,
            probs,
            class_idx=class_idx,
            n_bootstrap=n_bootstrap,
            seed=seed
        )

        lo, hi = np.percentile(
            boot,
            [
                2.5,
                97.5
            ]
        )

        rows.append({
            "Class": class_name,
            "n": int(
                mask.sum()
            ),
            "Error_Detection_AUROC": float(
                point
            ),
            "CI_Low": float(
                lo
            ),
            "CI_High": float(
                hi
            )
        })

    return pd.DataFrame(
        rows
    )



classwise_error_df = class_error_detection(
    selected_data["test_labels"],
    selected_data["test_probs"],
    class_names=CLASS_NAMES,
    n_bootstrap=2000,
    seed=SEED
)

print(
    classwise_error_df
)

classwise_error_df.to_csv(
    RESULT_DIR
    / "baseline_classwise_error_detection.csv",
    index=False
)


# ============================================================================
# 18. Paired stratified bootstrap 1000
# ============================================================================

# %% [markdown]
# ⭐⭐arm 비교용 paired stratified bootstrap⭐⭐
# 같은 bootstrap sample을 두 arm에 동시에 적용.
# 각 class를 따로 replacement resample해서 class distribution 유지.
# 이 CI는 test-sample uncertainty이며 training-seed variability를 반영하지 않음.

# %%
def stratified_bootstrap_indices(
    labels,
    rng
):
    labels = np.asarray(
        labels
    )

    sampled_parts = []

    for cls in np.unique(
        labels
    ):
        cls_idx = np.where(
            labels == cls
        )[0]

        sampled_cls = rng.choice(
            cls_idx,
            size=len(cls_idx),
            replace=True
        )

        sampled_parts.append(
            sampled_cls
        )

    return np.concatenate(
        sampled_parts
    )


def macro_auroc_from_probs(
    labels,
    probs
):
    return float(
        roc_auc_score(
            labels,
            probs,
            multi_class="ovr",
            average="macro"
        )
    )


def paired_stratified_bootstrap_difference(
    labels,
    probs_a,
    probs_b,
    metric="macro_auroc",
    n_bootstrap=1000,
    seed=42,
    n_bins=15
):
    labels = np.asarray(
        labels
    )

    probs_a = np.asarray(
        probs_a
    )

    probs_b = np.asarray(
        probs_b
    )

    rng = np.random.default_rng(
        seed
    )

    differences = []

    for _ in range(
        n_bootstrap
    ):
        idx = stratified_bootstrap_indices(
            labels,
            rng
        )

        y = labels[
            idx
        ]

        pa = probs_a[
            idx
        ]

        pb = probs_b[
            idx
        ]

        if metric == "macro_auroc":
            a = macro_auroc_from_probs(
                y,
                pa
            )

            b = macro_auroc_from_probs(
                y,
                pb
            )

        elif metric == "ece":
            a = expected_calibration_error(
                pa,
                y,
                n_bins=n_bins
            )

            b = expected_calibration_error(
                pb,
                y,
                n_bins=n_bins
            )

        else:
            raise ValueError(
                f"Unknown metric: {metric}"
            )

        differences.append(
            a - b
        )

    differences = np.asarray(
        differences
    )

    point_a = (
        macro_auroc_from_probs(
            labels,
            probs_a
        )
        if metric == "macro_auroc"
        else expected_calibration_error(
            probs_a,
            labels,
            n_bins=n_bins
        )
    )

    point_b = (
        macro_auroc_from_probs(
            labels,
            probs_b
        )
        if metric == "macro_auroc"
        else expected_calibration_error(
            probs_b,
            labels,
            n_bins=n_bins
        )
    )

    point_diff = (
        point_a
        - point_b
    )

    ci_low, ci_high = np.percentile(
        differences,
        [
            2.5,
            97.5
        ]
    )

    return {
        "point_difference": float(
            point_diff
        ),
        "ci_low": float(
            ci_low
        ),
        "ci_high": float(
            ci_high
        ),
        "bootstrap_differences": differences
    }


# %% [markdown]
# ⭐⭐Gaze-Original - Baseline Macro AUROC⭐⭐

# %%
bootstrap_original_minus_baseline_auc = paired_stratified_bootstrap_difference(
    baseline_test_labels,
    gaze_original_test_probs_raw,
    baseline_test_probs_raw,
    metric="macro_auroc",
    n_bootstrap=1000,
    seed=SEED
)

print(
    "Gaze-Original - Baseline Macro AUROC:",
    bootstrap_original_minus_baseline_auc[
        "point_difference"
    ],
    (
        bootstrap_original_minus_baseline_auc[
            "ci_low"
        ],
        bootstrap_original_minus_baseline_auc[
            "ci_high"
        ]
    )
)


# %% [markdown]
# ⭐⭐Gaze-Original - Shuffled-Gaze ECE⭐⭐

# %%
bootstrap_original_minus_shuffled_ece = paired_stratified_bootstrap_difference(
    baseline_test_labels,
    gaze_original_test_probs_raw,
    shuffled_test_probs_raw,
    metric="ece",
    n_bootstrap=1000,
    seed=SEED,
    n_bins=N_BINS
)

print(
    "Gaze-Original - Shuffled-Gaze ECE:",
    bootstrap_original_minus_shuffled_ece[
        "point_difference"
    ],
    (
        bootstrap_original_minus_shuffled_ece[
            "ci_low"
        ],
        bootstrap_original_minus_shuffled_ece[
            "ci_high"
        ]
    )
)


# %% [markdown]
# ⭐⭐Original vs 모든 arm Macro AUROC 차이 확인⭐⭐

# %%
bootstrap_auc_comparisons = {}

for comparison_name, other_probs in {
    "Baseline": baseline_test_probs_raw,
    "Shuffled-Gaze": shuffled_test_probs_raw,
    "Random-Gaze": random_test_probs_raw,
    "Lesion-Supervised": lesion_test_probs_raw
}.items():
    result = paired_stratified_bootstrap_difference(
        baseline_test_labels,
        gaze_original_test_probs_raw,
        other_probs,
        metric="macro_auroc",
        n_bootstrap=1000,
        seed=SEED
    )

    bootstrap_auc_comparisons[
        comparison_name
    ] = result

    print(
        "Original -",
        comparison_name,
        ":",
        f"{result['point_difference']:.4f}",
        (
            f"[{result['ci_low']:.4f}, "
            f"{result['ci_high']:.4f}]"
        )
    )


# ============================================================================
# 19. Figure: Risk-Coverage + Error Capture
# ============================================================================

# %% [markdown]
# ⭐⭐risk-coverage / error-capture figure⭐⭐

# %%
def error_capture_curve(
    labels,
    probs
):
    labels = np.asarray(
        labels
    )

    probs = np.asarray(
        probs
    )

    confidence = probs.max(
        axis=1
    )

    pred = probs.argmax(
        axis=1
    )

    error = (
        pred != labels
    ).astype(int)

    # low confidence first = referral priority
    order = np.argsort(
        confidence
    )

    sorted_error = error[
        order
    ]

    n = len(
        labels
    )

    referral_count = np.arange(
        0,
        n + 1
    )

    referral_rate = (
        referral_count
        / n
    )

    total_errors = (
        error.sum()
    )

    captured = np.concatenate(
        [
            [0],
            np.cumsum(
                sorted_error
            )
        ]
    )

    if total_errors > 0:
        error_capture = (
            captured
            / total_errors
        )
    else:
        error_capture = np.zeros_like(
            captured,
            dtype=float
        )

    return (
        referral_rate,
        error_capture
    )


# %%
baseline_rc = selective_detail[
    "Baseline"
]["test"]

referral_rate, error_capture = error_capture_curve(
    selected_data["test_labels"],
    selected_data["test_probs"]
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)

axes[0].plot(
    baseline_rc["coverage"],
    baseline_rc["risk"],
    label=(
        f"Baseline "
        f"(AURC={baseline_rc['aurc']:.3f})"
    )
)

axes[0].set_xlabel(
    "Coverage"
)
axes[0].set_ylabel(
    "Risk"
)
axes[0].set_title(
    "Risk-Coverage Curve"
)
axes[0].legend()
axes[0].grid(
    alpha=0.3
)

axes[1].plot(
    referral_rate,
    error_capture,
    label="Uncertainty-based referral"
)

axes[1].plot(
    [0, 1],
    [0, 1],
    "--",
    label="Random referral"
)

axes[1].set_xlabel(
    "Referral rate"
)
axes[1].set_ylabel(
    "Error capture rate"
)
axes[1].set_title(
    "Error Capture"
)
axes[1].legend()
axes[1].grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    RESULT_DIR
    / "risk_coverage_error_capture.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================================
# 20. Figure: Class-wise error-detection AUROC
# ============================================================================

# %%
plt.figure(
    figsize=(7, 5)
)

x = np.arange(
    len(classwise_error_df)
)

y = classwise_error_df[
    "Error_Detection_AUROC"
].values

lower = (
    y
    - classwise_error_df[
        "CI_Low"
    ].values
)

upper = (
    classwise_error_df[
        "CI_High"
    ].values
    - y
)

plt.errorbar(
    x,
    y,
    yerr=[
        lower,
        upper
    ],
    fmt="o",
    capsize=5
)

plt.axhline(
    0.5,
    linestyle="--"
)

plt.xticks(
    x,
    classwise_error_df[
        "Class"
    ].values
)

plt.ylim(
    0.0,
    1.0
)

plt.ylabel(
    "Error-detection AUROC"
)

plt.title(
    "Class-wise Error Detection"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    RESULT_DIR
    / "classwise_error_detection_auroc.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================================
# 21. 초록 제출 전 최종 결과 요약용 3-panel summary bar chart
# ============================================================================

# %%
summary_plot_df = pd.DataFrame({
    "Arm": [
        "Baseline",
        "Original",
        "Shuffled",
        "Random",
        "Lesion"
    ],
    "Macro_AUROC": [
        raw_results_df.loc[
            "Baseline",
            "AUROC_macro"
        ],
        raw_results_df.loc[
            "Gaze-Original",
            "AUROC_macro"
        ],
        raw_results_df.loc[
            "Shuffled-Gaze",
            "AUROC_macro"
        ],
        raw_results_df.loc[
            "Random-Gaze",
            "AUROC_macro"
        ],
        raw_results_df.loc[
            "Lesion-Supervised",
            "AUROC_macro"
        ]
    ],
    "ECE": [
        raw_results_df.loc[
            "Baseline",
            "ECE"
        ],
        raw_results_df.loc[
            "Gaze-Original",
            "ECE"
        ],
        raw_results_df.loc[
            "Shuffled-Gaze",
            "ECE"
        ],
        raw_results_df.loc[
            "Random-Gaze",
            "ECE"
        ],
        raw_results_df.loc[
            "Lesion-Supervised",
            "ECE"
        ]
    ],
    "E_AURC": [
        selective_detail[
            "Baseline"
        ]["test"]["E_AURC"],
        selective_detail[
            "Gaze-Original"
        ]["test"]["E_AURC"],
        selective_detail[
            "Shuffled-Gaze"
        ]["test"]["E_AURC"],
        selective_detail[
            "Random-Gaze"
        ]["test"]["E_AURC"],
        selective_detail[
            "Lesion-Supervised"
        ]["test"]["E_AURC"]
    ]
})

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 4.5)
)

axes[0].bar(
    summary_plot_df["Arm"],
    summary_plot_df["Macro_AUROC"]
)
axes[0].set_title(
    "A. Classification: Test Macro AUROC ↑"
)
axes[0].tick_params(
    axis="x",
    rotation=45
)

axes[1].bar(
    summary_plot_df["Arm"],
    summary_plot_df["ECE"]
)
axes[1].set_title(
    "B. Calibration: Test ECE ↓"
)
axes[1].tick_params(
    axis="x",
    rotation=45
)

axes[2].bar(
    summary_plot_df["Arm"],
    summary_plot_df["E_AURC"]
)
axes[2].set_title(
    "C. Selective prediction: Test E-AURC ↓"
)
axes[2].tick_params(
    axis="x",
    rotation=45
)

plt.tight_layout()

plt.savefig(
    RESULT_DIR
    / "five_arm_summary_bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================================
# 22. ARCHIVAL REPORTED RESULTS
# ============================================================================
#
# 아래 값은 이 파일을 현재 환경에서 재실행해 얻은 값이 아니다.
# 과거 대화에서 보존된 실행 출력 / 최종 보고 수치와
# 이전 복원 기록에서 확인된 값을 "historical record"로 별도 보존한다.
#
# 용도:
# - 당시 연구 결과의 숫자 보존
# - 재실행 시 sanity check
#
# 주의:
# - 이 고정값이 곧 이 통합 복구본의 end-to-end reproducibility를 의미하지 않는다.
# - 재실행 결과가 다를 경우 checkpoint, split, dataset 파일,
#   shuffled mapping, lesion-mask preprocessing, 라이브러리 버전을 우선 확인한다.
# ============================================================================

# %% [markdown]
# ⭐⭐당시 최종 5-arm uncalibrated test metrics⭐⭐

# %%
ARCHIVAL_FINAL_UNCALIBRATED = pd.DataFrame(
    [
        {
            "Condition": "Baseline",
            "Accuracy": 0.670732,
            "Macro_F1": 0.656323,
            "Macro_AUROC": 0.867083,
            "ECE": 0.055293,
            "Brier": 0.404887,
            "NLL": 0.668854,
        },
        {
            "Condition": "Gaze-Original",
            "Accuracy": 0.652439,
            "Macro_F1": 0.656791,
            "Macro_AUROC": 0.846250,
            "ECE": 0.059363,
            "Brier": 0.448311,
            "NLL": 0.717421,
        },
        {
            "Condition": "Shuffled-Gaze",
            "Accuracy": 0.689024,
            "Macro_F1": 0.673437,
            "Macro_AUROC": 0.825912,
            "ECE": 0.164286,
            "Brier": 0.485149,
            "NLL": 0.848986,
        },
        {
            "Condition": "Random-Gaze",
            "Accuracy": 0.658537,
            "Macro_F1": 0.638235,
            "Macro_AUROC": 0.823529,
            "ECE": 0.087810,
            "Brier": 0.454969,
            "NLL": 0.729690,
        },
        {
            "Condition": "Lesion-Supervised",
            "Accuracy": 0.640244,
            "Macro_F1": 0.588846,
            "Macro_AUROC": 0.822970,
            "ECE": 0.125479,
            "Brier": 0.474770,
            "NLL": 0.766754,
        },
    ]
).set_index("Condition")


# %% [markdown]
# ⭐⭐당시 최종 temperature-scaled test metrics⭐⭐

# %%
ARCHIVAL_FINAL_TEMPERATURE_SCALED = pd.DataFrame(
    [
        {
            "Condition": "Baseline",
            "Macro_AUROC": 0.867222,
            "ECE": 0.069960,
            "Brier": 0.404856,
            "NLL": 0.663410,
        },
        {
            "Condition": "Gaze-Original",
            "Macro_AUROC": 0.845816,
            "ECE": 0.067280,
            "Brier": 0.447637,
            "NLL": 0.719606,
        },
        {
            "Condition": "Shuffled-Gaze",
            "Macro_AUROC": 0.825457,
            "ECE": 0.089979,
            "Brier": 0.448480,
            "NLL": 0.733560,
        },
        {
            "Condition": "Random-Gaze",
            "Macro_AUROC": 0.823691,
            "ECE": 0.089310,
            "Brier": 0.451085,
            "NLL": 0.726215,
        },
        {
            "Condition": "Lesion-Supervised",
            "Macro_AUROC": 0.821174,
            "ECE": 0.092300,
            "Brier": 0.461256,
            "NLL": 0.747687,
        },
    ]
).set_index("Condition")


# %% [markdown]
# ⭐⭐당시 deployable arm AURC ranking⭐⭐

# %%
ARCHIVAL_DEPLOYABLE_AURC = pd.DataFrame(
    [
        {
            "Rank": 1,
            "Arm": "Baseline",
            "Val_AURC": 0.2131,
            "Test_AURC": 0.1537,
        },
        {
            "Rank": 2,
            "Arm": "Lesion-Supervised",
            "Val_AURC": 0.2236,
            "Test_AURC": 0.2225,
        },
        {
            "Rank": 3,
            "Arm": "Gaze-Original",
            "Val_AURC": 0.2551,
            "Test_AURC": 0.2356,
        },
    ]
)


# %% [markdown]
# ⭐⭐당시 Baseline validation-derived operating points⭐⭐

# %%
ARCHIVAL_BASELINE_OPERATING_POINTS = pd.DataFrame(
    [
        {
            "Target_Coverage": 1.0,
            "Confidence_Threshold": -np.inf,
            "Actual_Coverage": 1.0000,
            "Defer_Rate": 0.0000,
            "AI_Handled": 164,
            "Deferred": 0,
            "Selective_Accuracy": 0.6707,
            "Errors_Total": 54,
            "Errors_Captured_By_Defer": 0,
            "Error_Capture_Rate": 0.0000,
            "Referral_Error_Precision": np.nan,
        },
        {
            "Target_Coverage": 0.9,
            "Confidence_Threshold": 0.4856,
            "Actual_Coverage": 0.9268,
            "Defer_Rate": 0.0732,
            "AI_Handled": 152,
            "Deferred": 12,
            "Selective_Accuracy": 0.6974,
            "Errors_Total": 54,
            "Errors_Captured_By_Defer": 8,
            "Error_Capture_Rate": 0.1481,
            "Referral_Error_Precision": 0.6667,
        },
        {
            "Target_Coverage": 0.8,
            "Confidence_Threshold": 0.5181,
            "Actual_Coverage": 0.8780,
            "Defer_Rate": 0.1220,
            "AI_Handled": 144,
            "Deferred": 20,
            "Selective_Accuracy": 0.7083,
            "Errors_Total": 54,
            "Errors_Captured_By_Defer": 12,
            "Error_Capture_Rate": 0.2222,
            "Referral_Error_Precision": 0.6000,
        },
        {
            "Target_Coverage": 0.7,
            "Confidence_Threshold": 0.5845,
            "Actual_Coverage": 0.7500,
            "Defer_Rate": 0.2500,
            "AI_Handled": 123,
            "Deferred": 41,
            "Selective_Accuracy": 0.7724,
            "Errors_Total": 54,
            "Errors_Captured_By_Defer": 26,
            "Error_Capture_Rate": 0.4815,
            "Referral_Error_Precision": 0.6341,
        },
        {
            "Target_Coverage": 0.6,
            "Confidence_Threshold": 0.6240,
            "Actual_Coverage": 0.6646,
            "Defer_Rate": 0.3354,
            "AI_Handled": 109,
            "Deferred": 55,
            "Selective_Accuracy": 0.7890,
            "Errors_Total": 54,
            "Errors_Captured_By_Defer": 31,
            "Error_Capture_Rate": 0.5741,
            "Referral_Error_Precision": 0.5636,
        },
        {
            "Target_Coverage": 0.5,
            "Confidence_Threshold": 0.6664,
            "Actual_Coverage": 0.5610,
            "Defer_Rate": 0.4390,
            "AI_Handled": 92,
            "Deferred": 72,
            "Selective_Accuracy": 0.8152,
            "Errors_Total": 54,
            "Errors_Captured_By_Defer": 37,
            "Error_Capture_Rate": 0.6852,
            "Referral_Error_Precision": 0.5139,
        },
    ]
)


# %% [markdown]
# ⭐⭐당시 주요 bootstrap / selective-prediction 보고값⭐⭐

# %%
ARCHIVAL_REPORTED_SUMMARY = {
    "GazeOriginal_minus_Baseline_Macro_AUROC": {
        "estimate": -0.0208,
        "ci_low": -0.0574,
        "ci_high": 0.0180,
    },
    "GazeOriginal_minus_Shuffled_ECE": {
        "estimate": -0.1049,
        "ci_low": -0.1488,
        "ci_high": -0.0192,
    },
    "Baseline_Error_Detection_AUROC": {
        "estimate": 0.7692,
        "ci_low": 0.6912,
        "ci_high": 0.8414,
    },
    "Baseline_E_AURC": 0.0914,
    "Representative_Target_Coverage": 0.70,
    "Representative_Test_Coverage": 0.75,
    "Representative_Defer_Rate": 0.25,
    "Representative_Selective_Accuracy": 0.7724,
    "Representative_Total_Errors": 54,
    "Representative_Errors_Captured": 26,
    "Representative_Error_Capture_Rate": 0.4815,
    "Representative_Error_Enrichment": {
        "estimate": 1.93,
        "ci_low": 1.54,
        "ci_high": 2.43,
    },
}


# %% [markdown]
# ⭐⭐당시 class-wise error-detection AUROC⭐⭐

# %%
ARCHIVAL_CLASSWISE_ERROR_DETECTION = pd.DataFrame(
    [
        {
            "Class": "Normal",
            "n": 54,
            "Error_Detection_AUROC": 0.8321,
            "CI_Low": 0.7075,
            "CI_High": 0.9325,
        },
        {
            "Class": "CHF",
            "n": 58,
            "Error_Detection_AUROC": 0.8792,
            "CI_Low": 0.7743,
            "CI_High": 0.9602,
        },
        {
            "Class": "Pneumonia",
            "n": 52,
            "Error_Detection_AUROC": 0.5682,
            "CI_Low": 0.4033,
            "CI_High": 0.7189,
        },
    ]
)


print("\n=== ARCHIVAL FINAL UNCALIBRATED ===")
print(ARCHIVAL_FINAL_UNCALIBRATED)

print("\n=== ARCHIVAL FINAL TEMPERATURE SCALED ===")
print(ARCHIVAL_FINAL_TEMPERATURE_SCALED)

print("\n=== ARCHIVAL DEPLOYABLE AURC ===")
print(ARCHIVAL_DEPLOYABLE_AURC)

print("\n=== ARCHIVAL BASELINE OPERATING POINTS ===")
print(ARCHIVAL_BASELINE_OPERATING_POINTS)

print("\n=== ARCHIVAL REPORTED SUMMARY ===")
print(ARCHIVAL_REPORTED_SUMMARY)

print("\n=== ARCHIVAL CLASS-WISE ERROR DETECTION ===")
print(ARCHIVAL_CLASSWISE_ERROR_DETECTION)


# ============================================================================
# END OF CONVERSATION-BASED RESTORATION / ARCHIVAL FILE
# ============================================================================

