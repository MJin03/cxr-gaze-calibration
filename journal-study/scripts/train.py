"""5·6단계 공용 학습 스크립트.

Baseline과 gaze 모델이 완전히 같은 코드로 돌도록 --gaze 옵션 하나로 조건을 바꾼다.

    --gaze none      Baseline (영상만)
    --gaze real      실제 gaze heatmap으로 auxiliary loss
    --gaze shuffled  다른 환자의 gaze heatmap (shuffle_map_seed{split}.csv)

과제
    --task findings  pleural abnormality, enlarged cardiac silhouette, pulmonary edema (multi-label)
    --task class3    이전 연구 재현용 EGD 3분류 (정상/CHF/폐렴), 이전 연구 split 사용, gaze none만

개선 옵션 (2026-11-17 추가)
    --backbone imagenet|chexpert   chexpert = torchxrayvision densenet121-res224-chex (CheXpert로만 사전학습,
                                   MIMIC-CXR 미포함 → test 누수 없음). 1채널, [-1024, 1024] 정규화
    --schedule constant|cosine     cosine = 1 epoch warmup 후 cosine 감소
    --pool avg|attn                attn = 1×1 conv 위치 분포로 feature를 가중 평균해 분류 (gaze 조건이면 그 분포를 gaze로 학습)
    --tag 문자열                    run 이름 뒤에 붙는 설정 표시 (sweep.py가 자동으로 붙임)

C3 옵션 (2026-11-18 추가, ANALYSIS_PLAN.md 8절 C3)
    --pool attn3|hybrid3           finding마다 attention 지도를 따로 둔다 (1×1 conv 3채널)
                                   attn3   = finding별 attention 특징으로만 판단
                                   hybrid3 = [전체 평균 특징, finding별 attention 특징]을 이어 붙여 판단
    --gaze fs                      finding별 시선 지도 (heatmaps_fs_w{W}_128, W = --fs-window)
    --gaze fs-shuffled             다른 환자의 finding별 시선 지도 (같은 셔플 매핑)
    --gaze ellipse                 판독의가 그린 병변 타원 (heatmaps_ellipse_128, REFLACX 양성만)
    --gaze real|shuffled + attn3/hybrid3  전체 시선 지도 하나를 세 attention에 똑같이 준다 (real-global)
    지도가 없는 finding 채널은 gaze loss에서 뺀다 (mask)

11절 옵션 (2026-11-19 추가): 난이도 가중 label smoothing (train 영상에만, val·test label은 그대로)
    --smooth gaze           시선 난이도 d (gaze_difficulty.csv의 d_{finding})
    --smooth gaze-shuffled  다른 환자의 시선 난이도 (같은 셔플 매핑)
    --smooth certainty      판독의 확신도 난이도 u (u_{finding}, REFLACX만, EGD는 0.5)
    --smooth uniform        모든 영상 0.5 (같은 평균 강도의 일반 smoothing)
    --smooth-max ε          finding별 ε = ε_max × 난이도, 학습 target = y(1 − ε) + ε/2

11-2절 옵션 (2026-11-20 추가): 난이도 예측 헤드 + 영상별 logit 축소 (hybrid3만)
    finding별 특징으로 난이도 d̂ ∈ (0, 1)을 예측하고, 출력 logit = 원래 logit / (1 + softplus(β) × d̂)
    → 어려워 보이는 영상일수록 확신을 낮춘다. β는 학습으로 정해지고, test에서 따로 맞추는 값이 없다
    --diff-target gaze           d̂를 시선 난이도 d에 맞추도록 학습 (그 finding을 말한 train 영상만)
    --diff-target gaze-shuffled  다른 환자의 d에 맞추도록 (대조)
    --diff-target certainty      판독의 확신도 난이도 u에 맞추도록 (REFLACX만)
    --diff-target free           같은 구조, 난이도 지도 학습 없음 (구조 효과만)
    --diff-weight μ              난이도 예측 loss(BCE) 가중치

16절 옵션 (2026-11-24 추가): 순서형(ordinal) 확신도 출력
    --ordinal    finding마다 확신도 문턱 ≥3(기존 출력), ≥4, ≥5 확률을 함께 학습
                 logit_t4 = logit_t3 − softplus(g4), logit_t5 = logit_t4 − softplus(g5) (g = 영상별 간격, 단조 보장)
                 ≥4·≥5 label: REFLACX 판독 확신도 (cohort_readings.csv, 판독 여러 명이면 과반수).
                 EGD 영상은 확신도가 없어 음성(0)만 사용, 양성은 ≥4·≥5에서 제외(mask)
                 checkpoint 선택은 기존과 같은 val ≥3 loss. temperature는 출력마다 val에서
    --ordinal-weight w   ≥4·≥5 BCE 가중치 (기본 1, 사전 고정)

모델
    DenseNet-121 → 마지막 feature map (1024×16×16)
      ├ global average pooling → linear → 분류 logit
      └ 1×1 conv → 공간 softmax → 모델이 보는 위치 분포 (gaze 조건에서만 학습)
    gaze loss = KL(gaze heatmap ∥ 모델 분포), heatmap은 feature map 크기로 줄여서 비교

checkpoint는 val 분류 loss가 가장 낮은 epoch. test는 예측값만 저장하고 지표는 계산하지 않는다
(class3 재현 확인만 예외: 이전 연구의 test로 비교).

출력  ~/X-ray/runs/{task}_gaze-{gaze}_t{threshold}_seed{seed}/
    config.json, history.csv, best.pt, preds_val.csv, preds_test.csv, temperature.json, val_metrics.json

예시
    python train.py --task class3 --seed 0
    python train.py --task findings --gaze none --seed 0
    python train.py --task findings --gaze none --seed 0 --max-images 64 --epochs 1 --run-name dryrun   # 빠른 동작 확인
"""
import argparse
import json
import math
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import DenseNet121_Weights, densenet121

HOME = Path.home() / "X-ray"
PROC = HOME / "data/processed"
PREV_SPLIT = HOME / "archive/kim_mj_sens_X-ray/gaze_calibration_research/splits/egd_cxr_cohort_with_split_seed42.csv"
FINDINGS = ["pleural", "cardiac", "edema"]
CLASS3 = ["Normal", "CHF", "pneumonia"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ---------------------------------------------------------------- 데이터
def find_col(df: pd.DataFrame, candidates: list, what: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(f"{what} 열을 찾지 못했습니다. 후보 {candidates}, 실제 열 {list(df.columns)}")


def build_table(args) -> tuple:
    lab = pd.read_csv(PROC / "cohort_images_labeled.csv")
    if args.task == "findings":
        split = pd.read_csv(PROC / f"split_seed{args.split_seed}.csv")[["dicom_id", "split"]]
        df = lab.merge(split, on="dicom_id")
        targets = [f"{f}_t{args.threshold}" for f in FINDINGS]
    else:
        prev = pd.read_csv(args.prev_split)
        id_col = find_col(prev, ["dicom_id", "DICOM_ID"], "영상 ID")
        sp_col = find_col(prev, ["split", "Split", "set", "subset"], "split")
        prev = prev[[id_col, sp_col]].rename(columns={id_col: "dicom_id", sp_col: "split"})
        prev["split"] = prev["split"].astype(str).str.lower().replace({"validation": "val", "valid": "val"})
        df = lab[lab["class3"].notna()].merge(prev, on="dicom_id")
        df["class3_idx"] = df["class3"].map({c: i for i, c in enumerate(CLASS3)})
        targets = ["class3_idx"]
    if args.max_images:
        df = df.groupby("split").head(args.max_images)
    df = df.reset_index(drop=True)
    if args.task == "findings":
        add_smoothing(df, args)
        add_difficulty_target(df, args)
        add_ordinal_targets(df, args)
    return df, targets


ORD_T = [4, 5]


def ord_cols():
    return [f"o{t}_{f}" for t in ORD_T for f in FINDINGS]  # 순서: t4 (pleural, cardiac, edema), t5 (...)


def add_ordinal_targets(df: pd.DataFrame, args) -> None:
    """≥4·≥5 영상 label o{t}_{finding} (확신도 모르는 EGD 양성은 NaN = loss 제외)."""
    if not getattr(args, "ordinal", False):
        return
    reads = pd.read_csv(PROC / "cohort_readings.csv")
    ref = reads[reads.dataset == "reflacx"]
    for t in ORD_T:
        for f in FINDINGS:
            g = (ref[f"{f}_raw"] >= t).groupby(ref.dicom_id).agg(["sum", "size"])
            lab_ref = (g["sum"] > g["size"] / 2).astype(float)
            v = df.dicom_id.map(lab_ref)
            egd_only = v.isna()
            v[egd_only & (df[f"{f}_t3"] == 0)] = 0.0  # EGD 음성 → 모든 문턱에서 음성
            df[f"o{t}_{f}"] = v.values  # EGD 양성은 NaN


def add_difficulty_target(df: pd.DataFrame, args) -> None:
    """난이도 예측 헤드의 정답 dt_{finding}과 사용 여부 dm_{finding} (train 영상만)."""
    for f in FINDINGS:
        df[f"dt_{f}"], df[f"dm_{f}"] = 0.5, 0.0
    if args.diff_target in ("none", "free"):
        return
    g = pd.read_csv(PROC / "gaze_difficulty.csv").set_index("dicom_id")
    src = df["dicom_id"]
    if args.diff_target == "gaze-shuffled":
        smap = pd.read_csv(PROC / f"shuffle_map_seed{args.split_seed}.csv").set_index("dicom_id")["donor_dicom_id"]
        src = src.map(smap)
    train = (df["split"] == "train").values
    for f in FINDINGS:
        if args.diff_target == "certainty":
            val, ok = src.map(g[f"u_{f}"]), src.map(g[f"cert_{f}"]).notna()
        else:
            val, ok = src.map(g[f"d_{f}"]), src.map(g[f"dwell_{f}"]).notna()
        use = train & ok.values & val.notna().values
        df.loc[use, f"dt_{f}"] = val[use].clip(0, 1).values
        df.loc[use, f"dm_{f}"] = 1.0


def add_smoothing(df: pd.DataFrame, args) -> None:
    """finding별 smoothing 강도 eps_{finding} 열을 만든다 (train 영상만, 나머지는 0)."""
    for f in FINDINGS:
        df[f"eps_{f}"] = 0.0
    if args.smooth == "none" or args.smooth_max <= 0:
        return
    if args.smooth == "uniform":
        diff = pd.DataFrame({f: 0.5 for f in FINDINGS}, index=df.index)
    else:
        g = pd.read_csv(PROC / "gaze_difficulty.csv").set_index("dicom_id")
        col = "u" if args.smooth == "certainty" else "d"
        src = df["dicom_id"]
        if args.smooth == "gaze-shuffled":
            smap = pd.read_csv(PROC / f"shuffle_map_seed{args.split_seed}.csv").set_index("dicom_id")["donor_dicom_id"]
            src = src.map(smap)
        diff = pd.DataFrame({f: src.map(g[f"{col}_{f}"]).fillna(0.5).values for f in FINDINGS}, index=df.index)
    train = (df["split"] == "train").values
    for f in FINDINGS:
        df.loc[train, f"eps_{f}"] = args.smooth_max * diff.loc[train, f].clip(0, 1)


def gaze_source(df: pd.DataFrame, args) -> pd.Series:
    """영상별로 쓸 heatmap 파일의 dicom_id (없으면 NaN)."""
    if args.gaze == "none":
        return pd.Series(np.nan, index=df.index)
    src = df["dicom_id"].where(df["split"] == "train")
    if args.gaze in ("shuffled", "fs-shuffled"):
        smap = pd.read_csv(PROC / f"shuffle_map_seed{args.split_seed}.csv").set_index("dicom_id")["donor_dicom_id"]
        src = src.map(smap)
    return src


class CXRDataset(Dataset):
    """n_maps = 1 (기존 avg/attn: 512 지도 1장) 또는 3 (attn3/hybrid3: finding별 128 지도 3장)."""
    MULTI_RES = 128

    def __init__(self, df, targets, task, img_dir, hm_dir, train, img_size, n_maps=1, repeat_global=False, ordinal=False):
        self.df, self.targets, self.task = df.reset_index(drop=True), targets, task
        self.ord = ord_cols() if ordinal else []
        self.img_dir, self.hm_dir, self.train, self.size = img_dir, hm_dir, train, img_size
        self.n_maps, self.repeat_global = n_maps, repeat_global
        self.res = img_size if n_maps == 1 else self.MULTI_RES

    def __len__(self):
        return len(self.df)

    def load_map(self, src):
        hm = torch.from_numpy(np.load(self.hm_dir / f"{src}.npy").astype(np.float32))
        if hm.dim() == 2:
            hm = hm[None]
        if self.repeat_global:  # 전체 시선 지도 하나를 세 finding에 똑같이
            hm = hm.expand(self.n_maps, -1, -1).contiguous()
        if hm.shape[-1] != self.res:  # 입력 크기(또는 128)에 맞게 평균으로 줄인다
            hm = F.adaptive_avg_pool2d(hm[None], self.res)[0]
        return hm

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = TF.to_tensor(Image.open(self.img_dir / f"{r.dicom_id}.png").convert("L"))  # 1×H×W, 0~1
        has_gaze = isinstance(r.gaze_src, str) and (self.hm_dir / f"{r.gaze_src}.npy").exists()
        hm = self.load_map(r.gaze_src) if has_gaze else torch.zeros(self.n_maps, self.res, self.res)
        if self.train:
            angle = random.uniform(-5, 5)
            frac = [random.uniform(-0.03, 0.03) for _ in range(2)]
            shift = [int(f * self.size) for f in frac]
            scale = random.uniform(0.95, 1.05)
            img = TF.affine(img, angle=angle, translate=shift, scale=scale, shear=[0.0],
                            interpolation=TF.InterpolationMode.BILINEAR)
            if has_gaze:  # 영상과 똑같이 움직여야 위치가 맞는다 (지도 해상도에 맞춰 이동량 환산)
                hshift = shift if self.res == self.size else [int(round(f * self.res)) for f in frac]
                hm = TF.affine(hm, angle=angle, translate=hshift, scale=scale, shear=[0.0],
                               interpolation=TF.InterpolationMode.BILINEAR)
            img = ((img - 0.5) * random.uniform(0.9, 1.1) + 0.5 + random.uniform(-0.05, 0.05)).clamp(0, 1)
        s = hm.flatten(1).sum(1)  # 채널별 합
        mask = s > 0
        hm = hm / s.clamp_min(1e-12)[:, None, None]
        if self.n_maps == 1:
            has_gaze = bool(mask[0])
        else:
            has_gaze = mask
        if self.task == "findings":
            y = torch.tensor([float(r[t]) for t in self.targets])
            if self.train:  # 난이도 가중 label smoothing (eps가 0이면 원래 label 그대로)
                eps = torch.tensor([float(r[f"eps_{f}"]) for f in FINDINGS])
                y = y * (1 - eps) + eps / 2
            if self.ord:  # 순서형: 뒤에 ≥4·≥5 label (NaN = 제외)
                y = torch.cat([y, torch.tensor([float(r[c]) for c in self.ord])])
        else:
            y = torch.tensor(int(r[self.targets[0]]))
        return img, y, hm, torch.as_tensor(has_gaze), i


def seed_worker(_):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


# ---------------------------------------------------------------- 모델
class Net(nn.Module):
    def __init__(self, n_out: int, backbone: str, pretrained: bool, pool: str, diff: bool = False, ordinal: bool = False):
        super().__init__()
        self.backbone, self.pool = backbone, pool
        if backbone == "chexpert":
            import torchxrayvision as xrv
            base = xrv.models.DenseNet(weights="densenet121-res224-chex" if pretrained else None)
        else:
            base = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = base.features
        self.multi = pool in ("attn3", "hybrid3")
        self.head = nn.Linear(1024, n_out)
        if self.multi:  # finding별 분류기: attn3 = 1024, hybrid3 = 2048 (평균 + attention)
            d = 2048 if pool == "hybrid3" else 1024
            self.head_w = nn.Parameter(torch.empty(n_out, d))
            self.head_b = nn.Parameter(torch.zeros(n_out))
            nn.init.kaiming_uniform_(self.head_w, a=math.sqrt(5))
        self.diff = diff and pool == "hybrid3"
        if self.diff:  # 난이도 예측 헤드와 logit 축소 강도 β (softplus(−1) ≈ 0.31에서 시작)
            self.diff_w = nn.Parameter(torch.zeros(n_out, 2048))
            self.diff_b = nn.Parameter(torch.zeros(n_out))
            self.beta = nn.Parameter(torch.full((n_out,), -1.0))
        self.last_dlogit = None
        self.ordinal = ordinal
        if ordinal:  # finding별 ≥4, ≥5 간격 헤드 (hybrid3 = finding별 특징 2048, 그 외 = 공통 특징 1024)
            d = 2048 if pool == "hybrid3" else 1024 if pool in ("attn3", "avg", "attn") else 1024
            self.ord_w = nn.Parameter(torch.zeros(n_out, len(ORD_T), d))
            self.ord_b = nn.Parameter(torch.zeros(n_out, len(ORD_T)))
        self.last_ord = None
        self.gaze_head = nn.Conv2d(1024, n_out if self.multi else 1, kernel_size=1)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x):  # x: B×1×H×W, 0~1
        if self.backbone == "chexpert":
            x = x * 2048 - 1024
        else:
            x = (x.expand(-1, 3, -1, -1) - self.mean) / self.std
        f = F.relu(self.features(x))
        att = self.gaze_head(f)
        if self.multi:
            w = torch.softmax(att.flatten(2).float(), dim=-1).to(f.dtype)  # B×K×HW
            per = torch.einsum("bkn,bcn->bkc", w, f.flatten(2))            # B×K×1024 (finding별 attention 특징)
            if self.pool == "hybrid3":
                g = F.adaptive_avg_pool2d(f, 1).flatten(1)                 # B×1024
                per = torch.cat([g[:, None].expand(-1, per.shape[1], -1), per], dim=-1)
            logits = (per * self.head_w.to(per.dtype)).sum(-1) + self.head_b.to(per.dtype)
            if self.ordinal:
                self.last_ord = self.ord_logits(logits.float(), torch.einsum("bkd,ktd->bkt", per.float(), self.ord_w))
            if self.diff:
                dlogit = (per.float() * self.diff_w).sum(-1) + self.diff_b       # B×K
                scale = 1 + F.softplus(self.beta) * torch.sigmoid(dlogit)
                logits = logits.float() / scale
                self.last_dlogit = dlogit
            return logits, att
        if self.pool == "attn":
            w = torch.softmax(att.flatten(2), dim=-1)            # B×1×HW
            pooled = (f.flatten(2) * w).sum(-1)                   # B×1024
        else:
            pooled = F.adaptive_avg_pool2d(f, 1).flatten(1)
        logits = self.head(pooled)
        if self.ordinal:
            self.last_ord = self.ord_logits(logits.float(), torch.einsum("bd,ktd->bkt", pooled.float(), self.ord_w))
        return logits, att

    def ord_logits(self, l3, g):
        """l3: B×K (≥3 logit), g: B×K×T 간격 점수 → B×(T·K) [t4 finding들, t5 finding들], 단조 감소 보장."""
        gaps = F.softplus(g + self.ord_b)
        out = l3[..., None] - torch.cumsum(gaps, dim=-1)   # B×K×T
        return out.transpose(1, 2).flatten(1)


def cls_loss(logits, y, task):
    return (F.binary_cross_entropy_with_logits(logits, y) if task == "findings"
            else F.cross_entropy(logits, y))


def gaze_kl(att, hm, has_gaze):
    """KL(target ∥ model). target = heatmap을 feature map 크기로 줄여 다시 합 1로.
    has_gaze: B (지도 1장) 또는 B×K (finding별, 지도 없는 채널은 제외). 지도 하나하나를 평균."""
    if not has_gaze.any():
        return att.sum() * 0.0
    if has_gaze.dim() == 1:
        att, hm = att[has_gaze], hm[has_gaze]
    else:
        att = att.flatten(0, 1)[has_gaze.flatten()][:, None]
        hm = hm.flatten(0, 1)[has_gaze.flatten()][:, None]
    t = F.adaptive_avg_pool2d(hm, att.shape[-2:]).flatten(1)
    t = t / t.sum(1, keepdim=True).clamp_min(1e-12)
    logp = F.log_softmax(att.flatten(1).float(), dim=1)
    return (t * (torch.log(t.clamp_min(1e-12)) - logp)).sum(1).mean()


# ---------------------------------------------------------------- 평가 도구
@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    out, idx = [], []
    for img, _, _, _, i in loader:
        with torch.autocast(device.type, enabled=amp):
            logits, _ = model(img.to(device, non_blocking=True))
        if getattr(model, "ordinal", False):
            logits = torch.cat([logits.float(), model.last_ord.float()], 1)
        out.append(logits.float().cpu())
        idx.append(i)
    return torch.cat(out), torch.cat(idx)


def fit_temperature(logits: torch.Tensor, y: torch.Tensor, task: str) -> list:
    """val에서 NLL을 최소화하는 temperature. findings는 라벨마다 하나, class3는 하나."""
    def fit(lg, yy, loss_fn):
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

        def closure():
            opt.zero_grad()
            loss = loss_fn(lg / log_t.exp(), yy)
            loss.backward()
            return loss
        opt.step(closure)
        return float(log_t.detach().exp())
    if task == "findings":
        res = []
        for k in range(y.shape[1]):
            ok = ~torch.isnan(y[:, k])
            res.append(fit(logits[ok, k], y[ok, k], F.binary_cross_entropy_with_logits))
        return res
    return [fit(logits, y, F.cross_entropy)]


def ece_binary(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """양성 확률 p의 calibration error (15개 구간)."""
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def summarize(logits, y, task, temps, names):
    res = {}
    if task == "findings":
        for k, n in enumerate(names):
            yk = y[:, k].numpy()
            for tag, t in [("raw", 1.0), ("ts", temps[k])]:
                p = torch.sigmoid(logits[:, k] / t).numpy()
                res[f"{n}_auroc"] = float(roc_auc_score(yk, p)) if len(set(yk)) > 1 else float("nan")
                res[f"{n}_ece_{tag}"] = ece_binary(p, yk)
    else:
        p = torch.softmax(logits / temps[0], 1).numpy()
        yy = y.numpy()
        res["macro_auroc"] = float(roc_auc_score(yy, p, multi_class="ovr", average="macro"))
        res["accuracy"] = float((p.argmax(1) == yy).mean())
    return res


# ---------------------------------------------------------------- 메인
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["findings", "class3"], default="findings")
    ap.add_argument("--gaze", choices=["none", "real", "shuffled", "fs", "fs-shuffled", "ellipse"], default="none")
    ap.add_argument("--fs-window", type=float, default=3.0, help="finding별 시선 지도의 창 길이(초)")
    ap.add_argument("--gaze-weight", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--threshold", type=int, default=3, choices=[3, 4])
    ap.add_argument("--prev-split", type=Path, default=PREV_SPLIT)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--backbone", choices=["imagenet", "chexpert"], default="imagenet")
    ap.add_argument("--schedule", choices=["constant", "cosine"], default="constant")
    ap.add_argument("--pool", choices=["avg", "attn", "attn3", "hybrid3"], default="avg")
    ap.add_argument("--smooth", choices=["none", "gaze", "gaze-shuffled", "certainty", "uniform"], default="none")
    ap.add_argument("--smooth-max", type=float, default=0.0)
    ap.add_argument("--diff-target", choices=["none", "gaze", "gaze-shuffled", "certainty", "free"], default="none")
    ap.add_argument("--diff-weight", type=float, default=0.0)
    ap.add_argument("--ordinal", action="store_true", help="16절: 확신도 ≥3·≥4·≥5 순서형 출력")
    ap.add_argument("--ordinal-weight", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-images", type=int, default=0, help="동작 확인용: 세트마다 앞에서 N장만")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--out-root", type=Path, default=HOME / "runs")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.task == "class3" and args.gaze != "none":
        raise SystemExit("class3 재현 확인은 --gaze none 으로만 실행합니다.")
    multi = args.pool in ("attn3", "hybrid3")
    if args.gaze in ("fs", "fs-shuffled", "ellipse") and not multi:
        raise SystemExit(f"--gaze {args.gaze}는 finding별 attention이 필요합니다: --pool attn3 또는 hybrid3")
    if multi and args.task != "findings":
        raise SystemExit("attn3/hybrid3는 --task findings 에서만 씁니다.")
    if args.ordinal and (args.task != "findings" or args.diff_target != "none" or args.pool not in ("avg", "hybrid3")):
        raise SystemExit("--ordinal은 findings, pool avg/hybrid3, diff-target none에서만 씁니다.")

    name = args.run_name or (f"{args.task}_gaze-{args.gaze}_t{args.threshold}_seed{args.seed}"
                             + (f"_{args.tag}" if args.tag else ""))
    out = args.out_root / name
    if (out / "best.pt").exists() and not args.overwrite:
        raise SystemExit(f"{out} 에 이미 결과가 있습니다. 다시 돌리려면 --overwrite")
    out.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"

    df, targets = build_table(args)
    df["gaze_src"] = gaze_source(df, args)
    names = FINDINGS if args.task == "findings" else CLASS3
    img_dir = PROC / f"images_{args.img_size}"
    hm_dir = (PROC / f"heatmaps_fs_w{args.fs_window:g}_128" if args.gaze in ("fs", "fs-shuffled")
              else PROC / "heatmaps_ellipse_128" if args.gaze == "ellipse" else PROC / "heatmaps_512_s7")
    n_maps = len(FINDINGS) if multi else 1
    repeat_global = multi and args.gaze in ("real", "shuffled")
    parts = {s: df[df.split == s] for s in ["train", "val", "test"]}
    print(f"[{name}] device={device}, train {len(parts['train'])} / val {len(parts['val'])} / test {len(parts['test'])}")
    if args.smooth != "none" and args.smooth_max > 0:
        e = parts["train"][[f"eps_{f}" for f in FINDINGS]]
        print(f"  label smoothing {args.smooth} ε_max {args.smooth_max}: train 평균 ε " +
              ", ".join(f"{f} {e[f'eps_{f}'].mean():.3f}" for f in FINDINGS))
    if args.gaze != "none":
        has_file = parts["train"].gaze_src.map(lambda s: isinstance(s, str) and (hm_dir / f"{s}.npy").exists())
        print(f"  gaze 지도: {hm_dir.name}, 지도 있는 train 영상 {int(has_file.sum())}장"
              + (" (finding별, 없는 채널은 loss 제외)" if multi and not repeat_global else ""))

    g = torch.Generator().manual_seed(args.seed)
    mk = lambda d, tr: DataLoader(CXRDataset(d, targets, args.task, img_dir, hm_dir, tr, args.img_size,
                                             n_maps, repeat_global, args.ordinal),
                                  batch_size=args.batch_size if tr else args.batch_size * 2, shuffle=tr,
                                  num_workers=args.workers, pin_memory=amp, drop_last=tr and len(d) > args.batch_size,
                                  worker_init_fn=seed_worker, generator=g if tr else None,
                                  persistent_workers=args.workers > 0)
    tr_loader, va_loader, te_loader = mk(parts["train"], True), mk(parts["val"], False), mk(parts["test"], False)

    use_diff = args.diff_target != "none"
    if use_diff and args.pool != "hybrid3":
        raise SystemExit("--diff-target는 --pool hybrid3에서만 씁니다.")
    model = Net(len(names), args.backbone, not args.no_pretrained, args.pool, diff=use_diff, ordinal=args.ordinal).to(device)
    K = len(names)
    if args.ordinal:
        oc = parts["train"][ord_cols()]
        print("  순서형 ≥4·≥5 label (train 양성/사용): " + ", ".join(
            f"{c} {int(np.nansum(oc[c]))}/{int(oc[c].notna().sum())}" for c in ord_cols()))
    if args.task == "findings":  # 난이도 정답 (train 영상 순서와 같은 인덱스)
        idx_train = np.arange(len(parts["train"]))
        dt_all = parts["train"][[f"dt_{f}" for f in FINDINGS]].values.astype(np.float32)
        dm_all = parts["train"][[f"dm_{f}" for f in FINDINGS]].values.astype(np.float32)
        if args.diff_target not in ("none", "free"):
            print(f"  난이도 예측 헤드 {args.diff_target} μ {args.diff_weight}: 정답 있는 train 영상 비율 " +
                  ", ".join(f"{f} {dm_all[:, k].mean():.3f}" for k, f in enumerate(FINDINGS)))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = max(len(tr_loader), 1)
    total, warm = args.epochs * steps_per_epoch, steps_per_epoch
    lr_lambda = (lambda i: min(1.0, (i + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(i, total) / total))
                 if args.schedule == "cosine" else 1.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    try:
        commit = subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = ""
    json.dump({**{k: str(v) for k, v in vars(args).items()}, "targets": targets, "git_commit": commit,
               "torch": torch.__version__, "device": str(device),
               "n_train": len(parts["train"]), "n_val": len(parts["val"]), "n_test": len(parts["test"])},
              open(out / "config.json", "w"), indent=2, ensure_ascii=False)

    y_val = torch.tensor(parts["val"][targets].values, dtype=torch.float32 if args.task == "findings" else torch.long)
    if args.task == "class3":
        y_val = y_val.squeeze(1)

    y_val_ord = torch.tensor(parts["val"][ord_cols()].values, dtype=torch.float32) if args.ordinal else None
    best, bad, hist = math.inf, 0, []
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        sums = np.zeros(5)
        n = 0
        for img, y, hm, hg, i_batch in tr_loader:
            img, y, hm, hg = img.to(device, non_blocking=True), y.to(device), hm.to(device), hg.to(device)
            y_ord = None
            if args.ordinal:
                y, y_ord = y[:, :K], y[:, K:]
            with torch.autocast(device.type, enabled=amp):
                logits, att = model(img)
                lc = cls_loss(logits.float(), y, args.task)
            lg = gaze_kl(att, hm, hg) if args.gaze != "none" else torch.zeros((), device=device)
            loss = lc + args.gaze_weight * lg
            ld = torch.zeros((), device=device)
            if args.diff_target not in ("none", "free") and args.diff_weight > 0:
                bi = idx_train[i_batch.numpy()]
                dt = torch.tensor(dt_all[bi], device=device)
                dm = torch.tensor(dm_all[bi], device=device) > 0
                if dm.any():
                    ld = F.binary_cross_entropy_with_logits(model.last_dlogit[dm].float(), dt[dm])
                    loss = loss + args.diff_weight * ld
            lo = torch.zeros((), device=device)
            if args.ordinal:
                ok = ~torch.isnan(y_ord)
                if ok.any():
                    lo = F.binary_cross_entropy_with_logits(model.last_ord.float()[ok], y_ord[ok])
                    loss = loss + args.ordinal_weight * lo
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            sums += [loss.item(), lc.item(), lg.item(), ld.item(), lo.item()]
            n += 1
        v_logits, _ = predict(model, va_loader, device, amp)
        v_logits = v_logits[:, :K] if args.ordinal else v_logits
        v_loss = float(cls_loss(v_logits, y_val, args.task))
        v_met = summarize(v_logits, y_val, args.task, [1.0] * len(names), names)
        row = {"epoch": ep, "train_loss": sums[0] / n, "train_cls": sums[1] / n, "train_gaze_kl": sums[2] / n,
               "train_diff": sums[3] / n, "train_ord": sums[4] / n,
               "val_cls_loss": v_loss, **{k: v for k, v in v_met.items() if "auroc" in k or k == "accuracy"},
               "sec": round(time.time() - t0, 1)}
        hist.append(row)
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        aucs = ", ".join(f"{k.replace('_auroc', '')} {v:.3f}" for k, v in row.items() if "auroc" in k)
        print(f"  epoch {ep:2d} | train {row['train_loss']:.4f} (kl {row['train_gaze_kl']:.3f}) | "
              f"val loss {v_loss:.4f} | val AUROC {aucs} | {row['sec']}s")
        if v_loss < best:
            best, bad = v_loss, 0
            torch.save({"model": model.state_dict(), "epoch": ep}, out / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stopping (val loss가 {args.patience}번 연속 개선 안 됨)")
                break

    ck = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ck["model"])
    print(f"  best epoch {ck['epoch']} (val loss {best:.4f})")

    for split, loader in [("val", va_loader), ("test", te_loader)]:
        logits, idx = predict(model, loader, device, amp)
        d = parts[split].iloc[idx.numpy()][["dicom_id", "subject_id", "ViewPosition"] + targets].reset_index(drop=True)
        for k, nme in enumerate(names):
            d[f"logit_{nme}"] = logits[:, k].numpy()
        if args.ordinal:
            for j, c in enumerate(ord_cols()):  # o4_pleural → logit_pleural_t4
                t, f = c[1], c.split("_", 1)[1]
                d[f"logit_{f}_t{t}"] = logits[:, K + j].numpy()
        d.to_csv(out / f"preds_{split}.csv", index=False)
        if split == "val":
            v_full = logits[torch.argsort(idx)]
            v_logits = v_full[:, :K]

    temps = fit_temperature(v_logits, y_val, args.task)
    tdict = dict(zip(names if args.task == "findings" else ["all"], temps))
    if args.ordinal:
        t_ord = fit_temperature(v_full[:, K:], y_val_ord, args.task)
        for c, tv in zip(ord_cols(), t_ord):
            tdict[f"{c.split('_', 1)[1]}_t{c[1]}"] = tv
        print("  순서형 temperature:", {k: round(v, 3) for k, v in tdict.items() if "_t" in k})
    json.dump(tdict, open(out / "temperature.json", "w"), indent=2)
    v_met = summarize(v_logits, y_val, args.task, temps, names)
    v_met["best_epoch"] = ck["epoch"]
    json.dump(v_met, open(out / "val_metrics.json", "w"), indent=2)
    print("  temperature:", [round(t, 3) for t in temps])
    print("  val 지표:", {k: round(v, 4) for k, v in v_met.items()})

    if args.task == "class3":  # 이전 연구 재현 확인용으로만 test 지표를 본다
        te = pd.read_csv(out / "preds_test.csv")
        lg = torch.tensor(te[[f"logit_{c}" for c in CLASS3]].values, dtype=torch.float32)
        met = summarize(lg, torch.tensor(te["class3_idx"].values), "class3", [1.0], CLASS3)
        print(f"  [재현 확인] test macro AUROC {met['macro_auroc']:.4f}, accuracy {met['accuracy']:.4f} "
              f"(이전 연구 Baseline: 0.8671 / 0.6707, test {len(te)}장)")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
