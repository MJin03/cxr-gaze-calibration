"""5·6단계 공용 학습 스크립트.

Baseline과 gaze 모델이 완전히 같은 코드로 돌도록 --gaze 옵션 하나로 조건을 바꾼다.

    --gaze none      Baseline (영상만)
    --gaze real      실제 gaze heatmap으로 auxiliary loss
    --gaze shuffled  다른 환자의 gaze heatmap (shuffle_map_seed{split}.csv)

과제
    --task findings  pleural abnormality, enlarged cardiac silhouette, pulmonary edema (multi-label)
    --task class3    이전 연구 재현용 EGD 3분류 (정상/CHF/폐렴), 이전 연구 split 사용, gaze none만

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
        df = df.groupby("split", group_keys=False).apply(lambda g: g.head(args.max_images))
    return df.reset_index(drop=True), targets


def gaze_source(df: pd.DataFrame, args) -> pd.Series:
    """영상별로 쓸 heatmap 파일의 dicom_id (없으면 NaN)."""
    if args.gaze == "none":
        return pd.Series(np.nan, index=df.index)
    src = df["dicom_id"].where(df["split"] == "train")
    if args.gaze == "shuffled":
        smap = pd.read_csv(PROC / f"shuffle_map_seed{args.split_seed}.csv").set_index("dicom_id")["donor_dicom_id"]
        src = src.map(smap)
    return src


class CXRDataset(Dataset):
    def __init__(self, df, targets, task, img_dir, hm_dir, train, img_size):
        self.df, self.targets, self.task = df.reset_index(drop=True), targets, task
        self.img_dir, self.hm_dir, self.train, self.size = img_dir, hm_dir, train, img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = TF.to_tensor(Image.open(self.img_dir / f"{r.dicom_id}.png").convert("L"))  # 1×H×W, 0~1
        has_gaze = isinstance(r.gaze_src, str)
        hm = (torch.from_numpy(np.load(self.hm_dir / f"{r.gaze_src}.npy").astype(np.float32))[None]
              if has_gaze else torch.zeros(1, self.size, self.size))
        if self.train:
            angle = random.uniform(-5, 5)
            shift = [int(random.uniform(-0.03, 0.03) * self.size) for _ in range(2)]
            scale = random.uniform(0.95, 1.05)
            img = TF.affine(img, angle=angle, translate=shift, scale=scale, shear=[0.0],
                            interpolation=TF.InterpolationMode.BILINEAR)
            if has_gaze:  # 영상과 똑같이 움직여야 위치가 맞는다
                hm = TF.affine(hm, angle=angle, translate=shift, scale=scale, shear=[0.0],
                               interpolation=TF.InterpolationMode.BILINEAR)
            img = ((img - 0.5) * random.uniform(0.9, 1.1) + 0.5 + random.uniform(-0.05, 0.05)).clamp(0, 1)
        if has_gaze:
            s = hm.sum()
            hm = hm / s if s > 0 else hm
            has_gaze = bool(s > 0)
        img = TF.normalize(img.expand(3, -1, -1), MEAN, STD)
        if self.task == "findings":
            y = torch.tensor([float(r[t]) for t in self.targets])
        else:
            y = torch.tensor(int(r[self.targets[0]]))
        return img, y, hm, torch.tensor(has_gaze), i


def seed_worker(_):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


# ---------------------------------------------------------------- 모델
class Net(nn.Module):
    def __init__(self, n_out: int, pretrained: bool):
        super().__init__()
        base = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = base.features
        self.head = nn.Linear(1024, n_out)
        self.gaze_head = nn.Conv2d(1024, 1, kernel_size=1)

    def forward(self, x):
        f = F.relu(self.features(x))
        logits = self.head(F.adaptive_avg_pool2d(f, 1).flatten(1))
        return logits, self.gaze_head(f)


def cls_loss(logits, y, task):
    return (F.binary_cross_entropy_with_logits(logits, y) if task == "findings"
            else F.cross_entropy(logits, y))


def gaze_kl(att, hm, has_gaze):
    """KL(target ∥ model). target = heatmap을 feature map 크기로 줄여 다시 합 1로."""
    if not has_gaze.any():
        return att.sum() * 0.0
    att, hm = att[has_gaze], hm[has_gaze]
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
        return [fit(logits[:, k], y[:, k], F.binary_cross_entropy_with_logits) for k in range(y.shape[1])]
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
    ap.add_argument("--gaze", choices=["none", "real", "shuffled"], default="none")
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
    ap.add_argument("--max-images", type=int, default=0, help="동작 확인용: 세트마다 앞에서 N장만")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--out-root", type=Path, default=HOME / "runs")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.task == "class3" and args.gaze != "none":
        raise SystemExit("class3 재현 확인은 --gaze none 으로만 실행합니다.")

    name = args.run_name or f"{args.task}_gaze-{args.gaze}_t{args.threshold}_seed{args.seed}"
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
    hm_dir = PROC / f"heatmaps_{args.img_size}_s7"
    parts = {s: df[df.split == s] for s in ["train", "val", "test"]}
    print(f"[{name}] device={device}, train {len(parts['train'])} / val {len(parts['val'])} / test {len(parts['test'])}")
    if args.gaze != "none":
        print(f"  gaze heatmap 있는 train 영상: {parts['train'].gaze_src.notna().sum()}장")

    g = torch.Generator().manual_seed(args.seed)
    mk = lambda d, tr: DataLoader(CXRDataset(d, targets, args.task, img_dir, hm_dir, tr, args.img_size),
                                  batch_size=args.batch_size if tr else args.batch_size * 2, shuffle=tr,
                                  num_workers=args.workers, pin_memory=amp, drop_last=tr,
                                  worker_init_fn=seed_worker, generator=g if tr else None,
                                  persistent_workers=args.workers > 0)
    tr_loader, va_loader, te_loader = mk(parts["train"], True), mk(parts["val"], False), mk(parts["test"], False)

    model = Net(len(names), pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
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

    best, bad, hist = math.inf, 0, []
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        sums = np.zeros(3)
        n = 0
        for img, y, hm, hg, _ in tr_loader:
            img, y, hm, hg = img.to(device, non_blocking=True), y.to(device), hm.to(device), hg.to(device)
            with torch.autocast(device.type, enabled=amp):
                logits, att = model(img)
                lc = cls_loss(logits.float(), y, args.task)
            lg = gaze_kl(att, hm, hg) if args.gaze != "none" else torch.zeros((), device=device)
            loss = lc + args.gaze_weight * lg
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sums += [loss.item(), lc.item(), lg.item()]
            n += 1
        v_logits, _ = predict(model, va_loader, device, amp)
        v_loss = float(cls_loss(v_logits, y_val, args.task))
        v_met = summarize(v_logits, y_val, args.task, [1.0] * len(names), names)
        row = {"epoch": ep, "train_loss": sums[0] / n, "train_cls": sums[1] / n, "train_gaze_kl": sums[2] / n,
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
        d.to_csv(out / f"preds_{split}.csv", index=False)
        if split == "val":
            v_logits = logits[torch.argsort(idx)]

    temps = fit_temperature(v_logits, y_val, args.task)
    json.dump(dict(zip(names if args.task == "findings" else ["all"], temps)), open(out / "temperature.json", "w"), indent=2)
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
