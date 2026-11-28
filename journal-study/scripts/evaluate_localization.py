"""20절: 위치 평가 — 시선으로 학습한 모델이 실제 병변 위치를 보고 판단하나? (ANALYSIS_PLAN.md 20절, 실행 전에 고정)

데이터
  CheXlocalize (CheXpert test 518장의 전문의 병변 분할 gt_segmentations_test.json, 다른 전문의 분할 hb_segmentations_test.json)
    finding → 분할 이름: pleural → Pleural Effusion, cardiac → Cardiomegaly, edema → Edema
    대상: 그 finding 양성(labels.csv)이고 분할이 비어 있지 않은 영상
  VinDr test (전문의 박스, 9절과 같은 boxes.csv): pleural, cardiac — 일관성 확인용 보조
모델 위치 지도: hybrid3 조건이 외부 예측 때 저장한 finding별 attention (att_{dataset}.npy, 16×16 logit → softmax)
지표 (영상별 → seed 평균)
  hit (pointing game): attention이 가장 강한 칸이 병변과 겹치면 1 (9절과 같은 정의)
  energy: 병변 안에 들어간 attention 비율 (임계값 없음)
  mIoU (CheXlocalize 방식, 보조): attention을 512로 늘려 최댓값 대비 비율 τ 이상을 병변으로 보고 IoU. τ는 CheXpert validation
       (gt_segmentations_val)에서 조건·finding별로 mIoU 최대인 값을 고름 (test 사용 안 함). 사람 기준 = 다른 전문의 분할(hb)과 정답의 IoU
  무작위 기준: energy = 병변 면적 비율, hit = 병변과 겹치는 칸 비율
비교 (영상(환자) + seed bootstrap 2,000회, ci2)
  주: fs − hyb_none (같은 구조, 시선 학습 여부만 다름), fs − fs_shuffled (시선 내용)
  보조: fs − real_global, fs − ellipse, nih_fs − nih_hyb_none, nih_fs − nih_fs_shuffled (19절 모델이 있으면)

사용법  python evaluate_localization.py [--n-boot 2000]
출력    results/localization/ (per_image.csv, summary.csv, comparisons.csv)
"""
import argparse
import json

import numpy as np
import pandas as pd
from PIL import Image

from cal_common import CNX_CONDITIONS, DATASETS, NIH_CONDITIONS, PROC, RESULTS, S10, run_dir

RAW = PROC.parent / "raw/chexlocalize/chexlocalize/CheXlocalize"
OUT = RESULTS / "localization"
FIND = ["pleural", "cardiac", "edema"]   # att 채널 순서 (FINDINGS_MODEL)
TASK = {"pleural": "Pleural Effusion", "cardiac": "Cardiomegaly", "edema": "Edema"}
LABEL_COL = {"pleural": "pleural_effusion", "cardiac": "cardiac", "edema": "edema"}  # CheXlocalize 흉막 분할은 흉수
BASE_CONDS = ["fs", "hyb_none", "fs_shuffled", "real_global", "ellipse"]
COMPS = [("fs", "hyb_none", "primary"), ("fs", "fs_shuffled", "primary"), ("fs", "real_global", "secondary"),
         ("fs", "ellipse", "secondary"), ("nih_fs", "nih_hyb_none", "secondary"), ("nih_fs", "nih_fs_shuffled", "secondary"),
         ("cnx_fs", "cnx_hyb_none", "secondary"), ("cnx_fs", "cnx_fs_shuffled", "secondary")]
TAUS = np.round(np.arange(0.05, 0.96, 0.05), 2)


# ------------------------------------------------------------ COCO RLE (pycocotools 없이)
def rle_counts(s):
    cnts, p = [], 0
    while p < len(s):
        x, k, more = 0, 0, 1
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = c & 0x20
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(cnts) > 2:
            x += cnts[-2]
        cnts.append(x)
    return cnts


def rle_decode(rle):
    h, w = rle["size"]
    cnts = rle_counts(rle["counts"]) if isinstance(rle["counts"], str) else list(rle["counts"])
    flat = np.zeros(h * w, bool)
    pos, val = 0, False
    for c in cnts:
        if val:
            flat[pos:pos + c] = True
        pos += c
        val = not val
    return flat.reshape((h, w), order="F")


def to512(mask):
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((512, 512), Image.NEAREST)) > 127


def cover16(m512, g=16):
    k = 512 // g
    return m512.reshape(g, k, g, k).mean(axis=(1, 3))


def softmax_maps(att):
    flat = att.reshape(att.shape[0], -1).astype(np.float64)
    sm = np.exp(flat - flat.max(1, keepdims=True))
    return sm / sm.sum(1, keepdims=True)


def up512(sm16):
    """16×16 확률 지도 → 512×512 (BILINEAR), 최댓값 1로."""
    im = Image.fromarray(sm16.astype(np.float32), mode="F").resize((512, 512), Image.BILINEAR)
    a = np.asarray(im)
    return a / max(a.max(), 1e-12)


def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else np.nan


# ------------------------------------------------------------ 데이터
def chexloc(split):
    ds = "chexpert_test" if split == "test" else "chexpert_valid"
    lab = pd.read_csv(DATASETS[ds]["labels"])
    gt = json.load(open(RAW / f"gt_segmentations_{split}.json"))
    hb = json.load(open(RAW / "hb_segmentations_test.json")) if split == "test" else {}
    items = {}
    for f in FIND:
        rows = []
        for k, r in enumerate(lab.itertuples()):
            if getattr(r, LABEL_COL[f]) != 1 or r.image_id not in gt or TASK[f] not in gt[r.image_id]:
                continue
            m = rle_decode(gt[r.image_id][TASK[f]])
            if not m.any():
                continue
            m5 = to512(m)
            h5 = to512(rle_decode(hb[r.image_id][TASK[f]])) if r.image_id in hb and TASK[f] in hb[r.image_id] else None
            rows.append({"row": k, "image_id": r.image_id, "group": getattr(r, "group", r.image_id), "m512": m5, "hb512": h5})
        items[f] = rows
    return ds, items


def vindr_items():
    ds = "vindr_test"
    lab = pd.read_csv(DATASETS[ds]["labels"])
    box = pd.read_csv(PROC / "vindr_test/boxes.csv")
    items = {}
    for f in ["pleural", "cardiac"]:
        bf = box[box.finding == f].groupby("image_id")
        rows = []
        for k, r in enumerate(lab.itertuples()):
            if getattr(r, f) != 1 or r.image_id not in bf.groups:
                continue
            m = np.zeros((512, 512), bool)
            for b in bf.get_group(r.image_id).itertuples():
                m[int(np.floor(b.y_min_512)):int(np.ceil(b.y_max_512)), int(np.floor(b.x_min_512)):int(np.ceil(b.x_max_512))] = True
            rows.append({"row": k, "image_id": r.image_id, "group": r.image_id, "m512": m, "hb512": None})
        items[f] = rows
    return ds, items


def conds_available(ds):
    cs = BASE_CONDS + [c for c in ["nih_fs", "nih_hyb_none", "nih_fs_shuffled"] if c in NIH_CONDITIONS] \
        + [c for c in ["cnx_fs", "cnx_hyb_none", "cnx_fs_shuffled"] if c in CNX_CONDITIONS]
    return {c: [s for s in S10 if (run_dir(c, s) / f"att_{ds}.npy").exists()] for c in cs}


# ------------------------------------------------------------ 지표
def per_image(ds, items, seeds):
    out = {}
    for c, ss in seeds.items():
        for s in ss:
            att = np.load(run_dir(c, s) / f"att_{ds}.npy").astype(np.float32)
            for f, rows in items.items():
                if not rows:
                    continue
                idx = np.array([r["row"] for r in rows])
                g = att.shape[-1]
                sm = softmax_maps(att[idx, FIND.index(f)])
                cov = np.stack([cover16(r["m512"], g).ravel() for r in rows])
                hit = (cov[np.arange(len(rows)), sm.argmax(1)] > 0).astype(float)
                energy = (sm * cov).sum(1)
                out[(c, s, f)] = {"hit": hit, "energy": energy, "sm": sm.reshape(len(rows), g, g)}
    return out


def choose_tau(val_items, val_ds, seeds):
    """CheXpert validation에서 조건·finding별 mIoU 최대 τ (seed 평균)."""
    pi = per_image(val_ds, val_items, seeds)
    tau = {}
    for c, ss in seeds.items():
        for f, rows in val_items.items():
            if not rows or not ss:
                continue
            scores = []
            for t in TAUS:
                v = [np.nanmean([iou(up512(pi[(c, s, f)]["sm"][i]) >= t, r["m512"]) for i, r in enumerate(rows)])
                     for s in ss]
                scores.append(np.mean(v))
            tau[(c, f)] = float(TAUS[int(np.nanargmax(scores))])
    return tau


def boot_diff(a_imgs, b_imgs, groups, n_boot, rng):
    """a_imgs, b_imgs: {seed: 영상별 값}. 환자 + seed bootstrap 차이의 2.5·97.5% 분위."""
    ug, gi = np.unique(groups, return_inverse=True)
    sa, sb = list(a_imgs), list(b_imgs)
    A, B = np.stack([a_imgs[s] for s in sa]), np.stack([b_imgs[s] for s in sb])
    d = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[gi].astype(float)
        pa, pb = rng.integers(0, len(sa), len(sa)), rng.integers(0, len(sb), len(sb))
        d.append((A[pa] @ w).mean() / w.sum() - (B[pb] @ w).mean() / w.sum())
    return np.quantile(d, 0.025), np.quantile(d, 0.975)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    OUT.mkdir(parents=True, exist_ok=True)
    summ, comps, per_rows = [], [], []

    # mIoU 임계값 (CheXpert validation)
    vds, vitems = chexloc("val")
    vseeds = {c: s for c, s in conds_available(vds).items() if s}
    print(f"[CheXpert val] 분할 있는 양성: " + ", ".join(f"{f} {len(r)}장" for f, r in vitems.items()), flush=True)
    tau = choose_tau(vitems, vds, vseeds)
    print("  mIoU 임계값 τ (val에서 선택): " + ", ".join(f"{c}/{f} {t:.2f}" for (c, f), t in sorted(tau.items())), flush=True)

    for ds, items in [chexloc("test"), vindr_items()]:
        seeds = {c: s for c, s in conds_available(ds).items() if s}
        print(f"\n[{ds}] 조건 {list(seeds)} | 병변 있는 양성: " + ", ".join(f"{f} {len(r)}장" for f, r in items.items()), flush=True)
        pi = per_image(ds, items, seeds)
        for f, rows in items.items():
            if not rows:
                continue
            groups = np.array([r["group"] for r in rows])
            area = np.array([r["m512"].mean() for r in rows])
            cell_frac = np.array([(cover16(r["m512"]) > 0).mean() for r in rows])
            hb_iou = [iou(r["hb512"], r["m512"]) for r in rows if r["hb512"] is not None]
            summ.append({"dataset": ds, "finding": f, "cond": "무작위 기준", "n_img": len(rows),
                         "hit": cell_frac.mean(), "energy": area.mean(), "miou": np.nan})
            if hb_iou:
                summ.append({"dataset": ds, "finding": f, "cond": "사람(다른 전문의)", "n_img": len(hb_iou),
                             "hit": np.nan, "energy": np.nan, "miou": float(np.nanmean(hb_iou))})
            for c, ss in seeds.items():
                miou = np.nan
                if ds == "chexpert_test" and (c, f) in tau:
                    miou = float(np.mean([np.nanmean([iou(up512(pi[(c, s, f)]["sm"][i]) >= tau[(c, f)], r["m512"])
                                                       for i, r in enumerate(rows)]) for s in ss]))
                summ.append({"dataset": ds, "finding": f, "cond": c, "n_img": len(rows),
                             "hit": np.mean([pi[(c, s, f)]["hit"].mean() for s in ss]),
                             "energy": np.mean([pi[(c, s, f)]["energy"].mean() for s in ss]), "miou": miou})
                for s in ss:
                    per_rows += [{"dataset": ds, "finding": f, "cond": c, "seed": s, "image_id": r["image_id"],
                                  "hit": pi[(c, s, f)]["hit"][i], "energy": pi[(c, s, f)]["energy"][i]} for i, r in enumerate(rows)]
            for a, b, kind in COMPS:
                if a not in seeds or b not in seeds:
                    continue
                for m in ["hit", "energy"]:
                    est = np.mean([pi[(a, s, f)][m].mean() for s in seeds[a]]) - np.mean([pi[(b, s, f)][m].mean() for s in seeds[b]])
                    lo, hi = boot_diff({s: pi[(a, s, f)][m] for s in seeds[a]}, {s: pi[(b, s, f)][m] for s in seeds[b]}, groups, args.n_boot, rng)
                    comps.append({"dataset": ds, "finding": f, "comparison": f"{a}-{b}", "kind": kind, "metric": m,
                                  "estimate": est, "ci2_low": lo, "ci2_high": hi, "n_img": len(rows)})
            print(f"  {f} 완료", flush=True)

    sm, cp = pd.DataFrame(summ), pd.DataFrame(comps)
    sm.to_csv(OUT / "summary.csv", index=False)
    cp.to_csv(OUT / "comparisons.csv", index=False)
    pd.DataFrame(per_rows).to_csv(OUT / "per_image.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n=== 위치 지표 (seed 평균). hit = 가장 강한 칸이 병변에 닿은 비율, energy = 병변 안 attention 비율, mIoU = CheXlocalize 방식")
    print(sm.round(3).to_string(index=False))
    print("\n=== 비교 (앞 − 뒤, 양수 = 앞이 병변을 더 잘 봄) [ci2]. ✓ = CI가 0 미포함")
    cp["표시"] = cp.apply(lambda r: f"{r.estimate:+.3f} [{r.ci2_low:+.3f}, {r.ci2_high:+.3f}]{'✓' if r.ci2_low > 0 or r.ci2_high < 0 else ''}", axis=1)
    print(cp.pivot_table(index=["comparison", "kind", "metric"], columns=["dataset", "finding"], values="표시", aggfunc="first").fillna("없음").to_string())
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
