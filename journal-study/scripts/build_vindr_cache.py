"""D단계 1: VinDr-CXR test DICOM을 학습 영상과 같은 형식(512×512 흑백 PNG)으로 바꾸고, label·박스를 정리한다.

전처리 (ANALYSIS_PLAN.md 9절)
    1. DICOM VOI LUT(판독용 밝기 설정) 적용, MONOCHROME1이면 흑백 반전, 최솟값~최댓값을 0~255로
    2. 두 가지로 저장: plain (그대로) / eq (histogram equalization)
    3. 학습 영상과 같은 방식으로 비율 유지 없이 512×512로 직접 줄임 (BILINEAR)
    4. 두 방식 중 밝기 분포(256칸 히스토그램 평균)가 내부 train 영상과 더 가까운(L1 거리 작은) 쪽을 고른다
       → label과 모델 예측은 전혀 보지 않고 결정
label
    pleural = Pleural effusion 또는 Pleural thickening, pleural_effusion = Pleural effusion만 (민감도 분석),
    cardiac = Cardiomegaly. 폐부종은 양성 0장이라 평가하지 않음
박스 (pointing game용): annotations_test.csv의 Pleural effusion·Pleural thickening → pleural, Cardiomegaly → cardiac

출력  ~/X-ray/data/processed/vindr_test/
    images_512_plain/{image_id}.png, images_512_eq/{image_id}.png
    labels.csv, boxes.csv, preprocess_choice.json, qc_histogram.png, qc_examples.png

사용법  python build_vindr_cache.py [--workers 8] [--max-images N]
"""
import argparse
import json
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
from PIL import Image, ImageOps

from vindr_common import PROC, VINDR, VINDR_DIR

try:
    from pydicom.pixels import apply_voi_lut
except ImportError:  # pydicom 2.x
    from pydicom.pixel_data_handlers.util import apply_voi_lut

SIZE = 512


def to_uint8(path):
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array
    has_voi = "WindowCenter" in ds or "VOILUTSequence" in ds
    arr = apply_voi_lut(arr, ds).astype(np.float64) if has_voi else arr.astype(np.float64)
    if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
        arr = arr.max() - arr
    lo, hi = arr.min(), arr.max()
    arr = (arr - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(arr)
    return arr.astype(np.uint8), str(getattr(ds, "PhotometricInterpretation", "")), has_voi


def work(job):
    image_id, path, out_plain, out_eq = job
    try:
        u8, photo, has_voi = to_uint8(path)
        im = Image.fromarray(u8)
        h, w = u8.shape
        im.resize((SIZE, SIZE), Image.BILINEAR).save(out_plain / f"{image_id}.png")
        ImageOps.equalize(im).resize((SIZE, SIZE), Image.BILINEAR).save(out_eq / f"{image_id}.png")
        return {"image_id": image_id, "width": w, "height": h, "photometric": photo, "voi": has_voi, "error": ""}
    except Exception as e:  # noqa: BLE001
        return {"image_id": image_id, "width": np.nan, "height": np.nan, "photometric": "", "voi": False, "error": str(e)[:200]}


def mean_hist(paths):
    h = np.zeros(256)
    for p in paths:
        a = np.asarray(Image.open(p).convert("L"))
        h += np.bincount(a.ravel(), minlength=256) / a.size
    return h / max(len(paths), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-images", type=int, default=0, help="동작 확인용")
    ap.add_argument("--hist-sample", type=int, default=1000, help="내부 train 영상 표본 수 (밝기 분포 비교)")
    args = ap.parse_args()

    lab = pd.read_csv(VINDR / "annotations/image_labels_test.csv")
    box = pd.read_csv(VINDR / "annotations/annotations_test.csv")
    files = {p.stem: p for p in VINDR.rglob("*.dicom") if "test" in str(p.parent)}
    lab = lab[lab.image_id.isin(files)]
    if args.max_images:
        lab = lab.head(args.max_images)
    print(f"VinDr test: label {len(lab)}장, DICOM {len(files)}개")

    out_plain, out_eq = VINDR_DIR / "images_512_plain", VINDR_DIR / "images_512_eq"
    out_plain.mkdir(parents=True, exist_ok=True)
    out_eq.mkdir(parents=True, exist_ok=True)
    jobs = [(i, files[i], out_plain, out_eq) for i in lab.image_id]
    meta = []
    with Pool(args.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(work, jobs, chunksize=8), 1):
            meta.append(r)
            if k % 250 == 0:
                print(f"  {k}/{len(jobs)}장 변환", flush=True)
    meta = pd.DataFrame(meta)
    bad = meta[meta.error != ""]
    if len(bad):
        print(f"[주의] 변환 실패 {len(bad)}장 (평가에서 제외):")
        print(bad.head(10).to_string(index=False))
    meta = meta[meta.error == ""]

    # label
    out = lab[["image_id"]].copy()
    out["pleural"] = ((lab["Pleural effusion"] > 0) | (lab["Pleural thickening"] > 0)).astype(int)
    out["pleural_effusion"] = (lab["Pleural effusion"] > 0).astype(int)
    out["cardiac"] = (lab["Cardiomegaly"] > 0).astype(int)
    out["edema"] = (lab["Edema"] > 0).astype(int)
    out["no_finding"] = (lab["No finding"] > 0).astype(int)
    out = out.merge(meta.drop(columns="error"), on="image_id")
    out.to_csv(VINDR_DIR / "labels.csv", index=False)

    # 박스 (원본 픽셀 → 512 좌표)
    cls = {"Pleural effusion": "pleural", "Pleural thickening": "pleural", "Cardiomegaly": "cardiac"}
    b = box[box.class_name.isin(cls)].merge(out[["image_id", "width", "height"]], on="image_id")
    b["finding"] = b.class_name.map(cls)
    for c, dim in [("x_min", "width"), ("x_max", "width"), ("y_min", "height"), ("y_max", "height")]:
        b[c + "_512"] = b[c] * SIZE / b[dim]
    b.to_csv(VINDR_DIR / "boxes.csv", index=False)

    # 전처리 선택: 내부 train 영상과 밝기 분포 비교 (label·예측 사용 안 함)
    split = pd.read_csv(PROC / "split_seed42.csv")
    train_ids = split.loc[split.split == "train", "dicom_id"].sample(frac=1, random_state=0).head(args.hist_sample)
    h_int = mean_hist([PROC / "images_512" / f"{d}.png" for d in train_ids])
    hists, rows = {}, []
    for v, d in [("plain", out_plain), ("eq", out_eq)]:
        hists[v] = mean_hist([d / f"{i}.png" for i in out.image_id])
        rows.append({"variant": v, "L1_to_internal": float(np.abs(hists[v] - h_int).sum()),
                     "mean": float((hists[v] * np.arange(256)).sum())})
    choice = min(rows, key=lambda r: r["L1_to_internal"])["variant"]
    info = {"choice": choice, "internal_mean": float((h_int * np.arange(256)).sum()), "variants": rows,
            "n_images": int(len(out)), "n_failed": int(len(bad)), "rule": "내부 train 영상과 히스토그램 L1 거리 최소"}
    json.dump(info, open(VINDR_DIR / "preprocess_choice.json", "w"), indent=2, ensure_ascii=False)

    plt.figure(figsize=(6, 3.5))
    plt.plot(h_int, label="internal train (MIMIC-CXR-JPG)", color="k")
    for v in hists:
        plt.plot(hists[v], label=f"VinDr {v}")
    plt.legend(fontsize=8)
    plt.title("mean intensity histogram (512 PNG)", fontsize=9)
    plt.savefig(VINDR_DIR / "qc_histogram.png", dpi=80, bbox_inches="tight")
    plt.close()
    ex = out.image_id.head(3).tolist()
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.2))
    axes[0, 0].imshow(Image.open(PROC / "images_512" / f"{train_ids.iloc[0]}.png"), cmap="gray")
    axes[0, 0].set_title("internal", fontsize=8)
    axes[1, 0].imshow(Image.open(PROC / "images_512" / f"{train_ids.iloc[1]}.png"), cmap="gray")
    for j, i in enumerate(ex, 1):
        axes[0, j].imshow(Image.open(out_plain / f"{i}.png"), cmap="gray")
        axes[0, j].set_title(f"VinDr plain {i[:8]}", fontsize=8)
        axes[1, j].imshow(Image.open(out_eq / f"{i}.png"), cmap="gray")
        axes[1, j].set_title(f"VinDr eq {i[:8]}", fontsize=8)
    for a in axes.ravel():
        a.axis("off")
    fig.savefig(VINDR_DIR / "qc_examples.png", dpi=70, bbox_inches="tight")
    plt.close(fig)

    pd.set_option("display.width", 200)
    print("\n=== label (평가 대상 영상)")
    print(out[["pleural", "pleural_effusion", "cardiac", "edema", "no_finding"]].sum().to_string())
    print("\n=== 박스")
    print(b.groupby("finding").agg(boxes=("image_id", "size"), images=("image_id", "nunique")).to_string())
    print("\n=== DICOM 정보")
    print("  photometric:", meta.photometric.value_counts().to_dict(), "| VOI 있음:", int(meta.voi.sum()))
    print("\n=== 전처리 선택 (내부 train 밝기 분포와 비교, label·예측 사용 안 함)")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(f"  내부 평균 밝기 {info['internal_mean']:.1f} → 선택: {choice}")
    print(f"\n저장: {VINDR_DIR}")


if __name__ == "__main__":
    main()
