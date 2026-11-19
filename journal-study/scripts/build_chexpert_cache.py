"""11절: CheXpert test(또는 validation) 영상을 학습 영상과 같은 형식(512×512 흑백 PNG)으로 바꾸고 label을 정리한다.

데이터: 이전 AIMI 사이트의 CheXlocalize 묶음 (~/X-ray/data/raw/chexlocalize/chexlocalize/CheXpert/)
    test/ + test_labels.csv (전문의 5명 과반수) / val/ + val_labels.csv (전문의 3명 합의)
규칙 (ANALYSIS_PLAN.md 11절, 예측 전에 고정)
    - 정면(frontal) 영상만 (test: 경로에 frontal, val: Frontal/Lateral 열이 Frontal)
    - pleural = Pleural Effusion 또는 Pleural Other, pleural_effusion = Pleural Effusion,
      cardiac = Cardiomegaly, edema = Edema (label 0/1)
    - 영상: JPG를 흑백으로 읽어 plain(그대로)과 eq(histogram equalization)로 512×512 직접 축소 (BILINEAR)
      → 내부 train 영상과 평균 히스토그램 L1 거리가 작은 쪽 사용 (label·예측 사용 안 함)
    - 신뢰구간은 환자 단위 (labels.csv의 group = 환자 번호)

출력  ~/X-ray/data/processed/chexpert_{test,valid}/ (images_512_plain, images_512_eq, labels.csv, preprocess_choice.json)

사용법  python build_chexpert_cache.py --split test
        python build_chexpert_cache.py --split val
"""
import argparse
import json
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from build_vindr_cache import mean_hist
from vindr_common import PROC

RAW = PROC.parent / "raw/chexlocalize/chexlocalize/CheXpert"
SIZE = 512


def image_id(path: str) -> str:
    m = re.search(r"(patient\d+)/(study\d+)/(view\d+_\w+)\.jpg", path)
    return f"{m.group(1)}_{m.group(2)}_{m.group(3)}"


def work(job):
    iid, src, out_plain, out_eq = job
    try:
        im = Image.open(src).convert("L")
        w, h = im.size
        im.resize((SIZE, SIZE), Image.BILINEAR).save(out_plain / f"{iid}.png")
        ImageOps.equalize(im).resize((SIZE, SIZE), Image.BILINEAR).save(out_eq / f"{iid}.png")
        return {"image_id": iid, "width": w, "height": h, "error": ""}
    except Exception as e:  # noqa: BLE001
        return {"image_id": iid, "width": np.nan, "height": np.nan, "error": str(e)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "val"], required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--hist-sample", type=int, default=1000)
    args = ap.parse_args()
    out = PROC / ("chexpert_test" if args.split == "test" else "chexpert_valid")
    lab = pd.read_csv(RAW / f"{args.split}_labels.csv")
    n_all = len(lab)
    if "Frontal/Lateral" in lab.columns:
        lab = lab[lab["Frontal/Lateral"] == "Frontal"]
    else:
        lab = lab[lab.Path.str.contains("frontal")]
    lab = lab.copy()
    lab["image_id"] = lab.Path.map(image_id)
    lab["src"] = lab.Path.map(lambda p: RAW / args.split / "/".join(p.split("/")[-3:]))
    missing = lab[~lab.src.map(lambda p: p.exists())]
    if len(missing):
        print(f"[주의] 영상 파일이 없는 행 {len(missing)}개 (제외). 예: {missing.Path.iloc[0]}")
    lab = lab[lab.src.map(lambda p: p.exists())]
    print(f"CheXpert {args.split}: label {n_all}행 중 정면 {len(lab)}장")

    (out / "images_512_plain").mkdir(parents=True, exist_ok=True)
    (out / "images_512_eq").mkdir(parents=True, exist_ok=True)
    with Pool(args.workers) as pool:
        meta = pd.DataFrame(pool.map(work, [(r.image_id, r.src, out / "images_512_plain", out / "images_512_eq")
                                            for r in lab.itertuples()]))
    bad = meta[meta.error != ""]
    if len(bad):
        print(f"[주의] 변환 실패 {len(bad)}장 (제외)")
    meta = meta[meta.error == ""]

    res = pd.DataFrame({"image_id": lab.image_id.values,
                        "group": lab.Path.str.extract(r"(patient\d+)")[0].values,
                        "pleural": ((lab["Pleural Effusion"] == 1) | (lab["Pleural Other"] == 1)).astype(int).values,
                        "pleural_effusion": (lab["Pleural Effusion"] == 1).astype(int).values,
                        "cardiac": (lab["Cardiomegaly"] == 1).astype(int).values,
                        "edema": (lab["Edema"] == 1).astype(int).values})
    if "AP/PA" in lab.columns:
        res["view"] = lab["AP/PA"].values
    res = res.merge(meta.drop(columns="error"), on="image_id")
    res.to_csv(out / "labels.csv", index=False)

    split = pd.read_csv(PROC / "split_seed42.csv")
    ids = split.loc[split.split == "train", "dicom_id"].sample(frac=1, random_state=0).head(args.hist_sample)
    h_int = mean_hist([PROC / "images_512" / f"{d}.png" for d in ids])
    rows = []
    for v in ["plain", "eq"]:
        h = mean_hist([out / f"images_512_{v}" / f"{i}.png" for i in res.image_id])
        rows.append({"variant": v, "L1_to_internal": float(np.abs(h - h_int).sum()), "mean": float((h * np.arange(256)).sum())})
    choice = min(rows, key=lambda r: r["L1_to_internal"])["variant"]
    json.dump({"choice": choice, "variants": rows, "internal_mean": float((h_int * np.arange(256)).sum()),
               "rule": "내부 train 영상과 히스토그램 L1 거리 최소"}, open(out / "preprocess_choice.json", "w"), indent=2, ensure_ascii=False)

    print("\n=== label (정면 영상)")
    print(res[["pleural", "pleural_effusion", "cardiac", "edema"]].sum().to_string())
    print(f"영상 {len(res)}장, 환자 {res.group.nunique()}명")
    print("\n=== 전처리 선택 (label·예측 사용 안 함)")
    print(pd.DataFrame(rows).round(3).to_string(index=False), "→", choice)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
