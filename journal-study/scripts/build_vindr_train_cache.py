"""11절: VinDr-CXR train 무작위 3,000장을 VinDr test와 같은 방식으로 512 PNG로 바꾸고 label을 만든다.

선택: ~/X-ray/data/lists/vindr_train_3000.txt (고유 image_id에서 sample(n=3000, random_state=0))
전처리: VinDr test에서 고른 방식(preprocess_choice.json)을 그대로 쓴다 (같은 병원)
label: 판독의 3명 중 과반수 (판독의 수가 3이 아니면 절반 초과)
    pleural = Pleural effusion 또는 Pleural thickening, pleural_effusion = Pleural effusion만, cardiac = Cardiomegaly

출력  ~/X-ray/data/processed/vindr_train3000/ (images_512_plain, images_512_eq, labels.csv, preprocess_choice.json)

사용법  python build_vindr_train_cache.py [--workers 8]
"""
import argparse
import json
from multiprocessing import Pool

import pandas as pd

from build_vindr_cache import work
from vindr_common import PROC, VINDR

OUT = PROC / "vindr_train3000"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    ids = pd.read_csv(PROC.parent / "lists/vindr_train_3000.txt", header=None)[0].tolist()
    files = {p.stem: p for p in (VINDR / "train").glob("*.dicom") if p.stat().st_size > 0}
    have = [i for i in ids if i in files]
    print(f"목록 {len(ids)}장 중 받은 파일 {len(have)}장")
    if len(have) < len(ids):
        print(f"[주의] 아직 없는 파일 {len(ids) - len(have)}장 (다운로드가 끝났는지 확인)")

    (OUT / "images_512_plain").mkdir(parents=True, exist_ok=True)
    (OUT / "images_512_eq").mkdir(parents=True, exist_ok=True)
    meta = []
    with Pool(args.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(work, [(i, files[i], OUT / "images_512_plain", OUT / "images_512_eq") for i in have],
                                                  chunksize=8), 1):
            meta.append(r)
            if k % 250 == 0:
                print(f"  {k}/{len(have)}장 변환", flush=True)
    meta = pd.DataFrame(meta)
    bad = meta[meta.error != ""]
    if len(bad):
        print(f"[주의] 변환 실패 {len(bad)}장 (제외):")
        print(bad.head(10).to_string(index=False))
    meta = meta[meta.error == ""]

    lab = pd.read_csv(VINDR / "annotations/image_labels_train.csv")
    lab = lab[lab.image_id.isin(meta.image_id)]
    g = lab.groupby("image_id")
    n = g.size()
    vote = lambda col: (g[col].apply(lambda s: (s > 0).sum()) > n / 2).astype(int)
    out = pd.DataFrame({"image_id": n.index})
    pe, pt = vote("Pleural effusion"), vote("Pleural thickening")
    out["pleural"] = ((pe + pt) > 0).astype(int).values
    out["pleural_effusion"] = pe.values
    out["cardiac"] = vote("Cardiomegaly").values
    out["n_readers"] = n.values
    out = out.merge(meta.drop(columns="error"), on="image_id")
    out.to_csv(OUT / "labels.csv", index=False)

    test_choice = json.load(open(PROC / "vindr_test/preprocess_choice.json"))["choice"]
    json.dump({"choice": test_choice, "rule": "VinDr test에서 고른 전처리를 그대로 사용 (같은 병원)"},
              open(OUT / "preprocess_choice.json", "w"), ensure_ascii=False, indent=2)
    print("\n=== label (과반수)")
    print(out[["pleural", "pleural_effusion", "cardiac"]].sum().to_string())
    print("판독의 수:", out.n_readers.value_counts().to_dict(), "| 전처리:", test_choice)
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
