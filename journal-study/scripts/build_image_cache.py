"""5-0단계: 학습용 영상을 512×512 흑백 PNG로 한 번만 줄여 저장한다 (image cache).

gaze heatmap과 같은 방식(비율 유지 없이 512×512로 직접 resize)으로 줄여야
heatmap의 위치와 영상의 위치가 맞는다.

입력  ~/X-ray/data/processed/cohort_images.csv
출력  ~/X-ray/data/processed/images_512/{dicom_id}.png

사용법  python build_image_cache.py [--size 512] [--workers 8]
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from PIL import Image

PROC = Path.home() / "X-ray/data/processed"


def work(args):
    dicom_id, src, out_dir, size = args
    dst = out_dir / f"{dicom_id}.png"
    if dst.exists():
        return dicom_id, "skip"
    try:
        im = Image.open(src).convert("L").resize((size, size), Image.BILINEAR)
        im.save(dst)
        return dicom_id, "ok"
    except Exception as e:  # noqa: BLE001
        return dicom_id, f"error: {e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    img = pd.read_csv(PROC / "cohort_images.csv")
    out_dir = PROC / f"images_{args.size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(r.dicom_id, r.image_path, out_dir, args.size) for r in img.itertuples()]

    results = {"ok": 0, "skip": 0}
    errors = []
    with Pool(args.workers) as pool:
        for i, (did, status) in enumerate(pool.imap_unordered(work, jobs, chunksize=16), 1):
            if status in results:
                results[status] += 1
            else:
                errors.append((did, status))
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)}장 처리")

    print(f"\n완료: 새로 저장 {results['ok']}장, 이미 있음 {results['skip']}장, 오류 {len(errors)}장")
    for did, msg in errors[:10]:
        print("  ", did, msg)
    print(f"저장 위치: {out_dir} (전체 {len(list(out_dir.glob('*.png')))}장)")


if __name__ == "__main__":
    main()
