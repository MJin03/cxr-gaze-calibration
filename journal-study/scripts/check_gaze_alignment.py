"""4-1단계: 시선 좌표가 영상 위에 제대로 맞는지 확인한다.

시선 지도를 만들기 전에, 두 데이터셋의 시선 좌표가 우리가 쓰는 JPG 영상의 픽셀 좌표와
같은 기준인지 확인한다. 좌표가 어긋나면 시선 지도 전체가 엉뚱한 위치를 가리키게 된다.

확인 내용
    1. 시선 파일의 열 이름 (좌표 열, 머문 시간 열)
    2. 표본 영상에서 시선 점이 영상 안쪽에 들어오는 비율 (100%에 가까워야 정상)
    3. 영상 위에 시선 점을 겹친 그림 6장 → ~/X-ray/data/processed/qc/

사용법  python check_gaze_alignment.py [--n 200]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

PROC = Path.home() / "X-ray/data/processed"
QC = PROC / "qc"
DUR_CANDIDATES = ["FPOGD"]


def inside_ratio(x: np.ndarray, y: np.ndarray, w: int, h: int) -> float:
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    return float(ok.mean()) if len(ok) else np.nan


def overlay(img_path: str, x, y, size, title: str, out: Path) -> None:
    im = Image.open(img_path)
    plt.figure(figsize=(6, 6))
    plt.imshow(im, cmap="gray")
    plt.scatter(x, y, s=size, c=np.arange(len(x)), cmap="autumn", alpha=0.6)
    plt.title(title, fontsize=9)
    plt.axis("off")
    plt.savefig(out, dpi=80, bbox_inches="tight")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    QC.mkdir(parents=True, exist_ok=True)
    reads = pd.read_csv(PROC / "cohort_readings.csv")
    reads = reads[reads.gaze_ok & reads.image_ok]
    rng = np.random.default_rng(0)

    # EGD-CXR
    egd_fix_path = reads.loc[reads.dataset == "egd", "gaze_file"].iloc[0]
    egd_fix = pd.read_csv(egd_fix_path)
    print("=== EGD-CXR fixations.csv 열:", list(egd_fix.columns))
    dur_col = next((c for c in DUR_CANDIDATES if c in egd_fix.columns), None)
    print("  좌표 열: X_ORIGINAL, Y_ORIGINAL / 머문 시간 열:", dur_col)
    egd = reads[reads.dataset == "egd"].sample(min(args.n, (reads.dataset == "egd").sum()), random_state=0)
    ratios, n_fix, sizes = [], [], []
    by_id = egd_fix.groupby("DICOM_ID")
    for k, r in enumerate(egd.itertuples()):
        f = by_id.get_group(r.dicom_id)
        w, h = Image.open(r.image_path).size
        ratios.append(inside_ratio(f.X_ORIGINAL.values, f.Y_ORIGINAL.values, w, h))
        n_fix.append(len(f)); sizes.append((w, h))
        if k < 3:
            d = f[dur_col].values if dur_col else np.ones(len(f)) * 0.3
            overlay(r.image_path, f.X_ORIGINAL, f.Y_ORIGINAL, d * 300,
                    f"EGD {r.dicom_id[:12]} | {w}x{h}", QC / f"gaze_overlay_egd_{k + 1}.png")
    print(f"  표본 {len(egd)}장: 영상 안쪽 비율 평균 {np.nanmean(ratios):.1%}, 최소 {np.nanmin(ratios):.1%}")
    print(f"  영상당 고정점 수: 중앙값 {int(np.median(n_fix))}, 범위 {min(n_fix)}~{max(n_fix)}")
    print(f"  영상 크기 예: {sizes[:3]}")

    # REFLACX
    ref = reads[reads.dataset == "reflacx"]
    ref = ref.sample(min(args.n, len(ref)), random_state=0)
    first = pd.read_csv(ref.gaze_file.iloc[0])
    print("\n=== REFLACX fixations.csv 열:", list(first.columns))
    print("  좌표 열: x_position, y_position / 머문 시간: timestamp_end_fixation - timestamp_start_fixation")
    ratios, n_fix = [], []
    for k, r in enumerate(ref.itertuples()):
        f = pd.read_csv(r.gaze_file)
        w, h = Image.open(r.image_path).size
        ratios.append(inside_ratio(f.x_position.values, f.y_position.values, w, h))
        n_fix.append(len(f))
        if k < 3:
            d = (f.timestamp_end_fixation - f.timestamp_start_fixation).values
            overlay(r.image_path, f.x_position, f.y_position, d * 300,
                    f"REFLACX {r.reading_id} | {w}x{h}", QC / f"gaze_overlay_reflacx_{k + 1}.png")
    print(f"  표본 {len(ref)}건: 영상 안쪽 비율 평균 {np.nanmean(ratios):.1%}, 최소 {np.nanmin(ratios):.1%}")
    print(f"  판독당 고정점 수: 중앙값 {int(np.median(n_fix))}, 범위 {min(n_fix)}~{max(n_fix)}")

    print(f"\n겹침 그림 6장 저장: {QC}")


if __name__ == "__main__":
    main()
