"""C3단계 확인: finding별 시선 지도가 실제로 그 finding의 위치를 가리키는지 확인한다.

확인 1. 병변 타원과 겹침
    타원(판독의가 그린 병변 위치)과 finding별 시선 지도가 둘 다 있는 train·val 영상에서
    "시선 질량 중 타원 안에 들어간 비율"을 finding별 지도와 전체 시선 지도(기존)로 비교한다.
    finding별 지도가 더 높으면, 단어 시간으로 자른 시선이 그 finding의 위치를 더 잘 가리킨다는 뜻.
확인 2. 평균 지도의 중심 위치
    finding별 평균 지도의 무게중심(가로·세로, 0~1). 심장비대는 가운데, 흉막은 아래쪽 바깥에 있어야 자연스럽다.
그림  ~/X-ray/data/processed/qc/fs_mean_maps.png

사용법  python check_fs_maps.py [--windows 1.5 3 5]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROC = Path.home() / "X-ray/data/processed"
FINDINGS = ["pleural", "cardiac", "edema"]
OUT = 128


def pool(hm):
    k = hm.shape[-1] // OUT
    return hm.reshape(OUT, k, OUT, k).sum(axis=(1, 3))


def centroid(m):
    m = m / m.sum()
    yy, xx = np.mgrid[0:OUT, 0:OUT] + 0.5
    return (m * xx).sum() / OUT, (m * yy).sum() / OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=float, nargs="*", default=[1.5, 3.0, 5.0])
    args = ap.parse_args()
    idx = pd.read_csv(PROC / "fs_heatmap_index.csv")
    glob_dir, ell_dir = PROC / "heatmaps_512_s7", PROC / f"heatmaps_ellipse_{OUT}"

    print("=== 확인 1. 타원 안에 들어간 시선 질량 비율 (타원과 finding별 지도가 둘 다 있는 영상, 평균)")
    rows = []
    for k, f in enumerate(FINDINGS):
        both = idx[idx[f"ellipse_{f}"] & idx["fs3_" + f]]
        r = {"finding": f, "영상": len(both)}
        g_in, w_in = [], {w: [] for w in args.windows}
        for d in both.dicom_id:
            ell = np.load(ell_dir / f"{d}.npy").astype(float)[k] > 0
            g = pool(np.load(glob_dir / f"{d}.npy").astype(float))
            g_in.append(g[ell].sum() / g.sum())
            for w in args.windows:
                p = PROC / f"heatmaps_fs_w{w:g}_{OUT}" / f"{d}.npy"
                m = np.load(p).astype(float)[k]
                w_in[w].append(m[ell].sum() / m.sum() if m.sum() > 0 else np.nan)
        r["전체 시선"] = np.mean(g_in) if g_in else np.nan
        for w in args.windows:
            a = np.array(w_in[w])
            r[f"finding별 {w:g}초"] = np.nanmean(a) if len(a) else np.nan
            r[f"{w:g}초가 더 높은 영상 비율"] = np.nanmean(a > np.array(g_in)) if len(a) else np.nan
        rows.append(r)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n=== 확인 2. 평균 지도의 무게중심 (x: 0=영상 왼쪽 끝, 1=오른쪽 끝 / y: 0=위, 1=아래), train·val")
    means, cents = {}, []
    sub = idx.sample(min(len(idx), 1500), random_state=0)
    means["global"] = np.mean([pool(np.load(glob_dir / f"{d}.npy").astype(float)) for d in sub.dicom_id], axis=0)
    for k, f in enumerate(FINDINGS):
        ids = idx.loc[idx["fs3_" + f], "dicom_id"]
        ids = ids.sample(min(len(ids), 1500), random_state=0)
        means[f] = np.mean([np.load(PROC / f"heatmaps_fs_w3_{OUT}" / f"{d}.npy").astype(float)[k] for d in ids], axis=0)
    for name, m in means.items():
        x, y = centroid(m)
        cents.append({"지도": name, "x": x, "y": y})
    print(pd.DataFrame(cents).round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3))
    for ax, (name, m) in zip(axes, means.items()):
        ax.imshow(m, cmap="jet")
        ax.set_title(f"mean {name} (3s)" if name != "global" else "mean global gaze", fontsize=9)
        ax.axis("off")
    (PROC / "qc").mkdir(exist_ok=True)
    fig.savefig(PROC / "qc/fs_mean_maps.png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"\n그림: {PROC / 'qc/fs_mean_maps.png'}")


if __name__ == "__main__":
    main()
