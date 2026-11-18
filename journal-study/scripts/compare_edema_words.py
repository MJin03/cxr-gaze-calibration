"""C3단계 보완: 폐부종 단어 세트(v1, v2, v3)를 train·val 자료만으로 비교한다 (학습 결과는 보지 않음).

비교하는 것 (주 창 3초)
    1. coverage: 폐부종 지도가 생기는 영상 비율 (train, 전체·양성·음성)
    2. 위치: REFLACX 폐부종 타원이 있는 영상에서 타원 안에 들어간 시선 질량 비율
       (세트별 폐부종 지도 vs 기존 전체 시선 지도). 세트를 넓혀도 이 값이 전체 시선보다 높게 유지되어야 한다
    3. 음성 영상에서 가장 많이 걸린 단어 (어떤 말 때문에 coverage가 늘었는지)

사용법  python compare_edema_words.py
"""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from build_fs_heatmaps import (EDEMA_SETS, PROC, dwell_in_windows, gaze_map, load_fixations, load_words,
                               mention_windows, pool_to_out)

WIN = 3.0


def main():
    reads = pd.read_csv(PROC / "cohort_readings.csv")
    idx = pd.read_csv(PROC / "fs_heatmap_index.csv")  # train·val, 라벨 포함
    usable = reads[reads.gaze_ok & reads.dicom_id.isin(idx.dicom_id)]
    egd_file = reads.loc[reads.dataset == "egd", "gaze_file"].iloc[0]
    egd_groups = pd.read_csv(egd_file, usecols=["DICOM_ID", "X_ORIGINAL", "Y_ORIGINAL", "FPOGS", "FPOGD"]).groupby("DICOM_ID")
    ell_ids = set(idx.loc[idx.ellipse_edema, "dicom_id"])
    label = idx.set_index("dicom_id")["edema_t3"]

    has = {v: set() for v in EDEMA_SETS}
    neg_words = {v: Counter() for v in EDEMA_SETS}
    inside = {v: [] for v in EDEMA_SETS}
    glob_inside = []
    sizes = {}
    for i, r in enumerate(usable.itertuples(), 1):
        words = load_words(r)
        need_map = r.dicom_id in ell_ids
        fix = load_fixations(r, egd_groups) if need_map else None
        if need_map and r.dicom_id not in sizes:
            sizes[r.dicom_id] = Image.open(r.image_path).size
        for v, roots in EDEMA_SETS.items():
            wins = mention_windows(words, "edema", WIN, roots)
            if not wins:
                continue
            has[v].add(r.dicom_id)
            if label.get(r.dicom_id) == 0:
                neg_words[v].update(w for w in words.word if any(w.startswith(x) for x in roots))
            if need_map:
                w, h = sizes[r.dicom_id]
                m = gaze_map(fix, dwell_in_windows(fix, wins), w, h)
                if m is not None:
                    ell = np.load(PROC / "heatmaps_ellipse_128" / f"{r.dicom_id}.npy").astype(float)[2] > 0
                    inside[v].append((r.dicom_id, m[ell].sum()))
        if i % 1000 == 0:
            print(f"  {i}/{len(usable)} 판독", flush=True)

    for d in ell_ids:
        ell = np.load(PROC / "heatmaps_ellipse_128" / f"{d}.npy").astype(float)[2] > 0
        g = pool_to_out(np.load(PROC / "heatmaps_512_s7" / f"{d}.npy").astype(float))
        glob_inside.append(g[ell].sum() / g.sum())

    tr = idx[idx.split == "train"]
    y = tr.edema_t3 == 1
    rows = []
    for v in EDEMA_SETS:
        c = tr.dicom_id.isin(has[v])
        ins = [x for _, x in inside[v]]
        rows.append({"세트": v, "coverage 전체": c.mean(), "양성": c[y].mean(), "음성": c[~y].mean(),
                     "타원 영상 수": len({d for d, _ in inside[v]}), "타원 안 비율(세트)": np.mean(ins) if ins else np.nan,
                     "타원 안 비율(전체 시선)": np.mean(glob_inside) if glob_inside else np.nan})
    pd.set_option("display.width", 200)
    print("\n=== 폐부종 단어 세트 비교 (train coverage, 3초 창)")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("  ※ 타원 안 비율(세트)은 판독 단위 평균, 전체 시선은 영상 단위 평균 (참고용 비교)")
    print("\n=== 폐부종 음성 영상에서 가장 많이 걸린 단어 (판독 단위 횟수)")
    for v in EDEMA_SETS:
        print(f"  {v}: {neg_words[v].most_common(12)}")


if __name__ == "__main__":
    main()
