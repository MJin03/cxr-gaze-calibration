"""11절: 판독 시선에서 영상별·finding별 "난이도"를 계산한다 (gaze 난이도 label smoothing용).

아이디어: 판독의가 그 finding을 말하는 동안 오래 보거나, 여러 번 다시 말하거나, 시선이 넓게 흩어진 영상은
판독의에게도 어려운 영상이다. 이런 영상에서 모델이 덜 확신하도록 학습한다.

시선 특징 (finding별 시선 지도와 같은 창: 단어 말하기 3초 전 ~ 말 끝, 같은 단어 세트 = fs_build_config.json)
    dwell       창 안에 머문 시간 합 (초)
    revisits    창 개수 (따로 떨어져 말한 횟수)
    dispersion  창 안 고정점 위치의 퍼짐 (머문 시간 가중 표준편차, 가로·세로를 영상 크기로 나눈 뒤)
난이도 d (0~1)
    1. 판독 단위로 특징을 구하고 영상 단위로 평균 (판독이 2개인 영상)
    2. 같은 (finding, label, 데이터셋) 묶음 안에서 각 특징을 백분위로 바꿈
       → label 자체와 데이터셋 차이(EGD는 녹취가 짧음)가 난이도에 섞이지 않게
    3. 세 백분위의 평균을 다시 백분위로 → d
    4. 그 finding을 말하지 않은 영상은 d = 0.5 (중립)
비교용 판독의 확신도 난이도 u (0~1, REFLACX만)
    양성(확신도 3~5): u = (5 − 확신도) / 2 → 3: 1, 4: 0.5, 5: 0
    음성(확신도 0~2): u = 확신도 / 2      → 0: 0, 1: 0.5, 2: 1   (양성 기준 3에 가까울수록 불확실)
    EGD만 있는 영상(0/1 label): u = 0.5 (중립)

출력  ~/X-ray/data/processed/gaze_difficulty.csv  (train·val: dicom_id, d_{finding}, u_{finding}, 원 특징)
확인  난이도 분포, d와 u의 상관, (있으면) 기존 모델의 val 오류와 d의 상관

사용법  python build_gaze_difficulty.py
"""
import json

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr

import build_fs_heatmaps as fsb
from vindr_common import PROC, run_dir

FINDINGS = ["pleural", "cardiac", "edema"]
WIN = 3.0


def features(fix, wins, w, h):
    dwell = fsb.dwell_in_windows(fix, wins)
    inside = (fix.x >= 0) & (fix.x < w) & (fix.y >= 0) & (fix.y < h) & (dwell > 0)
    if not inside.any():
        return np.nan, np.nan, np.nan
    x, y, wt = fix.x.values[inside] / w, fix.y.values[inside] / h, dwell[inside.values]
    mx, my = np.average(x, weights=wt), np.average(y, weights=wt)
    disp = np.sqrt(np.average((x - mx) ** 2 + (y - my) ** 2, weights=wt))
    return dwell.sum(), len(wins), disp


def pct(s):
    return s.rank(pct=True)


def main():
    cfg = json.load(open(PROC / "fs_build_config.json"))
    fsb.ROOTS.update(cfg["roots"])
    print("단어 세트:", {k: v for k, v in fsb.ROOTS.items()})
    reads = pd.read_csv(PROC / "cohort_readings.csv")
    split = pd.read_csv(PROC / "split_seed42.csv")[["dicom_id", "split"]]
    lab = pd.read_csv(PROC / "cohort_images_labeled.csv").merge(split, on="dicom_id")
    lab = lab[lab.split.isin(["train", "val"])].reset_index(drop=True)
    usable = reads[reads.gaze_ok & reads.dicom_id.isin(lab.dicom_id)]
    egd_file = reads.loc[reads.dataset == "egd", "gaze_file"].iloc[0]
    egd_groups = pd.read_csv(egd_file, usecols=["DICOM_ID", "X_ORIGINAL", "Y_ORIGINAL", "FPOGS", "FPOGD"]).groupby("DICOM_ID")

    rows = []
    sizes = {}
    for i, r in enumerate(usable.itertuples(), 1):
        if r.dicom_id not in sizes:
            sizes[r.dicom_id] = Image.open(r.image_path).size
        w, h = sizes[r.dicom_id]
        words, fix = fsb.load_words(r), fsb.load_fixations(r, egd_groups)
        row = {"dicom_id": r.dicom_id, "reading_id": r.reading_id, "dataset": r.dataset}
        for f in FINDINGS:
            wins = fsb.mention_windows(words, f, WIN)
            row[f"dwell_{f}"], row[f"revisits_{f}"], row[f"disp_{f}"] = features(fix, wins, w, h) if wins else (np.nan,) * 3
        rows.append(row)
        if i % 1000 == 0:
            print(f"  {i}/{len(usable)} 판독", flush=True)
    rd = pd.DataFrame(rows)
    img = rd.groupby("dicom_id")[[c for c in rd.columns if c.startswith(("dwell_", "revisits_", "disp_"))]].mean().reset_index()
    out = lab[["dicom_id", "subject_id", "split", "datasets"] + [f"{f}_t3" for f in FINDINGS]].merge(img, on="dicom_id", how="left")
    out["dgroup"] = np.where(out.datasets.str.contains("reflacx"), "reflacx", "egd")

    # 시선 난이도 d: (finding, label, 데이터셋) 묶음 안 백분위
    for f in FINDINGS:
        has = out[f"dwell_{f}"].notna()
        d = pd.Series(0.5, index=out.index)
        g = out[has].groupby([f"{f}_t3", "dgroup"])
        comb = g[f"dwell_{f}"].transform(lambda s: pct(np.log1p(s))) + g[f"revisits_{f}"].transform(pct) + g[f"disp_{f}"].transform(pct)
        d[has] = comb.groupby([out.loc[has, f"{f}_t3"], out.loc[has, "dgroup"]]).transform(pct)
        out[f"d_{f}"] = d.values

    # 판독의 확신도 난이도 u (REFLACX만)
    ref = reads[(reads.dataset == "reflacx") & reads.dicom_id.isin(out.dicom_id)]
    for f in FINDINGS:
        c = ref[f"{f}_raw"]
        u = np.where(c >= 3, (5 - c) / 2, c / 2)
        um = pd.Series(u, index=ref.index).groupby(ref.dicom_id).mean()
        out[f"u_{f}"] = out.dicom_id.map(um).fillna(0.5).clip(0, 1)
        out[f"cert_{f}"] = out.dicom_id.map(c.groupby(ref.dicom_id).mean())
    out.to_csv(PROC / "gaze_difficulty.csv", index=False)

    # ------------------------------------------------------ 확인
    pd.set_option("display.width", 200)
    tr = out[out.split == "train"]
    print("\n=== 시선 난이도를 계산할 수 있는 train 영상 비율 (그 finding을 말한 영상)")
    print({f: round(tr[f"dwell_{f}"].notna().mean(), 3) for f in FINDINGS})
    print("\n=== 판독의 확신도별 평균 시선 난이도 d (train, REFLACX, 그 finding을 말한 영상)")
    rows = []
    for f in FINDINGS:
        s = tr[tr[f"dwell_{f}"].notna() & tr[f"cert_{f}"].notna()]
        r = {"finding": f}
        for c in range(6):
            m = s[s[f"cert_{f}"].round() == c]
            r[f"확신도{c}"] = f"{m[f'd_{f}'].mean():.2f} (n={len(m)})" if len(m) else "-"
        rho_p = spearmanr(s.loc[s[f"{f}_t3"] == 1, f"d_{f}"], s.loc[s[f"{f}_t3"] == 1, f"u_{f}"])
        rho_n = spearmanr(s.loc[s[f"{f}_t3"] == 0, f"d_{f}"], s.loc[s[f"{f}_t3"] == 0, f"u_{f}"])
        r["d–u 상관 양성"] = f"{rho_p.statistic:+.2f} (p={rho_p.pvalue:.3f})"
        r["d–u 상관 음성"] = f"{rho_n.statistic:+.2f} (p={rho_n.pvalue:.3f})"
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))
    print("  ※ u가 클수록 판독의가 불확실. d와 u가 양의 상관이면 시선 난이도가 판독의의 불확실성을 반영한다는 뜻")

    # 기존 모델(C1 none, avg)의 val 오류와 d의 상관 (val만 사용)
    errs = []
    for s in range(10):
        d_ = run_dir("c1_none", s)
        if (d_ / "preds_val.csv").exists():
            errs.append(pd.read_csv(d_ / "preds_val.csv").set_index("dicom_id"))
    if errs:
        va = out[out.split == "val"].set_index("dicom_id")
        print(f"\n=== 기존 모델(C1 none, seed {len(errs)}개 평균)의 val 오류 |p − y|와 시선 난이도 d의 상관 (그 finding을 말한 val 영상)")
        for f in FINDINGS:
            p = np.mean([1 / (1 + np.exp(-e.loc[va.index, f"logit_{f}"])) for e in errs], axis=0)
            err = np.abs(p - va[f"{f}_t3"].values)
            m = va[f"dwell_{f}"].notna().values
            rho = spearmanr(va[f"d_{f}"].values[m], err[m])
            print(f"  {f:8s} n={int(m.sum())}  Spearman {rho.statistic:+.3f} (p={rho.pvalue:.3f})")
        print("  ※ 양의 상관이면 시선이 어려웠던 영상에서 모델도 더 틀린다는 뜻")
    print(f"\n저장: {PROC / 'gaze_difficulty.csv'}")


if __name__ == "__main__":
    main()
