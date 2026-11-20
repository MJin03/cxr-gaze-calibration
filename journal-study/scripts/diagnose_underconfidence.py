"""사후 진단 (2026-11-20): 외부 데이터에서 모든 모델이 확신 부족(calibration slope > 1)인 원인 확인. 학습·예측 없음.

이미 만든 예측(preds_test.csv, preds_{dataset}.csv)만 사용한다. 개선 주장이 아니라 원인 진단용.

진단
  [1] 데이터별 slope: 내부 test vs CheXpert test vs VinDr train (temperature 없는 값, seed 평균 ± SD)
      → 내부도 1 근처가 아니면 병원 변화만의 문제가 아님
  [2] label 정확도 가설 (VinDr train, 판독의 3명):
      a. 같은 예측을 "과반수 label"과 "판독의 비율 label(0, 1/3, 2/3, 1)"에 각각 맞춰 slope 비교
         모델은 대부분 판독의 1명의 label로 학습했으므로, 비율 label 기준 slope가 1에 가깝고 과반수 기준이 1보다 크면
         "학습 label이 외부 합의 label보다 애매해서 생긴 확신 부족"을 지지
      b. 만장일치 영상(0/3, 3/3)과 의견이 갈린 영상(1/3, 2/3)의 비율과 각 영상군 slope
  [3] 내부 test 중 판독의 5명 영상: t3 과반수 label vs 판독의 비율 label slope (영상 수가 적어 참고용)
  [4] 앙상블 방식: 확률 평균 vs logit 평균의 slope (seed 평균과 비교)

출력  ~/X-ray/results/diag_underconf/*.csv
사용법  python diagnose_underconfidence.py [--conds c1_none fs gaze uniform a_gaze a_free]
"""
import argparse

import numpy as np
import pandas as pd

from cal_common import CONDITIONS, DATASETS, PROC, RESULTS, S10, run_dir
from evaluate_calibration import logit, sig, slope_fit
from vindr_common import VINDR

VINDR_COLS = {"pleural": ["Pleural effusion", "Pleural thickening"], "pleural_effusion": ["Pleural effusion"],
              "cardiac": ["Cardiomegaly"]}
INTERNAL_F = ["pleural", "cardiac", "edema"]


def slope(z, y):
    return slope_fit(z, y, np.ones(len(y)))[1]


def load_ext(dataset, conds):
    ds = DATASETS[dataset]
    lab = pd.read_csv(ds["labels"])
    out = {}
    for c in conds:
        for s in S10:
            f = run_dir(c, s) / f"preds_{dataset}.csv"
            if f.exists():
                t = pd.read_csv(f).set_index("image_id").loc[lab.image_id]
                out[(c, s)] = {k: t[f"logit_{m}"].values.astype(float) for k, m in ds["findings"].items()}
    return lab, out


def load_internal(conds):
    out, lab = {}, None
    for c in conds:
        for s in S10:
            f = run_dir(c, s) / "preds_test.csv"
            if f.exists():
                t = pd.read_csv(f).sort_values("dicom_id").reset_index(drop=True)
                lab = t if lab is None else lab
                out[(c, s)] = {k: t[f"logit_{k}"].values.astype(float) for k in INTERNAL_F}
    return lab, out


def seed_stat(pred, c, fn):
    v = [fn(pred[k]) for k in pred if k[0] == c]
    return (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else np.nan, len(v)) if v else (np.nan, np.nan, 0)


def fmt(m, sd):
    return f"{m:.3f}±{sd:.3f}" if np.isfinite(sd) else f"{m:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", nargs="*", default=["c1_none", "fs", "gaze", "uniform", "a_gaze", "a_free"])
    args = ap.parse_args()
    conds = [c for c in args.conds if c in CONDITIONS]
    out = RESULTS / "diag_underconf"
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    ilab, ipred = load_internal(conds)
    data = {"internal": (ilab, ipred, {f: f"{f}_t3" for f in INTERNAL_F})}
    for d in ["chexpert_test", "vindr_train"]:
        lab, pred = load_ext(d, conds)
        data[d] = (lab, pred, {f: f for f in DATASETS[d]["findings"]})

    # [1] 데이터별 slope
    rows = []
    for d, (lab, pred, fmap) in data.items():
        for f, col in fmap.items():
            y = lab[col].values.astype(float)
            for c in conds:
                m, sd, n = seed_stat(pred, c, lambda p: slope(p[f], y))
                rows.append({"data": d, "finding": f, "cond": c, "n_seed": n, "slope": m, "slope_sd": sd, "n_img": len(y), "pos": int(y.sum())})
    r1 = pd.DataFrame(rows)
    r1.to_csv(out / "1_slope_by_dataset.csv", index=False)
    print("=== [1] 데이터별 calibration slope (temperature 없음, seed 평균 ± SD). 1 = 이상적, > 1 = 확신 부족")
    tab = r1.assign(v=[fmt(a, b) for a, b in zip(r1.slope, r1.slope_sd)]).pivot_table(index=["finding", "cond"], columns="data", values="v", aggfunc="first")
    print(tab[[c for c in ["internal", "chexpert_test", "vindr_train"] if c in tab.columns]].to_string())

    # [2] VinDr 판독의 비율 label
    lab, pred, _ = data["vindr_train"]
    ann = pd.read_csv(VINDR / "annotations/image_labels_train.csv")
    ann = ann[ann.image_id.isin(lab.image_id)]
    rows, rows_b = [], []
    for f, cols in VINDR_COLS.items():
        pos = (ann[cols].sum(axis=1) > 0).astype(float)
        frac = pos.groupby(ann.image_id).mean().loc[lab.image_id].values
        maj = lab[f].values.astype(float)
        unan = (frac == 0) | (frac == 1)
        for c in conds:
            ms, ss, n = seed_stat(pred, c, lambda p: slope(p[f], maj))
            mf, sf, _ = seed_stat(pred, c, lambda p: slope(p[f], frac))
            mu, su, _ = seed_stat(pred, c, lambda p: slope(p[f][unan], maj[unan]))
            mk, sk, _ = seed_stat(pred, c, lambda p: slope(p[f][~unan], maj[~unan]) if (~unan).sum() >= 10 and 0 < maj[~unan].sum() < (~unan).sum() else np.nan)
            rows.append({"finding": f, "cond": c, "slope_majority": ms, "sd_majority": ss, "slope_reader_frac": mf, "sd_reader_frac": sf,
                         "slope_unanimous": mu, "sd_unanimous": su, "slope_split": mk, "sd_split": sk})
        rows_b.append({"finding": f, "영상": len(frac), "만장일치 음성(0/3)": int((frac == 0).sum()), "1/3 양성": int(np.isclose(frac, 1 / 3).sum()),
                       "2/3 양성": int(np.isclose(frac, 2 / 3).sum()), "만장일치 양성(3/3)": int((frac == 1).sum()),
                       "과반수 양성 중 의견 갈림 비율": f"{np.isclose(frac, 2 / 3).sum() / max((frac > 0.5).sum(), 1):.2f}"})
    r2 = pd.DataFrame(rows)
    r2.to_csv(out / "2_vindr_reader_agreement.csv", index=False)
    print("\n=== [2a] VinDr train: 같은 예측의 slope — 과반수 label 기준 vs 판독의 비율 label 기준 (seed 평균 ± SD)")
    print(pd.DataFrame({"finding": r2.finding, "조건": r2.cond, "과반수 label": [fmt(a, b) for a, b in zip(r2.slope_majority, r2.sd_majority)],
                        "판독의 비율 label": [fmt(a, b) for a, b in zip(r2.slope_reader_frac, r2.sd_reader_frac)]}).to_string(index=False))
    print("\n=== [2b] VinDr train: 판독의 일치도별 영상 수")
    print(pd.DataFrame(rows_b).to_string(index=False))
    print("\n=== [2b] 일치도별 slope (과반수 label, seed 평균 ± SD). 의견 갈린 영상은 수가 적어 불안정할 수 있음")
    print(pd.DataFrame({"finding": r2.finding, "조건": r2.cond, "만장일치 영상": [fmt(a, b) for a, b in zip(r2.slope_unanimous, r2.sd_unanimous)],
                        "의견 갈린 영상": [fmt(a, b) for a, b in zip(r2.slope_split, r2.sd_split)]}).to_string(index=False))

    # [3] 내부 test 판독의 5명 영상
    coh = pd.read_csv(PROC / "cohort_images_labeled.csv").set_index("dicom_id")
    m5 = coh.loc[ilab.dicom_id, "n_readings"].values >= 3
    rows = []
    print(f"\n=== [3] 내부 test 중 판독의 여러 명 영상 {int(m5.sum())}장: t3 과반수 label vs 판독의 비율 label (참고용)")
    if m5.sum() >= 10:
        for f in INTERNAL_F:
            yh = ilab[f"{f}_t3"].values[m5].astype(float)
            ys = coh.loc[ilab.dicom_id, f"{f}_reader_frac"].values[m5].astype(float)
            if not (0 < yh.sum() < len(yh)):
                continue
            for c in conds:
                mh, sh, _ = seed_stat(ipred, c, lambda p: slope(p[f][m5], yh))
                ms_, ss_, _ = seed_stat(ipred, c, lambda p: slope(p[f][m5], ys))
                rows.append({"finding": f, "조건": c, "양성": int(yh.sum()), "과반수 label": fmt(mh, sh), "판독의 비율 label": fmt(ms_, ss_)})
        r3 = pd.DataFrame(rows)
        r3.to_csv(out / "3_internal_multireader.csv", index=False)
        print(r3.to_string(index=False) if len(r3) else "  양성·음성이 모두 있는 finding이 없음")
    else:
        print("  10장 미만이라 생략")

    # [4] 앙상블 방식
    rows = []
    for d, (lab, pred, fmap) in data.items():
        for f, col in fmap.items():
            y = lab[col].values.astype(float)
            for c in conds:
                zs = [pred[k][f] for k in pred if k[0] == c]
                if len(zs) < 2:
                    continue
                m, sd, _ = seed_stat(pred, c, lambda p: slope(p[f], y))
                rows.append({"data": d, "finding": f, "조건": c, "seed 평균": fmt(m, sd),
                             "확률 평균 앙상블": f"{slope(logit(np.mean([sig(z) for z in zs], 0)), y):.3f}",
                             "logit 평균 앙상블": f"{slope(np.mean(zs, 0), y):.3f}"})
    r4 = pd.DataFrame(rows)
    r4.to_csv(out / "4_ensemble.csv", index=False)
    print("\n=== [4] 앙상블 방식별 slope")
    print(r4.to_string(index=False))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
