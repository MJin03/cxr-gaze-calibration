"""22절 평가: 시선 출처별 (ANALYSIS_PLAN.md 22절, 학습 전에 고정한 규칙 그대로).

조건: hyb_none (시선 없음), fs_src_egd (EGD 판독의 1명의 시선만), fs_src_reflacx (REFLACX 판독의 5명의 시선만), fs (전체)
비교 기준: 17절 무작위 비율 곡선 (0·25·50·75·100%)을 각 출처의 시선 비율 f에서 선형 보간한 AUROC
   f = train 영상 중 그 출처 영상의 비율 (cohort_images_labeled.csv의 datasets 열, egd+reflacx 영상은 제외)
지표: 외부 7곳 AUROC (seed 평균), 환자 + seed bootstrap (ci2), random-effects 통합
판정
  R1: fs_src_egd − hyb_none, fs_src_reflacx − hyb_none의 통합 CI 하한 > 0 (흉막 이상·심장비대·폐부종)
      → 두 판독의 집단 모두에서 효과가 나면 "특정 판독의 집단의 시선에 의존하지 않음"
  R2 (보조): 출처 − 같은 비율 무작위 시선(곡선 보간). 0에 가까우면 "출처 차이는 시선 양으로 설명됨"
  주의: 출처마다 영상 종류도 다름 (EGD는 심부전·폐렴·정상 선별 영상) → 판독의와 영상 차이가 섞임

사용법  python evaluate_gazesource.py [--n-boot 1000]
출력    results/gazesource/ (auroc.csv, comparisons.csv)
"""
import argparse

import numpy as np
import pandas as pd

from cal_common import DATASETS, PROC, RESULTS, S10, run_dir
from evaluate_calibration import AUC
from evaluate_discrimination import dersimonian_laird

OUT = RESULTS / "gazesource"
DS7 = ["chexpert_test", "vindr_train", "vindr_train_b", "chexpert_valid", "vindr_test", "padchest_a", "padchest_b"]
CURVE = [("hyb_none", 0.0), ("fs_gf25", 0.25), ("fs_gf50", 0.5), ("fs_gf75", 0.75), ("fs", 1.0)]
SRC = {"fs_src_egd": "egd", "fs_src_reflacx": "reflacx"}


def source_fracs():
    lab = pd.read_csv(PROC / "cohort_images_labeled.csv")[["dicom_id", "datasets"]]
    sp = pd.read_csv(PROC / "split_seed42.csv")
    tr = lab.merge(sp, on="dicom_id")
    tr = tr[tr.split == "train"]
    return {c: float((tr.datasets == s).mean()) for c, s in SRC.items()}


def scores(c, s, ds, m, ids):
    t = pd.read_csv(run_dir(c, s) / f"preds_{ds}.csv").set_index("image_id").loc[ids]
    return t[f"logit_{m}"].values.astype(float)


def interp(vals, f):
    xs = np.array([x for _, x in CURVE])
    return float(np.interp(f, xs, np.array([vals[c] for c, _ in CURVE])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DS7)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    fr = source_fracs()
    print("train 영상 중 출처 비율 f: " + ", ".join(f"{c} {v:.1%}" for c, v in fr.items()), flush=True)
    conds = [c for c, _ in CURVE] + list(SRC)
    rows, comps = [], []
    names = [f"{c}-hyb_none" for c in SRC] + [f"{c}-무작위{fr[c]:.0%}" for c in SRC] + ["fs_src_egd-fs_src_reflacx"]
    for ds in args.datasets:
        D = DATASETS[ds]
        lab = pd.read_csv(D["labels"])
        groups = lab["group"].values if "group" in lab.columns else np.arange(len(lab))
        ug, gi = np.unique(groups, return_inverse=True)
        seeds = {c: [s for s in S10 if (run_dir(c, s) / f"preds_{ds}.csv").exists()] for c in conds}
        miss = [c for c in conds if not seeds[c]]
        if miss:
            print(f"[{ds}] 예측 없는 조건이 있어 건너뜀: {miss}")
            continue
        for f, m in D["findings"].items():
            y = lab[f].values.astype(float)
            aucs = {(c, s): AUC(scores(c, s, ds, m, lab.image_id), y) for c in conds for s in seeds[c]}
            one = np.ones(len(y))
            pt = {c: np.mean([aucs[(c, s)](one) for s in seeds[c]]) for c in conds}
            for c in conds:
                rows.append({"dataset": ds, "finding": f, "cond": c, "auroc": pt[c]})

            def contrasts(v):
                return ([v[c] - v["hyb_none"] for c in SRC] + [v[c] - interp(v, fr[c]) for c in SRC]
                        + [v["fs_src_egd"] - v["fs_src_reflacx"]])
            point = contrasts(pt)
            boot = []
            for _ in range(args.n_boot):
                w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[gi].astype(float)
                val = {}
                for c in conds:
                    pick = rng.choice(seeds[c], len(seeds[c]))
                    cache = {s: aucs[(c, s)](w) for s in set(pick)}
                    val[c] = np.mean([cache[s] for s in pick])
                boot.append(contrasts(val))
            boot = np.array(boot)
            for j, nm in enumerate(names):
                comps.append({"dataset": ds, "finding": f, "comparison": nm, "estimate": point[j],
                              "low": np.nanquantile(boot[:, j], 0.025), "high": np.nanquantile(boot[:, j], 0.975),
                              "se": np.nanstd(boot[:, j])})
            print(f"  [{ds}] {f} bootstrap {args.n_boot}회 완료", flush=True)
    r, c = pd.DataFrame(rows), pd.DataFrame(comps)
    OUT.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT / "auroc.csv", index=False)
    c.to_csv(OUT / "comparisons.csv", index=False)
    pd.set_option("display.width", 230)
    print("\n=== AUROC (seed 평균)")
    print(r.pivot_table(index=["finding", "dataset"], columns="cond", values="auroc")[conds].round(3).to_string())
    print("\n=== 통합 (random-effects) [95% CI]. ✓ = CI가 0 미포함")
    for f, g in c.groupby("finding", sort=False):
        print(f"\n[{f}] (데이터 {g.dataset.nunique()}곳)")
        res = {}
        for nm, h in g.groupby("comparison", sort=False):
            mu, lo, hi, _, i2 = dersimonian_laird(h.estimate.values, np.maximum(h.se.values, 1e-6))
            res[nm] = (mu, lo, hi)
            print(f"  {nm:28s} {mu:+.4f} [{lo:+.4f}, {hi:+.4f}]{'✓' if lo > 0 or hi < 0 else ''} (I² {100 * i2:.0f}%)")
        r1 = all(res[f"{s}-hyb_none"][1] > 0 for s in SRC)
        print(f"  R1: {'충족 (두 판독의 집단 모두에서 시선 효과)' if r1 else '불충족'}")
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
