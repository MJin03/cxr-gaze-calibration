"""17절 평가: 시선 양 학습 곡선 (ANALYSIS_PLAN.md 17절, 학습 전에 고정한 규칙 그대로).

곡선의 점: 0% = hyb_none (같은 구조, 시선 없음), 25·50·75% = fs_gf25·50·75, 100% = fs. 참고로 c1_none(avg, 시선 없음)도 표시
지표: 외부 데이터별 AUROC (seed 평균). 신뢰구간 = 환자(또는 영상) + seed bootstrap (ci2), 데이터끼리 random-effects 통합
판정
  G1 양-반응: AUROC를 시선 비율(0~1)에 회귀한 기울기의 통합 추정 CI가 0보다 크면 "시선이 많을수록 판별력 증가"
  G2 포화: fs(100%) − fs_gf50의 통합 차이 CI가 0보다 크면 "50%에서 포화되지 않음 → 시선을 더 모으면 이득 예상",
     아니면 "절반의 시선으로 대부분 도달"
  보조: 각 비율 − 0%의 통합 이득, 100% 이득 대비 비율

사용법  python evaluate_gazefrac.py [--n-boot 1000] [--datasets ...]
출력    results/gazefrac/curve.csv, comparisons.csv
"""
import argparse

import numpy as np
import pandas as pd

from cal_common import DATASETS, RESULTS, S10, run_dir
from evaluate_calibration import AUC
from evaluate_discrimination import dersimonian_laird

OUT = RESULTS / "gazefrac"
DS7 = ["chexpert_test", "vindr_train", "vindr_train_b", "chexpert_valid", "vindr_test", "padchest_a", "padchest_b"]
CURVE = [("hyb_none", 0.0), ("fs_gf25", 0.25), ("fs_gf50", 0.5), ("fs_gf75", 0.75), ("fs", 1.0)]
X = np.array([x for _, x in CURVE])


def scores(cond, seed, ds, m, ids):
    t = pd.read_csv(run_dir(cond, seed) / f"preds_{ds}.csv").set_index("image_id").loc[ids]
    return t[f"logit_{m}"].values.astype(float)


def slope(v):
    xc = X - X.mean()
    return float((xc * (v - v.mean())).sum() / (xc ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DS7)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    conds = [c for c, _ in CURVE] + ["c1_none"]
    curve, comp = [], []
    for ds in args.datasets:
        D = DATASETS[ds]
        lab = pd.read_csv(D["labels"])
        groups = lab["group"].values if "group" in lab.columns else np.arange(len(lab))
        ug, gi = np.unique(groups, return_inverse=True)
        seeds = {c: [s for s in S10 if (run_dir(c, s) / f"preds_{ds}.csv").exists()] for c in conds}
        if any(not seeds[c] for c, _ in CURVE):
            print(f"[{ds}] 예측 없는 조건이 있어 건너뜀: {[c for c, _ in CURVE if not seeds[c]]}")
            continue
        for f, m in D["findings"].items():
            y = lab[f].values.astype(float)
            aucs = {(c, s): AUC(scores(c, s, ds, m, lab.image_id), y) for c in conds for s in seeds[c]}
            one = np.ones(len(y))
            pt = {c: np.mean([aucs[(c, s)](one) for s in seeds[c]]) for c in conds if seeds[c]}
            for c, x in CURVE + [("c1_none", np.nan)]:
                if c in pt:
                    curve.append({"dataset": ds, "finding": f, "cond": c, "gaze_frac": x, "auroc": pt[c], "n_seed": len(seeds[c])})
            v_pt = np.array([pt[c] for c, _ in CURVE])
            names = ["slope"] + [f"{c}-hyb_none" for c, _ in CURVE[1:]] + ["fs-fs_gf50"]
            point = [slope(v_pt)] + [pt[c] - pt["hyb_none"] for c, _ in CURVE[1:]] + [pt["fs"] - pt["fs_gf50"]]
            boot = []
            for _ in range(args.n_boot):
                w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[gi].astype(float)
                val = {}
                for c, _ in CURVE:
                    pick = rng.choice(seeds[c], len(seeds[c]))
                    cache = {s: aucs[(c, s)](w) for s in set(pick)}
                    val[c] = np.mean([cache[s] for s in pick])
                v = np.array([val[c] for c, _ in CURVE])
                boot.append([slope(v)] + [val[c] - val["hyb_none"] for c, _ in CURVE[1:]] + [val["fs"] - val["fs_gf50"]])
            boot = np.array(boot)
            for j, nm in enumerate(names):
                comp.append({"dataset": ds, "finding": f, "comparison": nm, "estimate": point[j],
                             "low": np.nanquantile(boot[:, j], 0.025), "high": np.nanquantile(boot[:, j], 0.975),
                             "se": np.nanstd(boot[:, j]), "pos": int(y.sum()), "n_img": len(y)})
            print(f"  [{ds}] {f} bootstrap {args.n_boot}회 완료", flush=True)
    cu, co = pd.DataFrame(curve), pd.DataFrame(comp)
    OUT.mkdir(parents=True, exist_ok=True)
    cu.to_csv(OUT / "curve.csv", index=False)
    co.to_csv(OUT / "comparisons.csv", index=False)
    pd.set_option("display.width", 250)

    print("\n=== 시선 양별 AUROC (seed 평균). 0% = hyb_none, 100% = fs, 참고 c1_none")
    tab = cu.assign(col=cu.cond.map({c: f"{int(x * 100)}%" for c, x in CURVE} | {"c1_none": "c1_none"})) \
        .pivot_table(index=["finding", "dataset"], columns="col", values="auroc")
    print(tab[[f"{int(x * 100)}%" for _, x in CURVE] + ["c1_none"]].round(3).to_string())

    print("\n=== 통합 (random-effects). 기울기 = 시선 0%→100%일 때 AUROC 변화. ✓ = CI가 0 미포함")
    for f, g in co.groupby("finding", sort=False):
        print(f"\n[{f}] (데이터 {g.dataset.nunique()}곳)")
        res = {}
        for nm, h in g.groupby("comparison", sort=False):
            mu, lo, hi, _, i2 = dersimonian_laird(h.estimate.values, np.maximum(h.se.values, 1e-6))
            res[nm] = (mu, lo, hi)
            print(f"  {nm:20s} {mu:+.4f} [{lo:+.4f}, {hi:+.4f}]{'✓' if lo > 0 or hi < 0 else ''} (I² {100 * i2:.0f}%)")
        full = res["fs-hyb_none"][0]
        if res["fs-hyb_none"][1] > 0:  # 100% 이득이 유의할 때만 비율 표시
            print("  100% 이득 대비: " + ", ".join(f"{c.replace('fs_gf', '')}% {res[f'{c}-hyb_none'][0] / full:.0%}" for c, _ in CURVE[1:-1]))
        g1 = res["slope"][1] > 0
        g2 = res["fs-fs_gf50"][1] > 0
        print(f"  G1 양-반응: {'충족 (시선이 많을수록 판별력 증가)' if g1 else '불충족'}")
        print(f"  G2 포화: {'50%에서 포화 안 됨 → 시선을 더 모으면 이득 예상' if g2 else '절반의 시선으로 대부분 도달 (100% − 50% 유의하지 않음)'}")
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
