"""21절: 임상 지표와 다중 비교 보정 (ANALYSIS_PLAN.md 21절, 실행 전에 고정). 학습 없음, CPU만.

① 임상 지표 (외부 7곳, seed 평균, 환자 + seed bootstrap ci2, random-effects 통합)
   sens@spec90 : 특이도 90%에서의 민감도 (음성 영상의 90%를 정상으로 통과시키는 문턱에서 잡아낸 양성 비율)
   spec@sens90 : 민감도 90%에서의 특이도 (양성의 90%를 잡는 문턱에서 정상으로 통과시킨 음성 비율)
   놓친 환자 / 1,000명 : 특이도 90%에서 (1 − sens) × 유병률 × 1,000 (그 데이터의 실제 유병률)
   → 문턱은 각 데이터의 ROC 곡선 위의 점 (AUROC처럼 순위만 사용, 확률 보정과 무관)
   비교: fs − c1_none, fs − fs_shuffled
   decision curve는 확률 보정에 좌우되어(외부 확신 부족·기저율 차이) 판별력 비교로 쓰지 않음 → 제외
② 다중 비교 보정 (Holm): 주 판별력 비교 6개(fs − c1_none, fs − fs_shuffled × 흉막·심장비대·폐부종)의 7곳 통합 AUROC
   (results/eval_disc_7sites/pooled.csv, 표준오차 = CI 폭 / 3.92) → 양측 p값 → Holm 보정. 보정 후 p < 0.05면 유지

25절 소견 확장 (2026-11-28 추가): --set more
   무기폐·경화, 외부 *_more 데이터(양성 ≥ 30), 비교 more_fs − more_c1, more_fs − more_fs_shuffled. 같은 지표·bootstrap·통합
   Holm은 판별력 평가(evaluate_discrimination.py --cond-set more)에서 이미 했으므로 여기서는 하지 않음

사용법  python evaluate_clinical.py [--n-boot 1000] [--set more]
출력    results/clinical/ (clinical.csv, pooled.csv, holm.csv), more = results/clinical_more/
"""
import argparse
from math import erf, sqrt

import numpy as np
import pandas as pd

from cal_common import DATASETS, MORE_EXTERNAL, RESULTS, S10, run_dir
from evaluate_discrimination import dersimonian_laird

OUT = RESULTS / "clinical"
DS7 = ["chexpert_test", "vindr_train", "vindr_train_b", "chexpert_valid", "vindr_test", "padchest_a", "padchest_b"]
CONDS = ["fs", "c1_none", "fs_shuffled"]
COMPS = [("fs", "c1_none"), ("fs", "fs_shuffled")]
PRIMARY_F = ["pleural", "cardiac", "edema"]


def wquantile(x, w, q):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return x[o][min(np.searchsorted(cw, q * cw[-1]), len(x) - 1)]


def op_metrics(s, y, w):
    pos, neg = y == 1, y == 0
    if w[pos].sum() == 0 or w[neg].sum() == 0:
        return np.array([np.nan, np.nan])
    t_spec = wquantile(s[neg], w[neg], 0.90)           # 음성 90%가 이 값 이하
    sens = (w[pos] * (s[pos] > t_spec)).sum() / w[pos].sum()
    t_sens = wquantile(s[pos], w[pos], 0.10)           # 양성 90%가 이 값 이상
    spec = (w[neg] * (s[neg] < t_sens)).sum() / w[neg].sum()
    return np.array([sens, spec])


def scores(c, s, ds, m, ids):
    t = pd.read_csv(run_dir(c, s) / f"preds_{ds}.csv").set_index("image_id").loc[ids]
    return t[f"logit_{m}"].values.astype(float)


def pnorm2(z):
    return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))


def holm():
    f = RESULTS / "eval_disc_7sites/pooled.csv"
    if not f.exists():
        print(f"[주의] {f} 없음 → Holm 보정 건너뜀")
        return None
    p = pd.read_csv(f)
    p = p[(p.pool == "통합(전체)") & (p.metric == "auroc") & p.comparison.isin(["fs-c1_none", "fs-fs_shuffled"])
          & p.finding.isin(PRIMARY_F)].copy()
    p["se"] = (p.ci_high - p.ci_low) / 3.92
    p["p"] = [pnorm2(e / s) for e, s in zip(p.estimate, p.se)]
    p = p.sort_values("p").reset_index(drop=True)
    m = len(p)
    adj, run = [], 0.0
    for i, pv in enumerate(p.p):
        run = max(run, min(1.0, (m - i) * pv))
        adj.append(run)
    p["p_holm"] = adj
    p["유지"] = np.where(p.p_holm < 0.05, "✓", "✗")
    return p[["comparison", "finding", "estimate", "ci_low", "ci_high", "p", "p_holm", "유지"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--set", choices=["base", "more"], default="base", help="25절: more = 무기폐·경화")
    args = ap.parse_args()
    global OUT, CONDS, COMPS
    if args.set == "more":
        OUT, CONDS = RESULTS / "clinical_more", ["more_fs", "more_c1", "more_fs_shuffled"]
        COMPS = [("more_fs", "more_c1"), ("more_fs", "more_fs_shuffled")]
        args.datasets = [d for d in (args.datasets or MORE_EXTERNAL) if DATASETS[d]["findings"]]
    else:
        args.datasets = args.datasets or DS7
    rng = np.random.default_rng(0)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, comps = [], []
    for ds in args.datasets:
        D = DATASETS[ds]
        lab = pd.read_csv(D["labels"])
        groups = lab["group"].values if "group" in lab.columns else np.arange(len(lab))
        ug, gi = np.unique(groups, return_inverse=True)
        seeds = {c: [s for s in S10 if (run_dir(c, s) / f"preds_{ds}.csv").exists()] for c in CONDS}
        for f, m in D["findings"].items():
            y = lab[f].values.astype(float)
            prev = y.mean()
            sc = {(c, s): scores(c, s, ds, m, lab.image_id) for c in CONDS for s in seeds[c]}
            one = np.ones(len(y))
            pt = {c: np.mean([op_metrics(sc[(c, s)], y, one) for s in seeds[c]], 0) for c in CONDS if seeds[c]}
            for c in pt:
                rows.append({"dataset": ds, "finding": f, "cond": c, "prev": prev, "sens@spec90": pt[c][0], "spec@sens90": pt[c][1],
                             "missed_per_1000": (1 - pt[c][0]) * prev * 1000})
            boot = {cp: [] for cp in COMPS}
            for _ in range(args.n_boot):
                w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[gi].astype(float)
                val = {}
                for c in pt:
                    pick = rng.choice(seeds[c], len(seeds[c]))
                    cache = {s: op_metrics(sc[(c, s)], y, w) for s in set(pick)}
                    val[c] = np.mean([cache[s] for s in pick], 0)
                for a, b in COMPS:
                    if a in val and b in val:
                        boot[(a, b)].append(val[a] - val[b])
            for a, b in COMPS:
                if not boot[(a, b)]:
                    continue
                bd = np.array(boot[(a, b)])
                for j, met in enumerate(["sens@spec90", "spec@sens90"]):
                    comps.append({"dataset": ds, "finding": f, "comparison": f"{a}-{b}", "metric": met,
                                  "estimate": pt[a][j] - pt[b][j], "low": np.nanquantile(bd[:, j], 0.025),
                                  "high": np.nanquantile(bd[:, j], 0.975), "se": np.nanstd(bd[:, j]), "prev": prev, "n": len(y)})
            print(f"  [{ds}] {f} bootstrap {args.n_boot}회 완료", flush=True)
    r, c = pd.DataFrame(rows), pd.DataFrame(comps)
    r.to_csv(OUT / "clinical.csv", index=False)
    c.to_csv(OUT / "comparisons.csv", index=False)
    pd.set_option("display.width", 220)

    print("\n=== 특이도 90%에서의 민감도 (seed 평균)")
    print(r.pivot_table(index=["finding", "dataset"], columns="cond", values="sens@spec90").round(3).to_string())
    pooled = []
    print("\n=== 통합 (random-effects). sens@spec90·spec@sens90 차이, 놓친 환자/1,000명 차이 = −Δsens × 통합 유병률 × 1,000. ✓ = CI가 0 미포함")
    for (cmp, f, met), g in c.groupby(["comparison", "finding", "metric"], sort=False):
        g = g.dropna(subset=["estimate", "se"])
        if g.empty:
            continue
        mu, lo, hi, _, i2 = dersimonian_laird(g.estimate.values, np.maximum(g.se.values, 1e-6))
        prev = float(np.average(g.prev, weights=g.n))
        pooled.append({"comparison": cmp, "finding": f, "metric": met, "estimate": mu, "low": lo, "high": hi, "i2": i2,
                       "n_data": len(g), "prev_pooled": prev})
    pl = pd.DataFrame(pooled)
    pl.to_csv(OUT / "pooled.csv", index=False)
    for cmp, g in pl.groupby("comparison", sort=False):
        print(f"\n[{cmp}]")
        for x in g.itertuples():
            star = "✓" if x.low > 0 or x.high < 0 else ""
            extra = f" → 놓친 환자 {-x.estimate * x.prev_pooled * 1000:+.1f}명/1,000명 (유병률 {x.prev_pooled:.1%})" if x.metric == "sens@spec90" else ""
            print(f"  {x.finding:16s} {x.metric:12s} {x.estimate:+.3f} [{x.low:+.3f}, {x.high:+.3f}]{star} (n={x.n_data}, I² {100 * x.i2:.0f}%){extra}")

    h = holm() if args.set == "base" else None
    if h is not None:
        h.to_csv(OUT / "holm.csv", index=False)
        print("\n=== 다중 비교 보정 (Holm, 주 판별력 비교 6개, 7곳 통합 AUROC)")
        print(h.round(4).to_string(index=False))
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
