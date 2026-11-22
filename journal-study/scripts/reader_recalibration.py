"""11-6절: 판독 방식 보정 (reader-protocol recalibration). 학습 없음, 새 병원 label 없음.

배경 (11-4절 진단): 학습 label 대부분이 판독의 1명이라 모델 확률 q는 "판독의 1명이 양성이라 할 확률"에 맞춰져 있다.
외부 데이터의 정답은 판독의 k명의 과반수라서 같은 q가 확신 부족(slope > 1)으로 보인다.

방법
    q = temperature scaling 후 확률 (내부 val에서 판독의 1명 label에 맞춘 확률)
    판독의 k명의 투표를 beta-binomial로 본다: 영상마다 판독의 1명의 양성 확률 θ ~ Beta(평균 q, 판독의 간 상관 ρ)
        a = q(1 − ρ)/ρ, b = (1 − q)(1 − ρ)/ρ
    보정 확률 = P(k명 중 과반수가 양성) = Σ_{v > k/2} BetaBinomial(v; k, a, b)
        ρ → 0: 판독의가 서로 독립 (가장 강한 보정), ρ → 1: 판독의가 모두 같은 판단 (보정 없음)
    ρ: 내부 REFLACX 판독의 5명 영상(내부 test 109장)에서 모델 q가 주어졌을 때 5명의 투표 가능도를 최대로 하는 값
        (조건·seed·finding마다 따로, 격자 0.001~0.99). 외부 label 사용 없음
    k: 외부 데이터의 label 방식 (데이터 설명서 기준, 평가 전에 고정) — K_PROTOCOL
    앙상블: seed별 보정 확률의 logit 평균

사용법
    python reader_recalibration.py fit                         # ρ 추정 (내부 데이터만) → results/reader_recal/rho.csv
    python reader_recalibration.py eval --dataset vindr_train_b   # 보정 전후 비교
"""
import argparse
import json
import math

import numpy as np
import pandas as pd

from cal_common import CONDITIONS, DATASETS, PROC, RESULTS, S10, run_dir
from evaluate_calibration import AUC, citl, ece, logit, nll, sig, slope_fit

K_PROTOCOL = {"vindr_train": 3, "vindr_train_b": 3, "chexpert_valid": 3, "chexpert_test": 5, "vindr_test": 5,
              "padchest_a": 1, "padchest_b": 1}  # PadChest: 판독문 1명 (14절)
RHO_GRID = np.r_[0.001, 0.005, np.arange(0.01, 0.991, 0.01)]
INTERNAL_F = ["pleural", "cardiac", "edema"]
OUT = RESULTS / "reader_recal"
EPS = 1e-6


def bb_pmf(q, k, rho):
    """N×(k+1) beta-binomial 확률. 정수 k가 작으므로 곱셈식으로 정확히 계산."""
    q = np.clip(np.asarray(q, float), EPS, 1 - EPS)[:, None]
    c = (1 - rho) / rho
    a, b = q * c, (1 - q) * c
    out = np.zeros((q.shape[0], k + 1))
    den = np.prod([a + b + t for t in range(k)], axis=0)
    for v in range(k + 1):
        num = np.ones_like(a)
        for i in range(v):
            num = num * (a + i)
        for j in range(k - v):
            num = num * (b + j)
        out[:, v] = math.comb(k, v) * (num / den)[:, 0]
    return out


def maj_prob(q, k, rho):
    p = bb_pmf(q, k, rho)[:, k // 2 + 1:].sum(1)
    return np.clip(p, EPS, 1 - EPS)


def internal_votes():
    """판독의 5명(REFLACX) 영상의 finding별 양성 투표 수 (확신도 ≥ 3, 2단계 label 규칙과 같음)."""
    r = pd.read_csv(PROC / "cohort_readings.csv")
    r = r[r.dataset == "reflacx"]
    n = r.groupby("dicom_id").size()
    ids = n[n == 5].index
    r = r[r.dicom_id.isin(ids)]
    return pd.DataFrame({f: (r[f"{f}_raw"] >= 3).groupby(r.dicom_id).sum() for f in INTERNAL_F})


def do_fit(args):
    votes = internal_votes()
    rows = []
    for c in CONDITIONS:
        for s in S10:
            d = run_dir(c, s)
            if not (d / "preds_test.csv").exists():
                continue
            t = pd.read_csv(d / "preds_test.csv").set_index("dicom_id")
            ids = votes.index.intersection(t.index)
            temp = json.load(open(d / "temperature.json"))
            for f in INTERNAL_F:
                q = sig(t.loc[ids, f"logit_{f}"].values / temp[f])
                v = votes.loc[ids, f].values.astype(int)
                ll = [np.log(np.clip(bb_pmf(q, 5, r)[np.arange(len(v)), v], 1e-300, None)).sum() for r in RHO_GRID]
                rho = float(RHO_GRID[int(np.argmax(ll))])
                y = (v >= 3).astype(float)
                z_ts, z_c = logit(q), logit(maj_prob(q, 5, rho))
                w = np.ones(len(y))
                rows.append({"cond": c, "seed": s, "finding": f, "rho": rho, "n_img": len(ids), "pos_majority": int(y.sum()),
                             "slope_ts_vs_majority": slope_fit(z_ts, y, w)[1], "slope_corr_vs_majority": slope_fit(z_c, y, w)[1],
                             "nll_ts": nll(sig(z_ts), y, w), "nll_corr": nll(sig(z_c), y, w)})
    if not rows:
        raise SystemExit("preds_test.csv가 있는 run이 없습니다.")
    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "rho.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"=== ρ 추정 (내부 판독의 5명 영상 {res.n_img.iloc[0]}장, 외부 label 사용 없음). 0 = 판독의 독립, 1 = 모두 같은 판단")
    g = res.groupby(["finding", "cond"]).agg(rho=("rho", "mean"), rho_sd=("rho", "std"), pos=("pos_majority", "first"),
                                             slope_ts=("slope_ts_vs_majority", "mean"), slope_corr=("slope_corr_vs_majority", "mean"),
                                             nll_ts=("nll_ts", "mean"), nll_corr=("nll_corr", "mean")).reset_index()
    print(g.round(3).to_string(index=False))
    print("\n※ 오른쪽 네 열은 ρ를 맞춘 같은 영상에서 잰 값이라 참고용 (확인은 외부 데이터에서)")
    print(f"저장: {OUT / 'rho.csv'}")


def metrics(z, y, w):
    p = sig(z)
    return {"slope": slope_fit(z, y, w)[1], "citl": citl(z, y, w), "nll": nll(p, y, w), "ece": ece(p, y, w),
            "brier": float((w * (p - y) ** 2).sum() / w.sum())}


def do_eval(args):
    ds, k = DATASETS[args.dataset], K_PROTOCOL[args.dataset]
    rho_tab = pd.read_csv(OUT / "rho.csv").set_index(["cond", "seed", "finding"]).rho
    lab = pd.read_csv(ds["labels"])
    groups = lab["group"].values if "group" in lab.columns else np.arange(len(lab))
    fmap = ds["findings"]
    conds = [c for c in args.conds if c in CONDITIONS]
    data, seeds = {}, {}
    for c in conds:
        ok = []
        for s in S10:
            d = run_dir(c, s)
            f_pred = d / f"preds_{args.dataset}.csv"
            if not f_pred.exists() or (c, s, "pleural") not in rho_tab.index:
                if args.allow_missing:
                    continue
                raise SystemExit(f"예측 파일 또는 ρ가 없습니다: {f_pred}")
            temp = json.load(open(d / "temperature.json"))
            t = pd.read_csv(f_pred).set_index("image_id").loc[lab.image_id]
            data[(c, s)] = {}
            for f, m in fmap.items():
                z = t[f"logit_{m}"].values.astype(float)
                q = sig(z / temp[m])
                data[(c, s)][f] = {"raw": z, "ts": logit(q), "corr": logit(maj_prob(q, k, rho_tab[(c, s, m)])),
                                   "indep": logit(maj_prob(q, k, 0.001)), "y": lab[f].values.astype(float)}
            ok.append(s)
        if ok:
            seeds[c] = ok
            data[(c, "ens")] = {f: {**{v: np.mean([data[(c, s)][f][v] for s in ok], 0) for v in ["raw", "ts", "corr", "indep"]},
                                    "y": data[(c, ok[0])][f]["y"]} for f in fmap}
    if not seeds:
        raise SystemExit("평가할 run이 없습니다.")
    variants = ["raw", "ts", "corr", "indep"]
    w1 = np.ones(len(lab))
    rows = [{"cond": c, "seed": str(s), "finding": f, "variant": v, **metrics(data[(c, s)][f][v], data[(c, s)][f]["y"], w1)}
            for c in seeds for s in seeds[c] + ["ens"] for f in fmap for v in variants]
    per = pd.DataFrame(rows)
    out = OUT / f"eval_{args.dataset}"
    out.mkdir(parents=True, exist_ok=True)
    per.to_csv(out / "per_run.csv", index=False)

    rng = np.random.default_rng(0)
    ug, gi = np.unique(groups, return_inverse=True)
    recs = []
    for b in range(args.n_boot):
        w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[gi].astype(float)
        for c in seeds:
            pick = rng.choice(seeds[c], len(seeds[c]))
            for f in fmap:
                val = {}
                for v in variants:
                    m = {s: (lambda d: (nll(sig(d[v]), d["y"], w), abs(slope_fit(d[v], d["y"], w)[1] - 1), ece(sig(d[v]), d["y"], w)))(data[(c, s)][f])
                         for s in set(pick) | {"ens"}}
                    val[v] = (np.mean([m[s] for s in pick], 0), np.array(m["ens"]))
                for a, bb in [("corr", "ts"), ("corr", "raw"), ("indep", "ts")]:
                    for j, metric in enumerate(["nll", "abs_slope_minus1", "ece"]):
                        recs.append({"cond": c, "finding": f, "comparison": f"{a}-{bb}", "metric": metric,
                                     "seed_ci2": val[a][0][j] - val[bb][0][j], "ens_ci": val[a][1][j] - val[bb][1][j]})
        if (b + 1) % 200 == 0:
            print(f"  bootstrap {b + 1}/{args.n_boot}", flush=True)
    boot = pd.DataFrame(recs)
    per["abs_slope_minus1"] = (per.slope - 1).abs()
    so = per[per.seed != "ens"]
    en = per[per.seed == "ens"].set_index(["cond", "finding", "variant"])
    comp = []
    for (c, f, cmp, metric), bd in boot.groupby(["cond", "finding", "comparison", "metric"], sort=False):
        a, bb = cmp.split("-")
        g = so[(so.cond == c) & (so.finding == f)]
        comp.append({"cond": c, "finding": f, "comparison": cmp, "metric": metric,
                     "seed_estimate": g[g.variant == a][metric].mean() - g[g.variant == bb][metric].mean(),
                     "seed_ci2_low": bd.seed_ci2.quantile(0.025), "seed_ci2_high": bd.seed_ci2.quantile(0.975),
                     "ens_estimate": en.loc[(c, f, a), metric] - en.loc[(c, f, bb), metric],
                     "ens_ci_low": bd.ens_ci.quantile(0.025), "ens_ci_high": bd.ens_ci.quantile(0.975)})
    comp = pd.DataFrame(comp)
    comp.to_csv(out / "comparisons.csv", index=False)

    pd.set_option("display.width", 220)
    names = {"raw": "raw (보정 없음)", "ts": "TS (내부 temperature)", "corr": f"판독 방식 보정 (k={k}, ρ 추정)", "indep": f"독립 가정 (k={k}, ρ=0)"}
    print(f"[{args.dataset}] 영상 {len(lab)}장, 판독 방식 k = {k}명 과반수, 조건 {list(seeds)}")
    for f in fmap:
        print(f"\n=== [{f}] 양성 {int(lab[f].sum())}/{len(lab)} — seed 평균 (logit 평균 앙상블)")
        tab = []
        for c in seeds:
            for v in variants:
                d = so[(so.cond == c) & (so.finding == f) & (so.variant == v)]
                e = en.loc[(c, f, v)]
                tab.append({"조건": c, "확률": names[v], "slope": f"{d.slope.mean():.3f}±{d.slope.std():.3f} ({e.slope:.3f})",
                            "intercept": f"{d.citl.mean():+.3f} ({e.citl:+.3f})", "NLL": f"{d.nll.mean():.4f} ({e.nll:.4f})",
                            "ECE": f"{d.ece.mean():.3f} ({e.ece:.3f})", "Brier": f"{d.brier.mean():.4f} ({e.brier:.4f})"})
        print(pd.DataFrame(tab).to_string(index=False))
    print("\n=== 비교 (앞 − 뒤, 음수가 좋음). seed 평균 ci2 / 앙상블 ci. ✓ = CI가 0을 포함하지 않음")
    lab_m = {"nll": "NLL", "abs_slope_minus1": "|slope−1|", "ece": "ECE"}
    for f in fmap:
        print(f"\n[{f}]")
        for (c, cmp), g in comp[comp.finding == f].groupby(["cond", "comparison"], sort=False):
            parts = []
            for metric in ["nll", "abs_slope_minus1", "ece"]:
                r = g[g.metric == metric].iloc[0]
                s2 = "✓" if (r.seed_ci2_low > 1e-12 or r.seed_ci2_high < -1e-12) else ""
                se = "✓" if (r.ens_ci_low > 1e-12 or r.ens_ci_high < -1e-12) else ""
                parts.append(f"{lab_m[metric]} {r.seed_estimate:+.4f}[{r.seed_ci2_low:+.4f},{r.seed_ci2_high:+.4f}]{s2} "
                             f"앙상블 {r.ens_estimate:+.4f}[{r.ens_ci_low:+.4f},{r.ens_ci_high:+.4f}]{se}")
            print(f"  {c:16s} {cmp:10s}\n    " + "\n    ".join(parts))
    print(f"\n저장: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["fit", "eval"])
    ap.add_argument("--dataset", choices=list(K_PROTOCOL))
    ap.add_argument("--conds", nargs="*", default=["c1_none", "fs", "gaze", "uniform", "a_gaze", "a_free"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--allow-missing", action="store_true", help="동작 확인용")
    args = ap.parse_args()
    if args.action == "fit":
        do_fit(args)
    else:
        if not args.dataset:
            raise SystemExit("eval에는 --dataset이 필요합니다.")
        do_eval(args)


if __name__ == "__main__":
    main()
