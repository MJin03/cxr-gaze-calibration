"""11절 평가: 보정(calibration) 중심 지표와 조건 비교 (ANALYSIS_PLAN.md 11절).

지표 (finding별)
    주: calibration slope (temperature 없이; 예측 logit에 대한 로지스틱 재보정 기울기, 1이 이상적, <1이면 과신)
        NLL (temperature 없이, 낮을수록 좋음)
    보조: calibration intercept (기울기를 1로 둔 절편 = calibration-in-the-large, 0이 이상적), ECE(15구간),
          Brier, 같은 지표의 temperature scaling 후 값, AUROC
    seed 평균 ± SD와 앙상블(보정 전 확률의 seed 평균) 둘 다
비교 (cal_common.COMPARISONS): 차이 = 앞 − 뒤
    NLL 차이 (음수면 앞이 좋음), |slope − 1| 차이 (음수면 앞이 1에 더 가까움), AUROC 차이 (양수면 앞이 좋음)
    bootstrap 2000회: 내부는 환자, 외부는 영상 단위. seed 평균은 ci2(+조건별 seed 복원추출), 앙상블은 영상(환자)만

사용법  python evaluate_calibration.py --dataset internal|vindr_test|vindr_train|chexpert_valid
"""
import argparse
import json

import numpy as np
import pandas as pd

from cal_common import COMPARISONS, CONDITIONS, DATASETS, RESULTS, S10, run_dir

INTERNAL_F = {"pleural": "pleural", "cardiac": "cardiac", "edema": "edema"}
EPS = 1e-7


def sig(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


# ------------------------------------------------------------ 가중 지표
def nll(p, y, w):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / w.sum())


def slope_fit(z, y, w, iters=30):
    """가중 로지스틱 회귀 y ~ a + b z (뉴턴법). (a, b) 반환."""
    a, b = 0.0, 1.0
    for _ in range(iters):
        p = sig(a + b * z)
        r = w * (y - p)
        v = w * p * (1 - p) + 1e-12
        g = np.array([r.sum(), (r * z).sum()])
        h = np.array([[v.sum(), (v * z).sum()], [(v * z).sum(), (v * z * z).sum()]])
        try:
            step = np.linalg.solve(h + 1e-6 * np.eye(2), g)
        except np.linalg.LinAlgError:
            break
        step = np.clip(step, -2, 2)  # 한 번에 너무 크게 움직이지 않게 (완전 분리 데이터 대비)
        a, b = a + step[0], b + step[1]
        if np.abs(step).max() < 1e-8:
            break
    return a, b


def citl(z, y, w, iters=30):
    """기울기 1 고정 절편 (calibration-in-the-large)."""
    a = 0.0
    for _ in range(iters):
        p = sig(a + z)
        step = (w * (y - p)).sum() / ((w * p * (1 - p)).sum() + 1e-12)
        a += step
        if abs(step) < 1e-8:
            break
    return a


def ece(p, y, w, bins=15):
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.any():
            e += w[m].sum() / w.sum() * abs((w[m] * y[m]).sum() / w[m].sum() - (w[m] * p[m]).sum() / w[m].sum())
    return float(e)


class AUC:
    def __init__(self, score, y):
        order = np.argsort(score, kind="mergesort")
        s = score[order]
        self.order, self.y = order, y[order].astype(float)
        self.group = np.r_[0, np.cumsum(s[1:] != s[:-1])]
        self.ng = self.group[-1] + 1

    def __call__(self, w):
        w = w[self.order]
        p = np.bincount(self.group, w * self.y, self.ng)
        n = np.bincount(self.group, w * (1 - self.y), self.ng)
        tp, tn = p.sum(), n.sum()
        if tp == 0 or tn == 0:
            return np.nan
        return float((p * (np.cumsum(n) - n + 0.5 * n)).sum() / (tp * tn))


def all_metrics(z_raw, z_ts, y, w=None):
    w = np.ones(len(y)) if w is None else w
    out = {}
    for tag, z in [("raw", z_raw), ("ts", z_ts)]:
        p = sig(z)
        a, b = slope_fit(z, y, w)
        out[f"slope_{tag}"], out[f"citl_{tag}"] = b, citl(z, y, w)
        out[f"nll_{tag}"], out[f"ece_{tag}"] = nll(p, y, w), ece(p, y, w)
        out[f"brier_{tag}"] = float((w * (p - y) ** 2).sum() / w.sum())
    out["auroc"] = AUC(z_raw, y)(w)
    out["prevalence"] = float((w * y).sum() / w.sum())
    return out


# ------------------------------------------------------------ 불러오기
def load(dataset, allow_missing):
    """data[(cond, seed)][finding] = dict(zr=raw logit, zt=TS logit, y). seed 'ens' 포함."""
    if dataset == "internal":
        fmap, labels, groups = INTERNAL_F, None, None
    else:
        ds = DATASETS[dataset]
        fmap, labels = ds["findings"], pd.read_csv(ds["labels"])
        groups = np.arange(len(labels))
    data, seeds = {}, {}
    for c in CONDITIONS:
        ok = []
        for s in S10:
            d = run_dir(c, s)
            f_pred = d / ("preds_test.csv" if dataset == "internal" else f"preds_{dataset}.csv")
            if not f_pred.exists():
                if allow_missing:
                    continue
                raise SystemExit(f"예측 파일이 없습니다: {f_pred}")
            temp = json.load(open(d / "temperature.json"))
            t = pd.read_csv(f_pred)
            if dataset == "internal":
                t = t.sort_values("dicom_id").reset_index(drop=True)
                if groups is None:
                    groups = t.subject_id.values
                ylab = {f: t[f"{f}_t3"].values for f in fmap}
            else:
                t = t.set_index("image_id").loc[labels.image_id].reset_index()
                ylab = {f: labels[f].values for f in fmap}
            data[(c, s)] = {f: dict(zr=t[f"logit_{m}"].values.astype(float), zt=t[f"logit_{m}"].values / temp[m],
                                    y=ylab[f].astype(float)) for f, m in fmap.items()}
            ok.append(s)
        if ok:
            seeds[c] = ok
            data[(c, "ens")] = {f: dict(zr=logit(np.mean([sig(data[(c, s)][f]["zr"]) for s in ok], axis=0)),
                                        zt=logit(np.mean([sig(data[(c, s)][f]["zt"]) for s in ok], axis=0)),
                                        y=data[(c, ok[0])][f]["y"]) for f in fmap}
    return data, seeds, list(fmap), groups


# ------------------------------------------------------------ 메인
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["internal"] + list(DATASETS), required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--allow-missing", action="store_true", help="동작 확인용")
    args = ap.parse_args()
    out = RESULTS / f"eval_cal_{args.dataset}"
    out.mkdir(parents=True, exist_ok=True)
    data, seeds, findings, groups = load(args.dataset, args.allow_missing)
    n_img = len(groups)
    print(f"[{out.name}] 영상 {n_img}장, 조건 {len(seeds)}개, finding {findings}")

    rows = [{"cond": c, "seed": str(s), "finding": f, **all_metrics(data[(c, s)][f]["zr"], data[(c, s)][f]["zt"], data[(c, s)][f]["y"])}
            for c in seeds for s in seeds[c] + ["ens"] for f in findings]
    per = pd.DataFrame(rows)
    per.to_csv(out / "per_run.csv", index=False)

    # bootstrap
    rng = np.random.default_rng(0)
    ug, gi = np.unique(groups, return_inverse=True)
    conds = sorted({c for a, b, _ in COMPARISONS for c in (a, b) if c in seeds})
    auc_pre = {(c, s, f): AUC(data[(c, s)][f]["zr"], data[(c, s)][f]["y"]) for c in conds for s in seeds[c] + ["ens"] for f in findings}
    recs = []
    for b in range(args.n_boot):
        w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[gi].astype(float)
        pick = {c: rng.choice(seeds[c], len(seeds[c])) for c in conds}
        vals = {}
        for c in conds:
            for f in findings:
                m = {}
                for s in seeds[c] + ["ens"]:
                    d = data[(c, s)][f]
                    m[s] = (nll(sig(d["zr"]), d["y"], w), abs(slope_fit(d["zr"], d["y"], w)[1] - 1), auc_pre[(c, s, f)](w))
                allm, pk = np.array([m[s] for s in seeds[c]]), np.array([m[s] for s in pick[c]])
                vals[(c, f)] = (allm.mean(0), pk.mean(0), np.array(m["ens"]))
        for a, bb, kind in COMPARISONS:
            if a not in seeds or bb not in seeds:
                continue
            for f in findings:
                for j, metric in enumerate(["nll_raw", "abs_slope_minus1_raw", "auroc"]):
                    recs.append({"comparison": f"{a}-{bb}", "kind": kind, "finding": f, "metric": metric,
                                 "seed_ci": vals[(a, f)][0][j] - vals[(bb, f)][0][j],
                                 "seed_ci2": vals[(a, f)][1][j] - vals[(bb, f)][1][j],
                                 "ens_ci": vals[(a, f)][2][j] - vals[(bb, f)][2][j]})
        if (b + 1) % 200 == 0:
            print(f"  bootstrap {b + 1}/{args.n_boot}", flush=True)
    boot = pd.DataFrame(recs)
    per["abs_slope_minus1_raw"] = (per.slope_raw - 1).abs()
    so, en = per[per.seed != "ens"], per[per.seed == "ens"].set_index(["cond", "finding"])
    comp = []
    for (cmp, kind, f, metric), bd in boot.groupby(["comparison", "kind", "finding", "metric"], sort=False):
        a, bb = cmp.split("-")
        est = so[(so.cond == a) & (so.finding == f)][metric].mean() - so[(so.cond == bb) & (so.finding == f)][metric].mean()
        comp.append({"comparison": cmp, "kind": kind, "finding": f, "metric": metric, "seed_estimate": est,
                     "seed_ci2_low": bd.seed_ci2.quantile(0.025), "seed_ci2_high": bd.seed_ci2.quantile(0.975),
                     "seed_ci_low": bd.seed_ci.quantile(0.025), "seed_ci_high": bd.seed_ci.quantile(0.975),
                     "ens_estimate": en.loc[(a, f), metric] - en.loc[(bb, f), metric],
                     "ens_ci_low": bd.ens_ci.quantile(0.025), "ens_ci_high": bd.ens_ci.quantile(0.975)})
    comp = pd.DataFrame(comp)
    comp.to_csv(out / "comparisons.csv", index=False)

    # 화면 요약
    pd.set_option("display.width", 220)
    cols = [("slope_raw", "slope"), ("nll_raw", "NLL"), ("citl_raw", "intercept"), ("ece_raw", "ECE"),
            ("slope_ts", "slope(TS)"), ("ece_ts", "ECE(TS)"), ("auroc", "AUROC")]
    for f in findings:
        y0 = data[(conds[0], seeds[conds[0]][0])][f]["y"]
        print(f"\n=== [{f}] 양성 {int(y0.sum())}/{len(y0)} — temperature 없는 값 (seed 평균 ± SD, 앙상블)")
        tab = []
        for c in CONDITIONS:
            if c not in seeds:
                continue
            d = so[(so.cond == c) & (so.finding == f)]
            row = {"조건": c}
            for k, name in cols:
                row[name] = f"{d[k].mean():.3f}±{d[k].std():.3f} ({en.loc[(c, f), k]:.3f})"
            tab.append(row)
        print(pd.DataFrame(tab).to_string(index=False))
    print("\n=== 비교 (앞 − 뒤). NLL·|slope−1|은 음수, AUROC는 양수가 앞 조건이 좋음. ✓ = CI가 0을 포함하지 않음")
    names = {"nll_raw": "NLL", "abs_slope_minus1_raw": "|slope−1|", "auroc": "AUROC"}
    for f in findings:
        print(f"\n[{f}]")
        for (cmp, kind), g in comp[comp.finding == f].groupby(["comparison", "kind"], sort=False):
            parts = []
            for metric in ["nll_raw", "abs_slope_minus1_raw", "auroc"]:
                r = g[g.metric == metric].iloc[0]
                s2 = "✓" if (r.seed_ci2_low > 1e-12 or r.seed_ci2_high < -1e-12) else ""
                se = "✓" if (r.ens_ci_low > 1e-12 or r.ens_ci_high < -1e-12) else ""
                parts.append(f"{names[metric]} {r.seed_estimate:+.3f}[{r.seed_ci2_low:+.3f},{r.seed_ci2_high:+.3f}]{s2} "
                             f"앙상블 {r.ens_estimate:+.3f}[{r.ens_ci_low:+.3f},{r.ens_ci_high:+.3f}]{se}")
            print(f"  {cmp:22s} [{kind}]\n    " + "\n    ".join(parts))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
