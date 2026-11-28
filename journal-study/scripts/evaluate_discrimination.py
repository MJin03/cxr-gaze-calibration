"""12절: 판별력 외부 검증 확장 (ANALYSIS_PLAN.md 12절, 결과 보기 전에 고정). 학습 없음.

9절(VinDr test)의 조건·비교를 외부 데이터 여러 곳으로 넓히고, 데이터 간 결과를 합친다.
    조건: fs, c1_none, fs_shuffled, real_global, hyb_none, ellipse (cal_common.DISC_CONDS)
    비교: fs − c1_none, fs − fs_shuffled (주) / fs − real_global, fs − hyb_none (보조) / fs − ellipse (비열등, 마진 0.02)
지표
    AUROC (주): 확률 순서만 보므로 temperature와 무관
    자동 문턱 missed (보조): 10절 규칙 (최근 500장 중 확신도 하위 25% referral, 시작은 내부 val, 무작위 순서 20번 평균)
    seed 평균과 앙상블(temperature 후 확률의 seed 평균)
신뢰구간: 환자(CheXpert) 또는 영상(VinDr) + 조건별 seed bootstrap 2000회 (ci2)
데이터 합치기: 데이터별 seed 평균 차이와 ci2 폭에서 구한 표준오차로 random-effects(DerSimonian-Laird) 통합 추정
    VinDr 데이터 3개는 같은 병원이라 독립 병원 수는 2곳(VinDr, CheXpert)임을 함께 보고

사용법  python evaluate_discrimination.py [--datasets chexpert_test vindr_train vindr_train_b chexpert_valid vindr_test] [--tag 이름]
        14절: --datasets (위 5개) padchest_a padchest_b --tag 7sites
        25절: --cond-set more --datasets internal_test_more --tag more_internal
              --cond-set more --datasets chexpert_test_more chexpert_valid_more vindr_test_more vindr_train_more
                                         vindr_train_b_more padchest_a_more padchest_b_more --tag more_ext
              (양성 < 30인 소견은 label 파일에 열이 없어 자동으로 빠지고, 평가할 소견이 없는 데이터는 건너뜀)
"""
import argparse
import json

import numpy as np
import pandas as pd

from cal_common import DATASETS, DISC_COMPARISONS, DISC_CONDS, NI_MARGIN, RESULTS, S10, run_dir

# 19절: 강한 모델(NIH 사전학습) 비교 세트. --cond-set nih
COND_SETS = {
    "base": (DISC_CONDS, DISC_COMPARISONS),
    "nih": (["nih_fs", "nih_c1", "nih_fs_shuffled", "nih_hyb_none", "fs", "c1_none"],
            [("nih_fs", "nih_c1", "primary"), ("nih_fs", "nih_fs_shuffled", "primary"),
             ("nih_fs", "nih_hyb_none", "secondary"),
             ("nih_c1", "c1_none", "strength"), ("nih_fs", "fs", "strength")]),
    # 23절: 강한 모델(ConvNeXt-Tiny). strength = 강한 모델 확인(S0), cnx_fs − fs = 강한 모델 + 시선이 기존 최선보다 나은지
    "cnx": (["cnx_fs", "cnx_c1", "cnx_fs_shuffled", "cnx_hyb_none", "fs", "c1_none"],
            [("cnx_fs", "cnx_c1", "primary"), ("cnx_fs", "cnx_fs_shuffled", "primary"),
             ("cnx_fs", "cnx_hyb_none", "secondary"),
             ("cnx_c1", "c1_none", "strength"), ("cnx_fs", "fs", "strength")]),
    # 25절: 소견 확장 (무기폐·경화). 데이터는 *_more (내부 test는 internal_test_more, 외부와 따로 실행)
    "more": (["more_fs", "more_c1", "more_fs_shuffled", "more_hyb_none"],
             [("more_fs", "more_c1", "primary"), ("more_fs", "more_fs_shuffled", "primary"),
              ("more_fs", "more_hyb_none", "secondary")]),
}
from evaluate_calibration import AUC, sig
from evaluate_rolling import metrics as roll_metrics
from evaluate_rolling import referral_prob

W, ORDERS = 500, 20
OUT = RESULTS / "eval_disc"


def load(dataset, allow_missing):
    ds = DATASETS[dataset]
    lab = pd.read_csv(ds["labels"])
    groups = lab["group"].values if "group" in lab.columns else np.arange(len(lab))
    data, seeds = {}, {}
    for c in DISC_CONDS:  # main()에서 --cond-set으로 바뀔 수 있음
        ok = []
        for s in S10:
            d = run_dir(c, s)
            f_pred = d / f"preds_{ds.get('preds', dataset)}.csv"   # 26절: 다른 label로 같은 예측을 평가할 때
            if not f_pred.exists():
                if allow_missing:
                    continue
                raise SystemExit(f"예측 파일이 없습니다: {f_pred}")
            temp = json.load(open(d / "temperature.json"))
            t = pd.read_csv(f_pred).set_index("image_id").loc[lab.image_id]
            v = pd.read_csv(d / "preds_val.csv")
            data[(c, s)] = {}
            for f, m in ds["findings"].items():
                pv = sig(v[f"logit_{m}"].values / temp[m])
                data[(c, s)][f] = dict(p=sig(t[f"logit_{m}"].values / temp[m]), cv=np.maximum(pv, 1 - pv), y=lab[f].values.astype(float))
            ok.append(s)
        if ok:
            seeds[c] = ok
            data[(c, "ens")] = {f: {k: np.mean([data[(c, s)][f][k] for s in ok], 0) for k in ["p", "cv"]} | {"y": data[(c, ok[0])][f]["y"]}
                                for f in ds["findings"]}
    return lab, groups, data, seeds, list(ds["findings"])


class Pre:
    def __init__(self, d, rng_seed):
        self.auc = AUC(d["p"], d["y"])
        self.r = referral_prob(d["cv"], d["p"], W, ORDERS, rng_seed)
        self.p, self.y = d["p"], d["y"]

    def __call__(self, w):
        return np.array([self.auc(w), roll_metrics(self.r, self.p, self.y, w)["missed"]])


def evaluate(dataset, n_boot, allow_missing):
    lab, groups, data, seeds, findings = load(dataset, allow_missing)
    pre = {(c, s, f): Pre(data[(c, s)][f], rng_seed=fi) for c in seeds for s in seeds[c] + ["ens"] for fi, f in enumerate(findings)}
    one = np.ones(len(lab))
    per = pd.DataFrame([{"dataset": dataset, "cond": c, "seed": str(s), "finding": f,
                         **dict(zip(["auroc", "missed"], pre[(c, s, f)](one)))}
                        for c in seeds for s in seeds[c] + ["ens"] for f in findings])
    rng = np.random.default_rng(0)
    ug, gi = np.unique(groups, return_inverse=True)
    recs = []
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[gi].astype(float)
        pick = {c: rng.choice(seeds[c], len(seeds[c])) for c in seeds}
        val = {}
        for c in seeds:
            for f in findings:
                m = {s: pre[(c, s, f)](w) for s in set(pick[c]) | {"ens"}}
                val[(c, f)] = (np.mean([m[s] for s in pick[c]], 0), m["ens"])
        for a, bb, kind in DISC_COMPARISONS:  # 위와 같음
            if a in seeds and bb in seeds:
                for f in findings:
                    for j, metric in enumerate(["auroc", "missed"]):
                        recs.append({"comparison": f"{a}-{bb}", "kind": kind, "finding": f, "metric": metric,
                                     "seed_ci2": val[(a, f)][0][j] - val[(bb, f)][0][j], "ens_ci": val[(a, f)][1][j] - val[(bb, f)][1][j]})
        if (b + 1) % 200 == 0:
            print(f"  [{dataset}] bootstrap {b + 1}/{n_boot}", flush=True)
    boot = pd.DataFrame(recs)
    so, en = per[per.seed != "ens"], per[per.seed == "ens"].set_index(["cond", "finding"])
    comp = []
    for (cmp, kind, f, metric), bd in boot.groupby(["comparison", "kind", "finding", "metric"], sort=False):
        a, bb = cmp.split("-")
        g = so[so.finding == f]
        comp.append({"dataset": dataset, "comparison": cmp, "kind": kind, "finding": f, "metric": metric,
                     "seed_estimate": g[g.cond == a][metric].mean() - g[g.cond == bb][metric].mean(),
                     "seed_ci2_low": bd.seed_ci2.quantile(0.025), "seed_ci2_high": bd.seed_ci2.quantile(0.975),
                     "ens_estimate": en.loc[(a, f), metric] - en.loc[(bb, f), metric],
                     "ens_ci_low": bd.ens_ci.quantile(0.025), "ens_ci_high": bd.ens_ci.quantile(0.975),
                     "n_img": len(lab), "pos": int(lab[f].sum())})
    return per, pd.DataFrame(comp)


def dersimonian_laird(est, se):
    w = 1 / se ** 2
    fixed = (w * est).sum() / w.sum()
    q = (w * (est - fixed) ** 2).sum()
    k = len(est)
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum())) if k > 1 else 0.0
    wr = 1 / (se ** 2 + tau2)
    mu = (wr * est).sum() / wr.sum()
    se_mu = np.sqrt(1 / wr.sum())
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    return mu, mu - 1.96 * se_mu, mu + 1.96 * se_mu, tau2, i2


def mark(lo, hi, kind, metric):
    if kind == "noninferiority" and metric == "auroc":
        return "비열등✓" if lo > -NI_MARGIN else ""
    return "✓" if (lo > 1e-12 or hi < -1e-12) else ""


def holm_more(pooled):
    """25절: 주 비교 4개(more_fs − more_c1, more_fs − more_fs_shuffled × 무기폐·경화)의 통합(전체) AUROC에 Holm 보정 (21절과 같은 방법)."""
    from math import erf, sqrt
    p = pooled[(pooled.pool == "통합(전체)") & (pooled.metric == "auroc") & (pooled.kind == "primary")].copy()
    if p.empty:
        return
    p["se"] = (p.ci_high - p.ci_low) / 3.92
    p["p"] = [2 * (1 - 0.5 * (1 + erf(abs(e / s) / sqrt(2)))) for e, s in zip(p.estimate, p.se)]
    p = p.sort_values("p").reset_index(drop=True)
    adj, run = [], 0.0
    for i, pv in enumerate(p.p):
        run = max(run, min(1.0, (len(p) - i) * pv))
        adj.append(run)
    p["p_holm"] = adj
    p["유지"] = np.where((p.p_holm < 0.05) & (p.estimate > 0), "✓", "✗")
    p = p[["comparison", "finding", "estimate", "ci_low", "ci_high", "n_data", "p", "p_holm", "유지"]]
    print(f"\n=== 25절 Holm 보정 (주 비교 {len(p)}개, 통합(전체) AUROC). 유지 = 보정 p < 0.05이고 fs가 더 좋음")
    print(p.round(4).to_string(index=False))
    p.to_csv(OUT / "holm.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["chexpert_test", "vindr_train", "vindr_train_b", "chexpert_valid", "vindr_test"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--allow-missing", action="store_true", help="동작 확인용")
    ap.add_argument("--tag", default="", help="결과 폴더 eval_disc_{tag} (기존 12절 결과를 덮어쓰지 않도록)")
    ap.add_argument("--cond-set", choices=list(COND_SETS), default="base", help="19절: nih = 강한 모델 비교")
    args = ap.parse_args()
    global OUT, DISC_CONDS, DISC_COMPARISONS
    DISC_CONDS, DISC_COMPARISONS = COND_SETS[args.cond_set]
    if args.tag:
        OUT = RESULTS / f"eval_disc_{args.tag}"
    OUT.mkdir(parents=True, exist_ok=True)
    pers, comps = [], []
    for d in args.datasets:
        if not DATASETS[d]["findings"]:
            print(f"[건너뜀] {d}: 평가할 소견 없음 (label 파일 없음 또는 양성 < 30)")
            continue
        per, comp = evaluate(d, args.n_boot, args.allow_missing)
        pers.append(per)
        comps.append(comp)
        per.to_csv(OUT / f"per_run_{d}.csv", index=False)
        comp.to_csv(OUT / f"comparisons_{d}.csv", index=False)
    per, comp = pd.concat(pers), pd.concat(comps)

    pd.set_option("display.width", 220)
    print("=== AUROC (seed 평균, 괄호 = 앙상블)")
    so = per[per.seed != "ens"].groupby(["finding", "cond", "dataset"]).auroc.mean().unstack("dataset")
    en = per[per.seed == "ens"].set_index(["finding", "cond", "dataset"]).auroc.unstack("dataset")
    tab = so.copy().astype(object)
    for idx in so.index:
        for col in so.columns:
            tab.loc[idx, col] = f"{so.loc[idx, col]:.3f} ({en.loc[idx, col]:.3f})" if pd.notna(so.loc[idx, col]) else "없음"
    print(tab[[d for d in args.datasets if d in tab.columns]].to_string())

    print("\n=== 비교 (fs − 대조, 데이터별 seed 평균 [ci2]). AUROC는 양수, missed는 음수가 fs가 좋음. ✓ = CI가 0을 포함하지 않음, 비열등✓ = 하한 > −0.02")
    pooled = []
    for f in comp.finding.unique():
        for metric in ["auroc", "missed"]:
            print(f"\n[{f} · {metric}]")
            rows = []
            for (cmp, kind), g in comp[(comp.finding == f) & (comp.metric == metric)].groupby(["comparison", "kind"], sort=False):
                row = {"비교": cmp, "종류": kind}
                for r in g.itertuples():
                    row[r.dataset] = f"{r.seed_estimate:+.3f}[{r.seed_ci2_low:+.3f},{r.seed_ci2_high:+.3f}]{mark(r.seed_ci2_low, r.seed_ci2_high, kind, metric)}"
                for name, gg in [("통합(전체)", g), ("통합(VinDr)", g[g.dataset.str.startswith("vindr")]),
                                 ("통합(CheXpert)", g[g.dataset.str.startswith("chexpert")]),
                                 ("통합(PadChest)", g[g.dataset.str.startswith("padchest")])]:
                    if len(gg) >= 2:
                        se = ((gg.seed_ci2_high - gg.seed_ci2_low) / (2 * 1.96)).clip(lower=1e-6).values
                        mu, lo, hi, tau2, i2 = dersimonian_laird(gg.seed_estimate.values, se)
                        row[name] = f"{mu:+.3f}[{lo:+.3f},{hi:+.3f}]{mark(lo, hi, kind, metric)} (n={len(gg)}, I² {i2:.0%})"
                        pooled.append({"comparison": cmp, "kind": kind, "finding": f, "metric": metric, "pool": name,
                                       "estimate": mu, "ci_low": lo, "ci_high": hi, "tau2": tau2, "i2": i2, "n_data": len(gg)})
                rows.append(row)
            print(pd.DataFrame(rows).fillna("없음").to_string(index=False))
    comp.to_csv(OUT / "comparisons_all.csv", index=False)
    pd.DataFrame(pooled).to_csv(OUT / "pooled.csv", index=False)
    if args.cond_set == "more" and pooled:
        holm_more(pd.DataFrame(pooled))
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
