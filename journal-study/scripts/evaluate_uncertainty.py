"""11-2절 B (보조): attention 퍼짐(정규화 엔트로피)을 불확실성 점수로 쓸 수 있는지 평가한다. 학습 없음.

점수 (finding별, 높을수록 불확실)
    conf  확신도만: 1 − max(p, 1 − p), p = temperature scaling 후 확률 (10절과 같은 확률)
    ent   엔트로피만: 그 finding의 attention softmax(16×16)의 엔트로피 / log(256), 0~1
    rank  둘의 순위 평균: (내부 val에서 conf의 백분위 + 내부 val에서 ent의 백분위) / 2 (가중치 조정 없음)
          기준 분포는 같은 run의 내부 val 영상 (label 사용 없음)
오류 = (p ≥ 0.5) ≠ label
지표
    오류 탐지 AUROC (점수가 오류를 얼마나 잘 골라내는지, 높을수록 좋음)
    AURC (불확실한 순서로 빼 가며 남은 영상의 오류율을 평균, 낮을수록 좋음)
    자동 문턱 missed: 10절 규칙 (최근 W = 500장 중 점수 상위 25%면 의사에게, 시작은 내부 val 점수, 무작위 순서 20번 평균)
         AI가 처리한 영상 중 놓친 양성 / 전체 양성 (낮을수록 좋음)
앙상블: 확률과 엔트로피 각각의 seed 평균으로 점수를 다시 계산
비교: 같은 조건 안에서 rank − conf, ent − conf (점수 비교). bootstrap 2000회 (환자 또는 영상 + 조건별 seed 복원추출 = ci2)

필요: predict_external.py --dataset <데이터> 와 --dataset internal_val (attention 저장)
출력  ~/X-ray/results/eval_unc_{dataset}/per_run.csv, comparisons.csv

사용법  python evaluate_uncertainty.py --dataset chexpert_test|vindr_train|chexpert_valid|vindr_test
"""
import argparse
import json

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from cal_common import DATASETS, FINDINGS_MODEL, INTERNAL_VAL, RESULTS, S10, UNCERTAINTY_CONDS, run_dir
from evaluate_calibration import AUC, sig

RATE, W, ORDERS = 0.25, 500, 20
SCORES = ["conf", "ent", "rank"]
METRICS = [("err_auroc", "오류 탐지 AUROC", +1), ("aurc", "AURC", -1), ("missed", "missed", -1)]


def entropy(att):
    """att: N×K×h×w attention logit → N×K 정규화 엔트로피."""
    a = att.reshape(att.shape[0], att.shape[1], -1).astype(np.float64)
    a = a - a.max(-1, keepdims=True)
    p = np.exp(a)
    p /= p.sum(-1, keepdims=True)
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(-1) / np.log(a.shape[-1])


def ecdf(ref, x):
    r = np.sort(ref)
    return np.searchsorted(r, x, side="right") / len(r)


def scores_from(p, e, pv, ev):
    """평가 영상과 내부 val의 점수 (평가, val) 쌍."""
    uc, ucv = 1 - np.maximum(p, 1 - p), 1 - np.maximum(pv, 1 - pv)
    return {"conf": (uc, ucv), "ent": (e, ev),
            "rank": ((ecdf(ucv, uc) + ecdf(ev, e)) / 2, (ecdf(ucv, ucv) + ecdf(ev, ev)) / 2)}


def referral_prob(u, uv, rng_seed):
    """영상별 '의사에게 보내질 확률' (무작위 순서 ORDERS개 평균). 최근 W장 중 상위 RATE면 보냄."""
    rng = np.random.default_rng(rng_seed)
    n, out = len(u), np.zeros(len(u))
    for _ in range(ORDERS):
        perm = rng.permutation(n)
        a = np.concatenate([rng.choice(uv, W, replace=True), u[perm]])
        thr = np.quantile(sliding_window_view(a, W)[:n], 1 - RATE, axis=1)
        refer = np.zeros(n, bool)
        refer[perm] = u[perm] >= thr
        out += refer
    return out / ORDERS


class Pre:
    """bootstrap 가중치만 바뀌므로 정렬 등은 미리 계산."""

    def __init__(self, u, p, y, r):
        self.err = ((p >= 0.5) != (y == 1)).astype(float)
        self.auc = AUC(u, self.err)
        o = np.argsort(u, kind="mergesort")              # 확실한 것부터
        self.o, self.err_o = o, self.err[o]
        self.miss = ((y == 1) & (p < 0.5)).astype(float) * (1 - r)
        self.pos, self.r = (y == 1).astype(float), r

    def __call__(self, w):
        wo = w[self.o]
        cw = np.cumsum(wo)
        risk = np.cumsum(wo * self.err_o) / np.maximum(cw, 1e-12)
        aurc = float((wo * risk).sum() / wo.sum())
        missed = float((w * self.miss).sum() / max((w * self.pos).sum(), 1e-12))
        return np.array([self.auc(w), aurc, missed, float((w * self.r).sum() / w.sum())])


def load(dataset, allow_missing):
    ds = DATASETS[dataset]
    labels = pd.read_csv(ds["labels"])
    groups = labels["group"].values if "group" in labels.columns else np.arange(len(labels))
    raw, seeds = {}, {}
    for c in UNCERTAINTY_CONDS:
        ok = []
        for s in S10:
            d = run_dir(c, s)
            need = [d / f"preds_{dataset}.csv", d / f"att_{dataset}.npy", d / f"preds_{INTERNAL_VAL}.csv", d / f"att_{INTERNAL_VAL}.npy"]
            if not all(f.exists() for f in need):
                if allow_missing:
                    continue
                raise SystemExit(f"예측 파일이 없습니다: {[str(f) for f in need if not f.exists()]}")
            temp = json.load(open(d / "temperature.json"))
            t = pd.read_csv(need[0])
            idx = pd.Series(np.arange(len(t)), index=t.image_id.astype(str)).loc[labels.image_id.astype(str)].values
            att, v, attv = np.load(need[1])[idx], pd.read_csv(need[2]), np.load(need[3])
            e, ev = entropy(att), entropy(attv)
            raw[(c, s)] = {m: dict(p=sig(t[f"logit_{m}"].values[idx] / temp[m]), e=e[:, j],
                                   pv=sig(v[f"logit_{m}"].values / temp[m]), ev=ev[:, j]) for j, m in enumerate(FINDINGS_MODEL)}
            ok.append(s)
        if ok:
            seeds[c] = ok
            raw[(c, "ens")] = {m: {k: np.mean([raw[(c, s)][m][k] for s in ok], axis=0) for k in ["p", "e", "pv", "ev"]}
                               for m in FINDINGS_MODEL}
    if not seeds:
        raise SystemExit("평가할 run이 없습니다.")
    return labels, groups, raw, seeds, ds["findings"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--allow-missing", action="store_true", help="동작 확인용")
    args = ap.parse_args()
    out = RESULTS / f"eval_unc_{args.dataset}"
    out.mkdir(parents=True, exist_ok=True)
    labels, groups, raw, seeds, fmap = load(args.dataset, args.allow_missing)
    findings = list(fmap)
    print(f"[{out.name}] 영상 {len(labels)}장, 조건 {list(seeds)}, finding {findings}")

    pre, rows = {}, []
    for c in seeds:
        for s in seeds[c] + ["ens"]:
            for fi, f in enumerate(findings):
                d, y = raw[(c, s)][fmap[f]], labels[f].values
                for si, (k, (u, uv)) in enumerate(scores_from(d["p"], d["e"], d["pv"], d["ev"]).items()):
                    r = referral_prob(u, uv, rng_seed=100 * fi + si)  # 같은 (finding, 점수)는 모든 조건이 같은 순서
                    pre[(c, s, f, k)] = Pre(u, d["p"], y, r)
                    m = pre[(c, s, f, k)](np.ones(len(y)))
                    rows.append({"cond": c, "seed": str(s), "finding": f, "score": k, "err_auroc": m[0], "aurc": m[1],
                                 "missed": m[2], "referral": m[3], "error_rate": pre[(c, s, f, k)].err.mean(),
                                 "mean_entropy": d["e"].mean()})
    per = pd.DataFrame(rows)
    per.to_csv(out / "per_run.csv", index=False)

    rng = np.random.default_rng(0)
    ug, gi = np.unique(groups, return_inverse=True)
    recs = []
    for b in range(args.n_boot):
        w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[gi].astype(float)
        pick = {c: rng.choice(seeds[c], len(seeds[c])) for c in seeds}
        for c in seeds:
            for f in findings:
                v = {}
                for k in SCORES:
                    m = {s: pre[(c, s, f, k)](w) for s in seeds[c] + ["ens"]}
                    v[k] = (np.mean([m[s] for s in seeds[c]], 0), np.mean([m[s] for s in pick[c]], 0), m["ens"])
                for k in ["rank", "ent"]:
                    for j, (metric, _, _) in enumerate(METRICS):
                        recs.append({"cond": c, "finding": f, "comparison": f"{k}-conf", "metric": metric,
                                     "seed_ci": v[k][0][j] - v["conf"][0][j], "seed_ci2": v[k][1][j] - v["conf"][1][j],
                                     "ens_ci": v[k][2][j] - v["conf"][2][j]})
        if (b + 1) % 200 == 0:
            print(f"  bootstrap {b + 1}/{args.n_boot}", flush=True)
    boot = pd.DataFrame(recs)
    so, en = per[per.seed != "ens"], per[per.seed == "ens"].set_index(["cond", "finding", "score"])
    comp = []
    for (c, f, cmp, metric), bd in boot.groupby(["cond", "finding", "comparison", "metric"], sort=False):
        k = cmp.split("-")[0]
        g = so[(so.cond == c) & (so.finding == f)]
        comp.append({"cond": c, "finding": f, "comparison": cmp, "metric": metric,
                     "seed_estimate": g[g.score == k][metric].mean() - g[g.score == "conf"][metric].mean(),
                     "seed_ci2_low": bd.seed_ci2.quantile(0.025), "seed_ci2_high": bd.seed_ci2.quantile(0.975),
                     "seed_ci_low": bd.seed_ci.quantile(0.025), "seed_ci_high": bd.seed_ci.quantile(0.975),
                     "ens_estimate": en.loc[(c, f, k), metric] - en.loc[(c, f, "conf"), metric],
                     "ens_ci_low": bd.ens_ci.quantile(0.025), "ens_ci_high": bd.ens_ci.quantile(0.975)})
    comp = pd.DataFrame(comp)
    comp.to_csv(out / "comparisons.csv", index=False)

    pd.set_option("display.width", 220)
    for f in findings:
        y = labels[f].values
        print(f"\n=== [{f}] 양성 {int(y.sum())}/{len(y)} — seed 평균 (앙상블). 오류 탐지 AUROC↑, AURC↓, missed↓ (W{W}, 하위 {int(RATE * 100)}% referral)")
        tab = []
        for c in seeds:
            for k in SCORES:
                d = so[(so.cond == c) & (so.finding == f) & (so.score == k)]
                e = en.loc[(c, f, k)]
                tab.append({"조건": c, "점수": k, "오류율": f"{d.error_rate.mean():.3f}",
                            "오류 탐지 AUROC": f"{d.err_auroc.mean():.3f}±{d.err_auroc.std():.3f} ({e.err_auroc:.3f})",
                            "AURC": f"{d.aurc.mean():.4f} ({e.aurc:.4f})",
                            "missed": f"{d.missed.mean():.3f} ({e.missed:.3f})", "referral": f"{d.referral.mean():.3f}",
                            "평균 엔트로피": f"{d.mean_entropy.mean():.3f}"})
        print(pd.DataFrame(tab).to_string(index=False))
    print("\n=== 점수 비교 (같은 조건 안, 앞 − conf). 오류 탐지 AUROC는 양수, AURC·missed는 음수가 좋음. ✓ = CI가 0을 포함하지 않음")
    names = {m: n for m, n, _ in METRICS}
    for f in findings:
        print(f"\n[{f}]")
        for (c, cmp), g in comp[comp.finding == f].groupby(["cond", "comparison"], sort=False):
            parts = []
            for metric, _, _ in METRICS:
                r = g[g.metric == metric].iloc[0]
                s2 = "✓" if (r.seed_ci2_low > 1e-12 or r.seed_ci2_high < -1e-12) else ""
                parts.append(f"{names[metric]} {r.seed_estimate:+.4f}[{r.seed_ci2_low:+.4f},{r.seed_ci2_high:+.4f}]{s2}")
            print(f"  {c:16s} {cmp:10s} " + " | ".join(parts))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
