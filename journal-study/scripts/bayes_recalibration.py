"""13절: 정보 수준별 베이지안 재보정 (ANALYSIS_PLAN.md 13절, 실행 전에 고정). 학습 없음, 이미 만든 예측만 사용.

틀: 보정 확률 = sigmoid(a + b·u). u = 기준 logit, (a, b)를 병원 정보로 갱신한다 (MAP 추정)
    사전 믿음: a ~ N(0, 1²), b ~ N(1, 0.5²)  → 정보가 없으면 (a, b) = (0, 1), 즉 기준 확률 그대로
    유병률 정보: 대상 영상 전체의 평균 예측 확률이 π_기록에 가깝도록 (표준편차 0.25·π_기록인 정규 가능도)
    label 정보: label 있는 n장의 Bernoulli 가능도
    → 정보가 늘어날수록 자연스럽게 반영되므로 단계별 방법을 따로 고르지 않는다

기준 logit u
    reader : 판독 방식 보정 확률(11-6절, 대상 데이터의 판독 방식 k, 내부 추정 ρ)의 logit
    ts     : 내부 temperature scaling 후 logit (비교용)

현실적 평가 (같은 병원의 다른 묶음을 "과거 기록"으로 사용, 평가 데이터 정답은 쓰지 않음)
    병원 VinDr: vindr_train, vindr_train_b, vindr_test / 병원 CheXpert: chexpert_valid, chexpert_test
    같은 병원 안의 모든 (기록 묶음 → 대상 묶음) 쌍: VinDr 6쌍, CheXpert 2쌍
    π_기록 = 기록 묶음의 실제 양성 비율, label n장 = 기록 묶음에서 무작위 n장 (반복 100번)

방법
    정보 없음  : raw, ts(내부 TS), reader(판독 방식 보정 = 사전 믿음 그대로)
    유병률     : prev_ts, prev_reader (기준 logit + 유병률 가능도)
               민감도: π_기록 × {0.5, 0.75, 1.25, 1.5}
    label n장  : ts_refit(temperature만 다시), platt(a, b 최대우도), bayes_ts, bayes_reader,
               bayes_reader_prev(유병률 + label), n ∈ {10, 25, 50, 100, 200}
    참고 하한  : oracle (대상 데이터 전체로 맞춘 platt, 같은 데이터에서 평가 → 실제로 불가능한 최선)
지표: 대상 묶음 전체에서 NLL, |slope − 1|, intercept, ECE (seed 평균, label n장은 반복 평균과 10~90% 범위)
조건: fs, c1_none / finding: 흉막, 심장비대, effusion, 폐부종(CheXpert)

사용법  python bayes_recalibration.py [--draws 100]
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from cal_common import DATASETS, RESULTS, S10, run_dir
from evaluate_calibration import citl, ece, logit, nll, sig, slope_fit
from reader_recalibration import K_PROTOCOL, maj_prob

HOSPITALS = {"VinDr": ["vindr_train", "vindr_train_b", "vindr_test"], "CheXpert": ["chexpert_valid", "chexpert_test"]}
CONDS = ["fs", "c1_none"]
NS = [10, 25, 50, 100, 200]
PRIOR = (0.0, 1.0, 1.0, 0.5)          # μa, μb, σa, σb
PREV_REL_SD = 0.25
PREV_MULT = [0.5, 0.75, 1.25, 1.5]
OUT = RESULTS / "bayes_recal"


def fit(ul=None, yl=None, ut=None, prev=None, prior=PRIOR, a_fixed=None):
    """MAP (a, b). ul, yl: label 있는 영상의 기준 logit과 정답 / ut: 대상 영상 기준 logit (유병률 가능도용)."""
    ma, mb, sa, sb = prior

    def obj(th):
        a, b = (a_fixed, th[0]) if a_fixed is not None else th
        f, ga, gb = (a - ma) ** 2 / (2 * sa ** 2) + (b - mb) ** 2 / (2 * sb ** 2), (a - ma) / sa ** 2, (b - mb) / sb ** 2
        if ul is not None and len(ul):
            z = a + b * ul
            p = sig(z)
            f += float((np.logaddexp(0, z) - yl * z).sum())
            ga += float((p - yl).sum())
            gb += float(((p - yl) * ul).sum())
        if prev is not None:
            pt = sig(a + b * ut)
            m, s = pt.mean(), PREV_REL_SD * prev
            dv = pt * (1 - pt)
            f += (m - prev) ** 2 / (2 * s ** 2)
            ga += (m - prev) / s ** 2 * dv.mean()
            gb += (m - prev) / s ** 2 * (dv * ut).mean()
        return f, (np.array([gb]) if a_fixed is not None else np.array([ga, gb]))

    x0 = np.array([mb]) if a_fixed is not None else np.array([ma, mb])
    bounds = [(0.01, 20)] if a_fixed is not None else [(-15, 15), (0.01, 20)]
    r = minimize(obj, x0, jac=True, method="L-BFGS-B", bounds=bounds)
    return (a_fixed, float(r.x[0])) if a_fixed is not None else (float(r.x[0]), float(r.x[1]))


FLAT = (0.0, 1.0, 100.0, 100.0)  # platt·ts_refit: 사실상 사전 믿음 없음 (분리 방지용 아주 약한 규제)


def metrics(z, y):
    w = np.ones(len(y))
    p = sig(z)
    return np.array([nll(p, y, w), abs(slope_fit(z, y, w)[1] - 1), citl(z, y, w), ece(p, y, w)])


MET = ["nll", "abs_slope_minus1", "intercept", "ece"]


def load_batch(dataset, cond, seed, rho_tab, k):
    ds = DATASETS[dataset]
    lab = pd.read_csv(ds["labels"])
    d = run_dir(cond, seed)
    temp = json.load(open(d / "temperature.json"))
    t = pd.read_csv(d / f"preds_{dataset}.csv").set_index("image_id").loc[lab.image_id]
    out = {}
    for f, m in ds["findings"].items():
        z = t[f"logit_{m}"].values.astype(float)
        q = sig(z / temp[m])
        out[f] = dict(raw=z, ts=logit(q), reader=logit(maj_prob(q, k, rho_tab[(cond, seed, m)])), y=lab[f].values.astype(float))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="*", default=S10)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rho_tab = pd.read_csv(RESULTS / "reader_recal/rho.csv").set_index(["cond", "seed", "finding"]).rho
    rows = []
    for hosp, batches in HOSPITALS.items():
        pairs = [(s, t) for s in batches for t in batches if s != t]
        for cond in CONDS:
            for seed in args.seeds:
                if not all((run_dir(cond, seed) / f"preds_{b}.csv").exists() for b in batches):
                    continue
                for src, tgt in pairs:
                    k = K_PROTOCOL[tgt]
                    S, T = load_batch(src, cond, seed, rho_tab, k), load_batch(tgt, cond, seed, rho_tab, k)
                    rng = np.random.default_rng(1000 * seed + 17 * batches.index(src) + batches.index(tgt))
                    for f in T:
                        if f not in S:
                            continue
                        s_, t_ = S[f], T[f]
                        y = t_["y"]
                        pi = float(s_["y"].mean())
                        base = dict(hospital=hosp, src=src, tgt=tgt, cond=cond, seed=seed, finding=f, prev_src=pi, prev_tgt=float(y.mean()))

                        def add(method, n, vals, tier):
                            vals = np.asarray(vals)
                            if vals.ndim == 1:
                                rows.append({**base, "tier": tier, "method": method, "n": n, **dict(zip(MET, vals)),
                                             **{f"{m}_p10": vals[j] for j, m in enumerate(MET)}, **{f"{m}_p90": vals[j] for j, m in enumerate(MET)}})
                            else:
                                rows.append({**base, "tier": tier, "method": method, "n": n, **dict(zip(MET, vals.mean(0))),
                                             **{f"{m}_p10": np.percentile(vals[:, j], 10) for j, m in enumerate(MET)},
                                             **{f"{m}_p90": np.percentile(vals[:, j], 90) for j, m in enumerate(MET)}})

                        # 정보 없음
                        add("raw", 0, metrics(t_["raw"], y), "none")
                        add("ts", 0, metrics(t_["ts"], y), "none")
                        add("reader", 0, metrics(t_["reader"], y), "none")
                        # 유병률
                        for bname in ["ts", "reader"]:
                            a, b = fit(ut=t_[bname], prev=pi)
                            add(f"prev_{bname}", 0, metrics(a + b * t_[bname], y), "prevalence")
                        for mult in PREV_MULT:
                            pm = float(np.clip(pi * mult, 1e-3, 0.999))
                            a, b = fit(ut=t_["reader"], prev=pm)
                            add(f"prev_reader_x{mult:g}", 0, metrics(a + b * t_["reader"], y), "prevalence_sensitivity")
                        # 참고 하한
                        a, b = fit(ul=t_["raw"], yl=y, prior=FLAT)
                        add("oracle", 0, metrics(a + b * t_["raw"], y), "oracle")
                        # label n장 (기록 묶음에서)
                        for n in NS:
                            if n > len(s_["y"]):
                                continue
                            res = {m: [] for m in ["ts_refit", "platt", "bayes_ts", "bayes_reader", "bayes_reader_prev"]}
                            for _ in range(args.draws):
                                idx = rng.choice(len(s_["y"]), n, replace=False)
                                yl = s_["y"][idx]
                                _, bt = fit(ul=s_["raw"][idx], yl=yl, prior=FLAT, a_fixed=0.0)
                                res["ts_refit"].append(metrics(bt * t_["raw"], y))
                                a, b = fit(ul=s_["raw"][idx], yl=yl, prior=FLAT)
                                res["platt"].append(metrics(a + b * t_["raw"], y))
                                a, b = fit(ul=s_["ts"][idx], yl=yl)
                                res["bayes_ts"].append(metrics(a + b * t_["ts"], y))
                                a, b = fit(ul=s_["reader"][idx], yl=yl)
                                res["bayes_reader"].append(metrics(a + b * t_["reader"], y))
                                a, b = fit(ul=s_["reader"][idx], yl=yl, ut=t_["reader"], prev=pi)
                                res["bayes_reader_prev"].append(metrics(a + b * t_["reader"], y))
                            for m, v in res.items():
                                add(m, n, v, "labels")
                print(f"  {hosp} {cond} seed {seed} 완료", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "all.csv", index=False)

    # 요약: 쌍·조건·seed 평균
    pd.set_option("display.width", 220)
    order = ["raw", "ts", "reader", "prev_ts", "prev_reader"] + [f"prev_reader_x{m:g}" for m in PREV_MULT] + \
            ["ts_refit", "platt", "bayes_ts", "bayes_reader", "bayes_reader_prev", "oracle"]
    for hosp in HOSPITALS:
        for f in res[res.hospital == hosp].finding.unique():
            g = res[(res.hospital == hosp) & (res.finding == f)]
            print(f"\n=== [{hosp} · {f}] 대상 묶음 평균 NLL (괄호 = |slope−1|), 쌍 {g[['src', 'tgt']].drop_duplicates().shape[0]}개 × 조건 {g.cond.nunique()} × seed 평균")
            tab = []
            for m in order:
                h = g[g.method == m]
                if not len(h):
                    continue
                row = {"방법": m}
                for n in sorted(h.n.unique()):
                    hn = h[h.n == n]
                    row["정보 없음/유병률" if n == 0 else f"label {n}장"] = f"{hn.nll.mean():.4f} ({hn.abs_slope_minus1.mean():.2f})"
                tab.append(row)
            print(pd.DataFrame(tab).fillna("").to_string(index=False))
            # 쌍별 판정용: 정보 없음 최선(raw/ts/reader 중) 대비 prev_reader가 좋아진 쌍 수
            piv = g[g.n == 0].groupby(["src", "tgt", "method"]).nll.mean().unstack("method")
            best0 = piv[["raw", "ts", "reader"]].min(axis=1)
            better = int((piv["prev_reader"] < piv["raw"]).sum())
            print(f"  유병률(prev_reader)이 raw보다 NLL 낮은 쌍: {better}/{len(piv)}  |  정보 없음 3개 중 쌍별 최선 평균 {best0.mean():.4f}")
    print(f"\n저장: {OUT / 'all.csv'}")


if __name__ == "__main__":
    main()
