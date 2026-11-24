"""16절 평가: 순서형 확신도 출력 (ANALYSIS_PLAN.md 16절, 학습 전에 고정한 규칙 그대로).

단계
  select  문턱 선택 규칙 t*(k): 내부 5명 판독 영상(109장)에서 "k명 과반수 label"에 대한 TS 확률 NLL이 가장 낮은 문턱
          (k = 5: 5명 과반수, k = 3: 5명 중 3명 조합 10개의 과반수 평균, k = 1: 판독 1명씩 평균). 판독 양성 = 확신도 ≥ 3.
          조건·finding별 seed 평균 NLL 기준. 외부 정답 사용 안 함
  noinfo  H5 정보 없음: 외부 7곳에서 ord[t*(k)] (TS) vs ord[≥3] (raw, TS) vs 기존 모델 raw·TS. NLL, slope(부호 있음), 절편, ECE
  prev    H6 유병률: 13절과 같은 같은 병원 쌍. ord_select_prev (평균 예측 확률이 π_기록에 가장 가까운 문턱),
          ord_select_prev_ts (그 문턱 + 13절 유병률 보정) vs 13절 prev_reader (기존 모델, results/bayes_recal)
  disc    H7 판별력 유지: AUROC ord_fs(≥3) − fs (주, 비열등 마진 0.02), ord_fs − ord_c1 (보조).
          데이터별 seed 평균 차이와 ci2(환자 + seed bootstrap), random-effects 통합
  all     위 네 단계를 차례로

사용법  python evaluate_ordinal.py all [--n-boot 1000]
출력    results/ordinal/ (threshold_choice.csv, noinfo.csv, prev.csv, disc.csv)
"""
import argparse
import json
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

from bayes_recalibration import HOSPITALS, fit
from cal_common import DATASETS, NI_MARGIN, RESULTS, S10, run_dir
from evaluate_calibration import AUC, citl, ece, nll, sig, slope_fit
from evaluate_discrimination import dersimonian_laird
from reader_recalibration import K_PROTOCOL, internal_votes

OUT = RESULTS / "ordinal"
PAIRS = {"ord_fs": "fs", "ord_c1": "c1_none"}
TS_ = [3, 4, 5]
INTERNAL_F = ["pleural", "cardiac", "edema"]
DS7 = ["chexpert_test", "vindr_train", "vindr_train_b", "chexpert_valid", "vindr_test", "padchest_a", "padchest_b"]


def col(m, t):
    return f"logit_{m}" if t == 3 else f"logit_{m}_t{t}"


def tkey(m, t):
    return m if t == 3 else f"{m}_t{t}"


def soft_majority(v, k):
    """5명 중 양성 v명일 때, 무작위 k명의 과반수가 양성일 확률 (k = 1, 3, 5)."""
    v = np.asarray(v, int)
    need = k // 2 + 1
    return np.array([sum(comb(x, j) * comb(5 - x, k - j) for j in range(need, k + 1)) / comb(5, k) for x in v])


def zs(cond, seed, dataset, m, t, ts=True, ids=None):
    """조건·seed·데이터의 logit (ts=True면 내부 temperature로 나눔). ids 순서로."""
    d = run_dir(cond, seed)
    f = d / ("preds_test.csv" if dataset == "internal_test" else f"preds_{dataset}.csv")
    key = "dicom_id" if dataset == "internal_test" else "image_id"
    p = pd.read_csv(f).set_index(key)
    if ids is not None:
        p = p.loc[ids]
    z = p[col(m, t)].values.astype(float)
    if ts:
        z = z / json.load(open(d / "temperature.json"))[tkey(m, t)]
    return z


def metrics(z, y):
    w = np.ones(len(y))
    return {"nll": nll(sig(z), y, w), "slope": slope_fit(z, y, w)[1], "intercept": citl(z, y, w), "ece": ece(sig(z), y, w)}


# ------------------------------------------------------------------ select
def do_select(args):
    votes = internal_votes()
    rows = []
    for c in PAIRS:
        for s in S10:
            if not (run_dir(c, s) / "preds_test.csv").exists():
                continue
            t = pd.read_csv(run_dir(c, s) / "preds_test.csv").set_index("dicom_id")
            ids = votes.index.intersection(t.index)
            for f in INTERNAL_F:
                v = votes.loc[ids, f].values
                for k in [1, 3, 5]:
                    h = soft_majority(v, k)
                    for th in TS_:
                        p = np.clip(sig(zs(c, s, "internal_test", f, th, ids=ids)), 1e-6, 1 - 1e-6)
                        rows.append({"cond": c, "seed": s, "finding": f, "k": k, "t": th, "n_img": len(ids),
                                     "nll": float(-(h * np.log(p) + (1 - h) * np.log(1 - p)).mean())})
    if not rows:
        raise SystemExit("순서형 run의 preds_test.csv가 없습니다.")
    r = pd.DataFrame(rows)
    g = r.groupby(["cond", "finding", "k", "t"]).nll.mean().reset_index()
    choice = g.loc[g.groupby(["cond", "finding", "k"]).nll.idxmin()].rename(columns={"t": "t_star", "nll": "nll_best"})
    OUT.mkdir(parents=True, exist_ok=True)
    choice.to_csv(OUT / "threshold_choice.csv", index=False)
    print(f"=== 문턱 선택 t*(k) (내부 5명 판독 {r.n_img.iloc[0]}장, TS 확률 NLL seed 평균; 외부 정답 사용 안 함)")
    tab = g.pivot_table(index=["cond", "finding", "k"], columns="t", values="nll").round(4)
    tab.columns = [f"NLL ≥{c}" for c in tab.columns]
    tab["t*"] = choice.set_index(["cond", "finding", "k"]).t_star
    print(tab.to_string())
    return choice


def tstar(choice, cond, model_f, k):
    return int(choice[(choice.cond == cond) & (choice.finding == model_f) & (choice.k == k)].t_star.iloc[0])


def load_choice():
    return pd.read_csv(OUT / "threshold_choice.csv")


# ------------------------------------------------------------------ noinfo (H5)
def do_noinfo(args):
    choice = load_choice()
    rows = []
    for ds in args.datasets:
        D = DATASETS[ds]
        lab = pd.read_csv(D["labels"])
        k = K_PROTOCOL[ds]
        for oc, base in PAIRS.items():
            for s in S10:
                if not (run_dir(oc, s) / f"preds_{ds}.csv").exists():
                    continue
                for f, m in D["findings"].items():
                    y = lab[f].values.astype(float)
                    ts_ = tstar(choice, oc, m, k)
                    meth = {f"{base}_raw": zs(base, s, ds, m, 3, False, lab.image_id),
                            f"{base}_ts": zs(base, s, ds, m, 3, True, lab.image_id),
                            "ord_t3_raw": zs(oc, s, ds, m, 3, False, lab.image_id)}
                    for th in TS_:
                        meth[f"ord_t{th}_ts"] = zs(oc, s, ds, m, th, True, lab.image_id)
                    meth["ord_sel_ts"] = meth[f"ord_t{ts_}_ts"]
                    for name, z in meth.items():
                        rows.append({"dataset": ds, "k": k, "cond": oc, "seed": s, "finding": f, "t_star": ts_,
                                     "method": name.replace(base, "base"), **metrics(z, y)})
    r = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT / "noinfo.csv", index=False)
    pd.set_option("display.width", 250)
    order = ["base_raw", "base_ts", "ord_t3_raw", "ord_t3_ts", "ord_t4_ts", "ord_t5_ts", "ord_sel_ts"]
    print("\n=== H5 정보 없음: NLL (괄호 = slope, 1보다 크면 확신 부족) — seed 평균. base = 기존 모델 (ord_fs↔fs, ord_c1↔c1_none)")
    for oc in PAIRS:
        g = r[r.cond == oc]
        tab = g.groupby(["dataset", "finding", "method"]).agg(nll=("nll", "mean"), slope=("slope", "mean"), t=("t_star", "first")).reset_index()
        tab["v"] = tab.apply(lambda x: f"{x.nll:.4f} ({x.slope:.2f})", axis=1)
        p = tab.pivot_table(index=["dataset", "finding"], columns="method", values="v", aggfunc="first")[order]
        p["t*"] = tab.groupby(["dataset", "finding"]).t.first()
        print(f"\n[{oc}]")
        print(p.to_string())
        mean = g.groupby(["dataset", "finding", "method"]).nll.mean().unstack()
        win_t3 = (mean.ord_sel_ts < mean.ord_t3_ts)
        win_base = (mean.ord_sel_ts < mean.base_raw)
        both = win_t3 & win_base
        changed = mean.index[(g.groupby(["dataset", "finding"]).t_star.first() != 3).reindex(mean.index).values]
        print(f"  ord_sel이 ord≥3(TS)보다 NLL 낮음: {int(win_t3.sum())}/{len(mean)}, 기존 raw보다 낮음: {int(win_base.sum())}/{len(mean)}, "
              f"둘 다: {int(both.sum())}/{len(mean)}  (문턱이 ≥3이 아닌 칸 {len(changed)}개)")
        print("  H5 판정 (finding별, 데이터 과반수에서 둘 다 낮음):")
        for f, b in both.groupby(level=1):
            print(f"    {f:16s} {int(b.sum())}/{len(b)} → {'충족' if b.sum() > len(b) / 2 else '불충족'}")


# ------------------------------------------------------------------ prev (H6)
def do_prev(args):
    rows = []
    for hosp, batches in HOSPITALS.items():
        pairs = [(a, b) for a in batches for b in batches if a != b]
        for oc, base in PAIRS.items():
            for s in S10:
                if not all((run_dir(oc, s) / f"preds_{b}.csv").exists() for b in batches):
                    continue
                for src, tgt in pairs:
                    ls_, lt = pd.read_csv(DATASETS[src]["labels"]), pd.read_csv(DATASETS[tgt]["labels"])
                    for f, m in DATASETS[tgt]["findings"].items():
                        if f not in DATASETS[src]["findings"]:
                            continue
                        y, pi = lt[f].values.astype(float), float(ls_[f].mean())
                        zt = {th: zs(oc, s, tgt, m, th, True, lt.image_id) for th in TS_}
                        gap = {th: abs(sig(zt[th]).mean() - pi) for th in TS_}
                        tsel = min(gap, key=gap.get)
                        a, b = fit(ut=zt[tsel], prev=pi)
                        a3, b3 = fit(ut=zt[3], prev=pi)
                        base_r = dict(hospital=hosp, src=src, tgt=tgt, cond=oc, seed=s, finding=f, t_sel=tsel)
                        for name, z in [("ord_select_prev", zt[tsel]), ("ord_select_prev_ts", a + b * zt[tsel]),
                                        ("ord_t3_prev", a3 + b3 * zt[3])]:
                            rows.append({**base_r, "method": name, **metrics(z, y)})
    r = pd.DataFrame(rows)
    ref = pd.concat([pd.read_csv(f) for f in [RESULTS / "bayes_recal/all.csv", RESULTS / "bayes_recal/all_PadChest.csv"] if f.exists()])
    ref = ref[(ref.method == "prev_reader") & (ref.n == 0) & ref.cond.isin(PAIRS.values())].copy()
    ref["cond"] = ref.cond.map({v: k for k, v in PAIRS.items()})
    ref = ref.rename(columns={"abs_slope_minus1": "abs_slope"})[["hospital", "src", "tgt", "cond", "seed", "finding", "method", "nll", "intercept", "ece"]]
    allr = pd.concat([r, ref])
    OUT.mkdir(parents=True, exist_ok=True)
    allr.to_csv(OUT / "prev.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n=== H6 유병률: 대상 묶음 NLL (쌍·seed 평균). prev_reader = 13절 기존 모델 결과")
    for oc in PAIRS:
        g = allr[allr.cond == oc]
        mean = g.groupby(["hospital", "finding", "src", "tgt", "method"]).nll.mean().unstack()
        tab = mean.groupby(level=[0, 1]).mean()[["prev_reader", "ord_t3_prev", "ord_select_prev", "ord_select_prev_ts"]]
        tab["쌍(sel_ts<prev_reader)"] = (mean.ord_select_prev_ts < mean.prev_reader).groupby(level=[0, 1]).agg(lambda b: f"{int(b.sum())}/{len(b)}")
        tab["쌍(sel<prev_reader)"] = (mean.ord_select_prev < mean.prev_reader).groupby(level=[0, 1]).agg(lambda b: f"{int(b.sum())}/{len(b)}")
        print(f"\n[{oc}]")
        print(tab.round(4).to_string())
        w1 = (mean.ord_select_prev_ts < mean.prev_reader)
        w2 = (mean.ord_select_prev < mean.prev_reader)
        print(f"  전체 쌍: ord_select_prev_ts가 prev_reader보다 낮음 {int(w1.sum())}/{len(w1)}, ord_select_prev {int(w2.sum())}/{len(w2)}"
              f" → H6 {'충족' if max(w1.sum(), w2.sum()) > len(w1) / 2 else '불충족'}")


# ------------------------------------------------------------------ disc (H7)
def do_disc(args):
    rng = np.random.default_rng(0)
    comps = [("ord_fs", "fs", "noninferiority"), ("ord_fs", "ord_c1", "secondary")]
    rows = []
    for ds in args.datasets:
        D = DATASETS[ds]
        lab = pd.read_csv(D["labels"])
        groups = lab["group"].values if "group" in lab.columns else np.arange(len(lab))
        ug, gi = np.unique(groups, return_inverse=True)
        conds = sorted({c for a, b, _ in comps for c in (a, b)})
        seeds = {c: [s for s in S10 if (run_dir(c, s) / f"preds_{ds}.csv").exists()] for c in conds}
        for f, m in D["findings"].items():
            y = lab[f].values.astype(float)
            aucs = {(c, s): AUC(zs(c, s, ds, m, 3, False, lab.image_id), y) for c in conds for s in seeds[c]}
            one = np.ones(len(y))
            point = {c: np.mean([aucs[(c, s)](one) for s in seeds[c]]) for c in conds}
            boot = {cp: [] for cp in comps}
            for _ in range(args.n_boot):
                w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[gi].astype(float)
                val = {}
                for c in conds:
                    pick = rng.choice(seeds[c], len(seeds[c]))
                    cache = {s: aucs[(c, s)](w) for s in set(pick)}
                    val[c] = np.mean([cache[s] for s in pick])
                for cp in comps:
                    boot[cp].append(val[cp[0]] - val[cp[1]])
            for (a, b, kind) in comps:
                bd = np.array(boot[(a, b, kind)])
                rows.append({"dataset": ds, "finding": f, "comparison": f"{a}-{b}", "kind": kind,
                             "auroc_a": point[a], "auroc_b": point[b], "diff": point[a] - point[b],
                             "low": np.nanquantile(bd, 0.025), "high": np.nanquantile(bd, 0.975), "se": np.nanstd(bd)})
            print(f"  [{ds}] {f} bootstrap {args.n_boot}회 완료", flush=True)
    r = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT / "disc.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"\n=== H7 판별력 (AUROC, ≥3 출력, seed 평균 [ci2]). 비열등 = 통합 CI 하한 > −{NI_MARGIN}")
    for cmp, g in r.groupby("comparison", sort=False):
        print(f"\n[{cmp}]")
        for f, h in g.groupby("finding", sort=False):
            cells = "  ".join(f"{x.dataset} {x.diff:+.3f}[{x.low:+.3f},{x.high:+.3f}]" for x in h.itertuples())
            mu, lo, hi, _, i2 = dersimonian_laird(h["diff"].values, np.maximum(h.se.values, 1e-6))
            ni = "비열등✓" if lo > -NI_MARGIN else "비열등 아님"
            sig_ = "✓" if (lo > 0 or hi < 0) else ""
            print(f"  {f:16s} 통합 {mu:+.3f}[{lo:+.3f},{hi:+.3f}]{sig_} (n={len(h)}, I² {100 * i2:.0f}%) {ni if 'fs-fs' in cmp else ''}")
            print(f"      {cells}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["select", "noinfo", "prev", "disc", "all"])
    ap.add_argument("--datasets", nargs="*", default=DS7)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    if args.action in ("select", "all"):
        do_select(args)
    if args.action in ("noinfo", "all"):
        do_noinfo(args)
    if args.action in ("prev", "all"):
        do_prev(args)
    if args.action in ("disc", "all"):
        do_disc(args)


if __name__ == "__main__":
    main()
