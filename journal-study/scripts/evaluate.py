"""7단계: test 평가 (ANALYSIS_PLAN.md에 고정한 계획대로).

입력  ~/X-ray/runs/findings_gaze-{none,real,shuffled}_t{TH}_seed{0..4}/preds_{val,test}.csv, temperature.json
출력  ~/X-ray/results/eval_t{TH}/
    per_run.csv          조건 × seed(+앙상블) × finding × 지표
    summary.csv          조건 × finding × 지표의 seed 평균 ± 표준편차
    comparisons.csv      real−none, real−shuffled 차이와 환자 단위 bootstrap 95% CI
    referral_curve.csv   referral 0~90% 곡선 (seed 평균)
    subgroups.csv        AP/PA, landscape/portrait
    metadata_baseline.csv, reader_disagreement.csv

사용법
    python evaluate.py                                    # 사전 고정 결과 (seed 0~4, 태그 없음)
    python evaluate.py --name eval_C1 --seeds 0 1 2 3 4 5 6 7 8 9 \
        --tags none=bb-imagenet_lr0.0001_sch-cosine_pool-avg_lam0 real=... shuffled=...

C3처럼 다른 gaze 조건을 비교하려면 "역할=gaze이름:태그"로 쓴다 (역할 none=기준, real=비교 대상, shuffled=대조)
    python evaluate.py --name eval_C3 --seeds 0 ... 9 \
        --tags none=none:bb-..._pool-hybrid3_lam0 real=fs:bb-..._pool-hybrid3_lam0.3 shuffled=fs-shuffled:bb-..._pool-hybrid3_lam0.3

신뢰구간 두 가지 (comparisons.csv)
    ci_low/ci_high     test 환자만 다시 뽑는 bootstrap (사전 계획)
    ci2_low/ci2_high   test 환자 + 조건별 seed를 함께 다시 뽑는 bootstrap (학습 변동 반영, 8절 추가)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HOME = Path.home() / "X-ray"
PROC, RUNS = HOME / "data/processed", HOME / "runs"
CONDS, SEEDS = ["none", "real", "shuffled"], [0, 1, 2, 3, 4]
TAGS = {c: "" for c in ["none", "real", "shuffled"]}
GAZE = {c: c for c in ["none", "real", "shuffled"]}  # 역할 → 실제 gaze 조건 이름 (run 폴더 이름)
FINDINGS, PRIMARY_F = ["pleural", "cardiac", "edema"], ["pleural", "cardiac"]
RATES, PRIMARY_RATE = [0.10, 0.25, 0.30, 0.50], 0.25
CURVE = np.round(np.arange(0, 0.91, 0.05), 2)
SENS_TARGETS = [0.95, 0.98]
COMPARE = [("real", "none"), ("real", "shuffled")]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ------------------------------------------------------------ 불러오기
def load_runs(th: int) -> dict:
    """runs[(cond, seed)][finding] = dict(pv, yv, pt, yt). test는 dicom_id 순서로 정렬."""
    runs, test_ids = {}, None
    for c in CONDS:
        for s in SEEDS:
            d = RUNS / (f"findings_gaze-{GAZE[c]}_t{th}_seed{s}" + (f"_{TAGS[c]}" if TAGS[c] else ""))
            v = pd.read_csv(d / "preds_val.csv").sort_values("dicom_id").reset_index(drop=True)
            t = pd.read_csv(d / "preds_test.csv").sort_values("dicom_id").reset_index(drop=True)
            temp = json.load(open(d / "temperature.json"))
            if test_ids is None:
                test_ids = t[["dicom_id", "subject_id", "ViewPosition"]]
            assert (t.dicom_id.values == test_ids.dicom_id.values).all(), "조건 간 test 구성이 다름"
            runs[(c, s)] = {f: dict(pv=sigmoid(v[f"logit_{f}"].values / temp[f]), yv=v[f"{f}_t{th}"].values,
                                    pt=sigmoid(t[f"logit_{f}"].values / temp[f]), yt=t[f"{f}_t{th}"].values,
                                    pt_raw=sigmoid(t[f"logit_{f}"].values))
                            for f in FINDINGS}
    for c in CONDS:  # seed 5개 앙상블: 보정된 확률의 평균
        runs[(c, "ens")] = {f: {k: np.mean([runs[(c, s)][f][k] for s in SEEDS], axis=0)
                                if k.startswith("p") else runs[(c, SEEDS[0])][f][k]
                                for k in ["pv", "yv", "pt", "yt", "pt_raw"]} for f in FINDINGS}
    return runs, test_ids


# ------------------------------------------------------------ 지표
def referral_arrays(pv, pt, yt, rate):
    conf_v, conf_t = np.maximum(pv, 1 - pv), np.maximum(pt, 1 - pt)
    refer = conf_t <= np.quantile(conf_v, rate) if rate > 0 else np.zeros(len(pt), bool)
    pred = pt >= 0.5
    handled = ~refer
    return dict(refer=refer, handled=handled, fn=handled & (yt == 1) & ~pred,
                fp=handled & (yt == 0) & pred, pos=yt == 1)


def referral_metrics(a, w=None):
    w = np.ones(len(a["refer"])) if w is None else w
    n, pos, handled = w.sum(), (w * a["pos"]).sum(), (w * a["handled"]).sum()
    fn, fp = (w * a["fn"]).sum(), (w * a["fp"]).sum()
    return dict(referral_actual=(w * a["refer"]).sum() / n, fn_count=fn,
                missed_pos_rate=fn / pos if pos else np.nan,
                fn_rate_handled=fn / handled if handled else np.nan,
                selective_risk=(fn + fp) / handled if handled else np.nan)


def sens_workload(pv, yv, pt, yt, target):
    t = np.quantile(pv[yv == 1], 1 - target, method="lower")
    auto_neg = pt < t
    return dict(workload_reduction=auto_neg.mean(), test_sensitivity=1 - auto_neg[yt == 1].mean())


def ece(p, y, bins=15):
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def aurc(p, y):
    conf = np.maximum(p, 1 - p)
    err = ((p >= 0.5) != (y == 1)).astype(float)
    order = np.argsort(-conf)
    risk = np.cumsum(err[order]) / np.arange(1, len(err) + 1)
    opt = np.cumsum(np.sort(err)) / np.arange(1, len(err) + 1)
    return risk.mean(), risk.mean() - opt.mean()


def run_metrics(r):
    pv, yv, pt, yt = r["pv"], r["yv"], r["pt"], r["yt"]
    out = {"auroc": roc_auc_score(yt, pt), "ece_raw": ece(r["pt_raw"], yt), "ece_ts": ece(pt, yt),
           "brier": np.mean((pt - yt) ** 2)}
    out["aurc"], out["e_aurc"] = aurc(pt, yt)
    err = (pt >= 0.5) != (yt == 1)
    out["error_detection_auroc"] = roc_auc_score(err, 1 - np.maximum(pt, 1 - pt)) if 0 < err.sum() < len(err) else np.nan
    for rate in RATES:
        for k, v in referral_metrics(referral_arrays(pv, pt, yt, rate)).items():
            out[f"{k}@{int(rate * 100)}"] = v
    for tg in SENS_TARGETS:
        for k, v in sens_workload(pv, yv, pt, yt, tg).items():
            out[f"{k}@sens{int(tg * 100)}"] = v
    return out


# ------------------------------------------------------------ bootstrap
def bootstrap(runs, test_ids, n_boot, seed=0):
    """환자 bootstrap과 환자+seed bootstrap을 함께 계산한다.
    diff  : 모든 seed 평균의 차이 (환자만 다시 뽑음)
    diff2 : 조건마다 seed도 복원추출한 평균의 차이 (학습 변동까지 반영)"""
    rng = np.random.default_rng(seed)
    pats, pidx = np.unique(test_ids.subject_id.values, return_inverse=True)
    arrays = {(c, s, f, rate): referral_arrays(runs[(c, s)][f]["pv"], runs[(c, s)][f]["pt"], runs[(c, s)][f]["yt"], rate)
              for c in CONDS for s in SEEDS for f in FINDINGS for rate in RATES}
    n_s = len(SEEDS)
    recs = []
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, len(pats), len(pats)), minlength=len(pats))[pidx].astype(float)
        pick = {c: rng.integers(0, n_s, n_s) for c in CONDS}
        for f in FINDINGS:
            per_seed = {}
            for c in CONDS:
                yt = runs[(c, SEEDS[0])][f]["yt"]
                per_seed[(c, "auroc")] = np.array([roc_auc_score(yt, runs[(c, s)][f]["pt"], sample_weight=w) for s in SEEDS])
                for rate in RATES:
                    ms = [referral_metrics(arrays[(c, s, f, rate)], w) for s in SEEDS]
                    for k in ["missed_pos_rate", "fn_count", "fn_rate_handled"]:
                        per_seed[(c, f"{k}@{int(rate * 100)}")] = np.array([m[k] for m in ms])
            for a, bb in COMPARE:
                for metric in {m for _, m in per_seed}:
                    va, vb = per_seed[(a, metric)], per_seed[(bb, metric)]
                    recs.append({"finding": f, "comparison": f"{a}-{bb}", "metric": metric, "b": b,
                                 "diff": va.mean() - vb.mean(),
                                 "diff2": va[pick[a]].mean() - vb[pick[bb]].mean()})
        if (b + 1) % 200 == 0:
            print(f"  bootstrap {b + 1}/{n_boot}")
    return pd.DataFrame(recs)


# ------------------------------------------------------------ 하위 분석
def image_shape(paths):
    return pd.DataFrame([dict(zip(["width", "height"], Image.open(p).size)) for p in paths])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--tags", nargs="*", default=[], help="조건=태그 (sweep.py가 붙인 run 이름 뒤쪽)")
    ap.add_argument("--name", default="", help="결과 폴더 이름 (기본 eval_t{threshold})")
    args = ap.parse_args()
    global SEEDS
    SEEDS = list(args.seeds)
    for item in args.tags:
        c, t = item.split("=", 1)
        if c not in TAGS:
            raise SystemExit(f"역할은 none, real, shuffled 중 하나입니다: {item}")
        if ":" in t:
            GAZE[c], t = t.split(":", 1)
        TAGS[c] = t
    out = HOME / "results" / (args.name or f"eval_t{args.threshold}")
    out.mkdir(parents=True, exist_ok=True)

    runs, test_ids = load_runs(args.threshold)
    print(f"[{out.name}] seed {SEEDS}, 태그 {TAGS}")
    if any(GAZE[c] != c for c in CONDS):
        print("역할 → gaze 조건:", {c: GAZE[c] for c in CONDS})
    print(f"test {len(test_ids)}장, 환자 {test_ids.subject_id.nunique()}명")

    # 1. 실행별 지표
    rows = [{"cond": c, "seed": s, "finding": f, **run_metrics(runs[(c, s)][f])}
            for c in CONDS for s in SEEDS + ["ens"] for f in FINDINGS]
    per_run = pd.DataFrame(rows)
    per_run["gaze"] = per_run.cond.map(GAZE)
    per_run.to_csv(out / "per_run.csv", index=False)
    seeds_only = per_run[per_run.seed != "ens"]
    metrics = [m for m in per_run.columns if m not in ["cond", "seed", "finding", "gaze"]]
    summ = seeds_only.groupby(["finding", "cond"])[metrics].agg(["mean", "std"])
    summ.to_csv(out / "summary.csv")

    # 2. 비교 + bootstrap
    print(f"환자 단위 bootstrap {args.n_boot}회 ...")
    boot = bootstrap(runs, test_ids, args.n_boot)
    point = []
    for f in FINDINGS:
        for a, b in COMPARE:
            for metric in boot.metric.unique():
                col = metric
                d = seeds_only[(seeds_only.finding == f)]
                est = d[d.cond == a][col].mean() - d[d.cond == b][col].mean()
                bd = boot[(boot.finding == f) & (boot.comparison == f"{a}-{b}") & (boot.metric == metric)]["diff"]
                bd2 = boot[(boot.finding == f) & (boot.comparison == f"{a}-{b}") & (boot.metric == metric)]["diff2"]
                point.append({"finding": f, "comparison": f"{a}-{b}", "metric": metric, "estimate": est,
                              "ci_low": bd.quantile(0.025), "ci_high": bd.quantile(0.975),
                              "ci2_low": bd2.quantile(0.025), "ci2_high": bd2.quantile(0.975)})
    comp = pd.DataFrame(point)
    comp["compared"] = comp.comparison.map(lambda x: "-".join(GAZE[k] for k in x.split("-")))
    comp.to_csv(out / "comparisons.csv", index=False)

    # 3. referral 곡선
    curve = []
    for c in CONDS:
        for f in FINDINGS:
            for rate in CURVE:
                ms = [referral_metrics(referral_arrays(runs[(c, s)][f]["pv"], runs[(c, s)][f]["pt"], runs[(c, s)][f]["yt"], rate))
                      for s in SEEDS]
                curve.append({"cond": c, "finding": f, "rate": rate,
                              **{k: np.mean([m[k] for m in ms]) for k in ms[0]}})
    pd.DataFrame(curve).to_csv(out / "referral_curve.csv", index=False)

    # 4. 하위 분석: AP/PA, landscape/portrait
    imgs = pd.read_csv(PROC / "cohort_images_labeled.csv")
    split = pd.read_csv(PROC / "split_seed42.csv")[["dicom_id", "split"]]
    imgs = imgs.merge(split, on="dicom_id")
    imgs = pd.concat([imgs.reset_index(drop=True), image_shape(imgs.image_path)], axis=1)
    imgs["landscape"] = (imgs.width > imgs.height).astype(int)
    imgs["ap"] = (imgs.ViewPosition == "AP").astype(int)
    tmeta = test_ids.merge(imgs[["dicom_id", "landscape", "ap"]], on="dicom_id", how="left")
    sub = []
    for gname, mask in [("AP", tmeta.ap == 1), ("PA", tmeta.ap == 0),
                        ("landscape", tmeta.landscape == 1), ("portrait", tmeta.landscape == 0)]:
        m = mask.values
        for c in CONDS:
            for f in FINDINGS:
                vals = []
                for s in SEEDS:
                    r = runs[(c, s)][f]
                    a = referral_arrays(r["pv"], r["pt"], r["yt"], PRIMARY_RATE)
                    a = {k: v[m] for k, v in a.items()}
                    mm = referral_metrics(a)
                    mm["auroc"] = roc_auc_score(r["yt"][m], r["pt"][m]) if len(set(r["yt"][m])) > 1 else np.nan
                    vals.append(mm)
                sub.append({"group": gname, "n": int(m.sum()), "positives": int(runs[(c, SEEDS[0])][f]["yt"][m].sum()),
                            "cond": c, "finding": f, **{k: np.mean([v[k] for v in vals]) for k in vals[0]}})
    pd.DataFrame(sub).to_csv(out / "subgroups.csv", index=False)

    # 5. 메타데이터 기준선
    tr = imgs[imgs.split == "train"]
    te = imgs.set_index("dicom_id").loc[test_ids.dicom_id]
    meta = []
    for f in FINDINGS:
        X = lambda d: np.c_[d.ap, d.landscape, d.ap * d.landscape]
        lr = LogisticRegression().fit(X(tr), tr[f"{f}_t{args.threshold}"])
        meta.append({"finding": f, "metadata_auroc": roc_auc_score(te[f"{f}_t{args.threshold}"], lr.predict_proba(X(te))[:, 1])})
    pd.DataFrame(meta).to_csv(out / "metadata_baseline.csv", index=False)

    # 6. radiologist 불일치 vs 모델 불확실성 (5명 판독 test 영상)
    multi = imgs[(imgs.split == "test") & (imgs.n_readings >= 5)].set_index("dicom_id")
    pos_in_test = {d: i for i, d in enumerate(test_ids.dicom_id)}
    idx = np.array([pos_in_test[d] for d in multi.index])
    rd = []
    for f in FINDINGS:
        frac = multi[f"{f}_reader_frac"].values
        disagree = ((frac > 0) & (frac < 1)).astype(int)
        for c in CONDS:
            u = np.mean([1 - np.maximum(runs[(c, s)][f]["pt"], 1 - runs[(c, s)][f]["pt"])[idx] for s in SEEDS], axis=0)
            rd.append({"finding": f, "cond": c, "n": len(idx), "disagree_n": int(disagree.sum()),
                       "ap_share": multi.ap.mean(),
                       "uncertainty_auroc_for_disagreement": roc_auc_score(disagree, u) if 0 < disagree.sum() < len(idx) else np.nan,
                       "mean_unc_disagree": u[disagree == 1].mean(), "mean_unc_unanimous": u[disagree == 0].mean()})
    pd.DataFrame(rd).to_csv(out / "reader_disagreement.csv", index=False)

    # ------------------------------------------------------ 화면 요약
    pd.set_option("display.width", 200)
    print("\n=== 주 지표: referral 25%에서 missed-positive rate (seed 평균 ± SD, 낮을수록 좋음)")
    for f in FINDINGS:
        tag = "" if f in PRIMARY_F else " (보조)"
        print(f"\n[{f}{tag}] test 양성 {int(runs[('none', 0)][f]['yt'].sum())}건")
        for c in CONDS:
            d = seeds_only[(seeds_only.finding == f) & (seeds_only.cond == c)]
            e = per_run[(per_run.finding == f) & (per_run.cond == c) & (per_run.seed == "ens")].iloc[0]
            print(f"  {c + ('(' + GAZE[c] + ')' if GAZE[c] != c else ''):9s} missed {d['missed_pos_rate@25'].mean():.3f} ± {d['missed_pos_rate@25'].std():.3f} "
                  f"(놓친 양성 {d['fn_count@25'].mean():.1f}건, 실제 referral {d['referral_actual@25'].mean():.3f}) | "
                  f"AUROC {d.auroc.mean():.3f} ± {d.auroc.std():.3f} | 앙상블 missed {e['missed_pos_rate@25']:.3f}, AUROC {e.auroc:.3f}")
        for a, b in COMPARE:
            r = comp[(comp.finding == f) & (comp.comparison == f"{a}-{b}") & (comp.metric == "missed_pos_rate@25")].iloc[0]
            r2 = comp[(comp.finding == f) & (comp.comparison == f"{a}-{b}") & (comp.metric == "auroc")].iloc[0]
            print(f"  {GAZE[a]}−{GAZE[b]}: missed {r.estimate:+.3f} 환자CI [{r.ci_low:+.3f}, {r.ci_high:+.3f}] 환자+seed CI [{r.ci2_low:+.3f}, {r.ci2_high:+.3f}] | "
                  f"AUROC {r2.estimate:+.3f} 환자CI [{r2.ci_low:+.3f}, {r2.ci_high:+.3f}] 환자+seed CI [{r2.ci2_low:+.3f}, {r2.ci2_high:+.3f}]")
    print("\n=== 메타데이터 기준선 AUROC (촬영 방향 + 사진 모양만)")
    print(pd.DataFrame(meta).round(3).to_string(index=False))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
