"""E단계: 이전 연구(IWAIT 2027), 사전 고정 설정(1~7절), 개선 설정(8절 C3)을 한 표로 비교한다.

새로 학습·예측하지 않고 저장된 결과만 읽는다.
    이전 연구     : verify_previous_results.py의 보고값 (EGD 3분류, 단일 run)
    3분류 재현    : ~/X-ray/runs/class3_gaze-none_t3_seed*/ (같은 과제를 seed 여러 개로)
    내부 test     : 각 run의 preds_val·preds_test·temperature로 evaluate.run_metrics 재계산 (7·8절과 같은 정의)
    VinDr test    : ~/X-ray/results/eval_vindr/per_run.csv (9절)
    자동 문턱     : ~/X-ray/results/eval_rolling/{internal,vindr}_per_run.csv (10절, W500)

출력  ~/X-ray/results/stageE/
    iwait_vs_reproduction.csv, main_table.csv (seed 평균 ± SD, 앙상블)

사용법  python compare_stageE.py
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from evaluate import run_metrics
from vindr_common import CONDITIONS, RESULTS, RUNS, run_dir
from verify_previous_results import REPORTED_RAW, REPORTED_TS

LABEL = {"pre_none": "pre_none", "pre_real": "pre_real", "pre_shuffled": "pre_shuffled",
         "c1_none": "c1_none", "hyb_none": "hyb_none", "real_global": "real_global",
         "fs": "fs", "fs_shuffled": "fs_shuffled", "ellipse": "ellipse"}
ORDER = list(LABEL)
INTERNAL_F = ["pleural", "cardiac", "edema"]
VINDR_F = ["pleural", "cardiac", "pleural_effusion"]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def class3_reproduction():
    rows = []
    for d in sorted(RUNS.glob("class3_gaze-none_t3_seed*")):
        if not (d / "preds_test.csv").exists():
            continue
        t = pd.read_csv(d / "preds_test.csv")
        temp = json.load(open(d / "temperature.json"))["all"]
        cols = [c for c in t.columns if c.startswith("logit_")]
        z = t[cols].values / temp
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        y = t["class3_idx"].values
        conf, corr = p.max(1), (p.argmax(1) == y).astype(float)
        edges, ece = np.linspace(0, 1, 16), 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.any():
                ece += m.mean() * abs(corr[m].mean() - conf[m].mean())
        bs = np.mean(((p - np.eye(3)[y]) ** 2).sum(1))
        rows.append({"run": d.name, "Macro_AUROC": roc_auc_score(y, p, multi_class="ovr"), "Accuracy": corr.mean(),
                     "ECE_ts": ece, "Brier_ts": bs})
    return pd.DataFrame(rows)


def internal_metrics(allow_missing):
    rows = []
    for c, (_, _, ss) in CONDITIONS.items():
        got = {}
        for s in ss:
            d = run_dir(c, s)
            if not (d / "preds_test.csv").exists():
                if allow_missing:
                    continue
                raise SystemExit(f"없음: {d}")
            v = pd.read_csv(d / "preds_val.csv")
            t = pd.read_csv(d / "preds_test.csv").sort_values("dicom_id").reset_index(drop=True)
            temp = json.load(open(d / "temperature.json"))
            got[s] = {f: dict(pv=sigmoid(v[f"logit_{f}"].values / temp[f]), yv=v[f"{f}_t3"].values,
                              pt=sigmoid(t[f"logit_{f}"].values / temp[f]), yt=t[f"{f}_t3"].values,
                              pt_raw=sigmoid(t[f"logit_{f}"].values)) for f in INTERNAL_F}
        if not got:
            continue
        ens = {f: {k: (np.mean([got[s][f][k] for s in got], axis=0) if k.startswith("p") else got[list(got)[0]][f][k])
                   for k in ["pv", "yv", "pt", "yt", "pt_raw"]} for f in INTERNAL_F}
        for s, r in list(got.items()) + [("ens", ens)]:
            for f in INTERNAL_F:
                rows.append({"cond": c, "seed": str(s), "finding": f, **run_metrics(r[f])})
    return pd.DataFrame(rows)


def summarize(per, metrics, findings, rolling=None):
    """조건 × finding: seed 평균 ± SD와 앙상블 값을 문자열로."""
    per = per.copy()
    per["seed"] = per["seed"].astype(str)
    out = []
    for c in ORDER:
        for f in findings:
            d = per[(per.cond == c) & (per.finding == f)]
            if d.empty:
                continue
            s, e = d[d.seed != "ens"], d[d.seed == "ens"]
            row = {"조건": LABEL[c], "finding": f, "seed 수": len(s)}
            for m, name in metrics:
                row[name] = f"{s[m].mean():.3f} ± {s[m].std():.3f}" + (f" ({e[m].iloc[0]:.3f})" if len(e) else "")
            if rolling is not None:
                r = rolling[(rolling.cond == c) & (rolling.finding == f) & (rolling.window == 500)]
                rs, re = r[r.seed.astype(str) != "ens"], r[r.seed.astype(str) == "ens"]
                if len(rs):
                    row["자동 문턱 missed"] = f"{rs.missed.mean():.3f} ± {rs.missed.std():.3f} ({re.missed.iloc[0]:.3f})"
                    row["자동 문턱 referral"] = f"{rs.referral.mean():.3f}"
            out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-missing", action="store_true", help="동작 확인용")
    ap.add_argument("--only", choices=["all", "iwait", "internal", "vindr"], default="all", help="화면에 출력할 부분 (파일은 모두 저장)")
    args = ap.parse_args()
    out = RESULTS / "stageE"
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 40)

    # 1. 이전 연구 vs 3분류 재현
    rep = pd.DataFrame([{"조건": k, **{f"보고 raw {m}": v for m, v in REPORTED_RAW[k].items()},
                         **{f"보고 TS {m}": v for m, v in REPORTED_TS[k].items()}} for k in REPORTED_RAW])
    c3 = class3_reproduction()
    rep.to_csv(out / "iwait_reported.csv", index=False)
    c3.to_csv(out / "iwait_vs_reproduction.csv", index=False)
    show = (lambda part: args.only in ("all", part))
    if show("iwait"):
        print("=== 1. 이전 연구(IWAIT, EGD 3분류, 단일 run) 보고값")
        print(rep[["조건", "보고 raw Macro_AUROC", "보고 raw Accuracy", "보고 raw ECE", "보고 TS ECE", "보고 TS Brier"]].round(3).to_string(index=False))
    if len(c3) and show("iwait"):
        print(f"\n같은 과제(Baseline)를 seed {len(c3)}개로 재현: Macro AUROC {c3.Macro_AUROC.mean():.3f} ± {c3.Macro_AUROC.std():.3f} "
              f"(범위 {c3.Macro_AUROC.min():.3f}~{c3.Macro_AUROC.max():.3f}), Accuracy {c3.Accuracy.mean():.3f} ± {c3.Accuracy.std():.3f}, "
              f"ECE(TS) {c3.ECE_ts.mean():.3f} ± {c3.ECE_ts.std():.3f}")
        print(c3.round(3).to_string(index=False))

    metrics = [("auroc", "AUROC"), ("ece_ts", "ECE(TS)"), ("brier", "Brier"), ("aurc", "AURC"),
               ("missed_pos_rate@25", "고정 문턱 missed"), ("referral_actual@25", "고정 문턱 referral"),
               ("test_sensitivity@sens95", "민감도95 문턱의 실제 민감도"), ("workload_reduction@sens95", "업무 절감@민감도95")]
    # 2. 내부 test
    internal = internal_metrics(args.allow_missing)
    internal.to_csv(out / "internal_per_run.csv", index=False)
    roll_i = pd.read_csv(RESULTS / "eval_rolling/internal_per_run.csv")
    t_int = summarize(internal, metrics, INTERNAL_F, roll_i)
    # 3. VinDr
    vindr = pd.read_csv(RESULTS / "eval_vindr/per_run.csv")
    roll_v = pd.read_csv(RESULTS / "eval_rolling/vindr_per_run.csv")
    t_vin = summarize(vindr, metrics, VINDR_F, roll_v)
    main_tab = pd.concat([t_int.assign(데이터="내부 test"), t_vin.assign(데이터="VinDr")])
    main_tab.to_csv(out / "main_table.csv", index=False)

    block_a = ["조건", "AUROC", "ECE(TS)", "Brier"]
    block_b = ["조건", "고정 문턱 missed", "고정 문턱 referral", "자동 문턱 missed", "자동 문턱 referral", "민감도95 문턱의 실제 민감도"]
    for name, tab, part in [("2. 내부 test", t_int, "internal"), ("3. VinDr test", t_vin, "vindr")]:
        if not show(part):
            continue
        for f, g in tab.groupby("finding", sort=False):
            g = g.copy()
            for col in ["고정 문턱 referral", "민감도95 문턱의 실제 민감도"]:  # 짧게: 평균만
                g[col] = g[col].str.split(" ").str[0]
            print(f"\n=== {name} [{f}] 분류·보정: seed 평균 ± SD (앙상블)")
            print(g[block_a].to_string(index=False))
            print(f"--- {name} [{f}] selective prediction")
            print(g[[c for c in block_b if c in g.columns]].to_string(index=False))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
