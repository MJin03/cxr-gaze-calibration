"""10절 (사후 분석): 순위 기반 자동 문턱(rolling quantile referral)으로 selective prediction을 다시 평가한다.

배경: VinDr에서 내부 val로 고정한 확신도 문턱은 실제로 2.5~12.5%만 보내 비교가 불가능했다 (9절 결과).
규칙: "최근 들어온 영상 W장의 확신도 중 하위 25%에 들면 의사에게 보낸다". 병원별 설정·label이 필요 없다.
    - 처음 설치할 때는 그 run의 내부 val 확신도(보정 후)에서 W장을 뽑아 시작 (복원추출)
    - 영상은 무작위 순서로 들어온다고 보고 시뮬레이션, 순서를 R번(기본 20) 바꿔 평균
    - 과거 영상의 확신도만 쓰므로 평가 영상 자신을 보고 문턱을 정하지 않는다 (label 사용 없음)
    - W = 500 (주), 200·1000 (민감도 분석). 모든 조건이 같은 순서(같은 무작위 순서)를 쓴다
지표: 실제 referral 비율, missed-positive rate (AI가 처리한 영상 중 놓친 양성 / 전체 양성)
    순서 R번의 평균은 "영상별 referral 확률"로 정확히 계산되므로 bootstrap은 영상(또는 환자) 가중치로 빠르게 한다
신뢰구간: 내부 = 환자 단위, VinDr = 영상 단위 bootstrap 2000회
    seed 평균: ci(영상/환자만), ci2(+ 조건별 seed 복원추출) / 앙상블: ci(영상/환자만)

출력  ~/X-ray/results/eval_rolling/{internal,vindr}_per_run.csv, comparisons.csv

사용법  python evaluate_rolling.py [--windows 500 200 1000] [--orders 20]
"""
import argparse
import json

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from vindr_common import COMPARISONS, CONDITIONS, RESULTS, VINDR_DIR, run_dir

RATE = 0.25
SETS = {"internal": ["pleural", "cardiac", "edema"], "vindr": ["pleural", "cardiac", "pleural_effusion"]}
MODEL_OUT = {"pleural": "pleural", "cardiac": "cardiac", "edema": "edema", "pleural_effusion": "pleural"}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load(allow_missing):
    """data[set][(cond, seed)][finding] = dict(cv=val 확신도, p=평가 확률, y=label). 앙상블 포함."""
    vlab = pd.read_csv(VINDR_DIR / "labels.csv")
    data = {"internal": {}, "vindr": {}}
    groups = {}
    seeds = {}
    for c, (_, _, ss) in CONDITIONS.items():
        ok = []
        for s in ss:
            d = run_dir(c, s)
            if not ((d / "preds_test.csv").exists() and (d / "preds_vindr.csv").exists()):
                if allow_missing:
                    continue
                raise SystemExit(f"예측 파일이 없습니다: {d}")
            temp = json.load(open(d / "temperature.json"))
            v = pd.read_csv(d / "preds_val.csv")
            t = pd.read_csv(d / "preds_test.csv").sort_values("dicom_id").reset_index(drop=True)
            x = pd.read_csv(d / "preds_vindr.csv").set_index("image_id").loc[vlab.image_id].reset_index()
            groups.setdefault("internal", t.subject_id.values)
            groups.setdefault("vindr", np.arange(len(x)))
            for name, df, lab in [("internal", t, t), ("vindr", x, vlab)]:
                data[name][(c, s)] = {}
                for f in SETS[name]:
                    m = MODEL_OUT[f]
                    pv = sigmoid(v[f"logit_{m}"].values / temp[m])
                    y = lab[f"{f}_t3"].values if name == "internal" else lab[f].values
                    data[name][(c, s)][f] = dict(cv=np.maximum(pv, 1 - pv), p=sigmoid(df[f"logit_{m}"].values / temp[m]), y=y)
            ok.append(s)
        if ok:
            seeds[c] = ok
    for name in data:  # 앙상블: 보정 확률의 seed 평균 (val도 같은 방식)
        for c, ss in seeds.items():
            data[name][(c, "ens")] = {}
            for f in SETS[name]:
                m = MODEL_OUT[f]
                pv = np.mean([sigmoid(pd.read_csv(run_dir(c, s) / "preds_val.csv")[f"logit_{m}"].values
                                      / json.load(open(run_dir(c, s) / "temperature.json"))[m]) for s in ss], axis=0)
                data[name][(c, "ens")][f] = dict(cv=np.maximum(pv, 1 - pv), p=np.mean([data[name][(c, s)][f]["p"] for s in ss], axis=0),
                                                 y=data[name][(c, ss[0])][f]["y"])
    return data, seeds, groups


def referral_prob(cv, p, w, orders, rng_seed):
    """영상별로 '의사에게 보내질 확률' (무작위 순서 orders개 평균)."""
    rng = np.random.default_rng(rng_seed)
    conf = np.maximum(p, 1 - p)
    n = len(conf)
    out = np.zeros(n)
    for _ in range(orders):
        perm = rng.permutation(n)
        init = rng.choice(cv, w, replace=True)
        a = np.concatenate([init, conf[perm]])
        win = sliding_window_view(a, w)[:n]           # 영상 k 직전의 최근 w장
        thr = np.quantile(win, RATE, axis=1)
        refer = np.zeros(n, bool)
        refer[perm] = conf[perm] <= thr
        out += refer
    return out / orders


def metrics(r, p, y, wt=None):
    wt = np.ones(len(y)) if wt is None else wt
    pos = (y == 1).astype(float)
    miss = pos * (p < 0.5) * (1 - r)
    return dict(referral=(wt * r).sum() / wt.sum(), missed=(wt * miss).sum() / max((wt * pos).sum(), 1e-12),
                fn=(wt * miss).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, nargs="*", default=[500, 200, 1000])
    ap.add_argument("--orders", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--allow-missing", action="store_true", help="동작 확인용")
    args = ap.parse_args()
    out = RESULTS / "eval_rolling"
    out.mkdir(parents=True, exist_ok=True)
    data, seeds, groups = load(args.allow_missing)
    rng = np.random.default_rng(0)
    comp_rows = []
    pd.set_option("display.width", 220)

    for name in ["internal", "vindr"]:
        rows, rprob = [], {}
        for wi, w in enumerate(args.windows):
            for c in seeds:
                for s in seeds[c] + ["ens"]:
                    for f in SETS[name]:
                        d = data[name][(c, s)][f]
                        # 같은 (창 크기, finding)에서는 모든 조건이 같은 무작위 순서를 쓰도록 seed 고정
                        r = referral_prob(d["cv"], d["p"], w, args.orders, rng_seed=1000 * wi + SETS[name].index(f))
                        rprob[(w, c, s, f)] = r
                        rows.append({"window": w, "cond": c, "seed": str(s), "finding": f, **metrics(r, d["p"], d["y"])})
        per = pd.DataFrame(rows)
        per.to_csv(out / f"{name}_per_run.csv", index=False)

        # bootstrap (주 창 W = 첫 번째)
        w0 = args.windows[0]
        g = groups[name]
        ug, gi = np.unique(g, return_inverse=True)
        recs = []
        for b in range(args.n_boot):
            wt = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[gi].astype(float)
            pick = {c: rng.choice(seeds[c], len(seeds[c])) for c in seeds}
            for a, bb, kind in COMPARISONS:
                if a not in seeds or bb not in seeds:
                    continue
                for f in SETS[name]:
                    val = {}
                    for c in (a, bb):
                        m = {s: metrics(rprob[(w0, c, s, f)], data[name][(c, s)][f]["p"], data[name][(c, s)][f]["y"], wt)["missed"]
                             for s in seeds[c] + ["ens"]}
                        val[c] = (np.mean([m[s] for s in seeds[c]]), np.mean([m[s] for s in pick[c]]), m["ens"])
                    recs.append({"comparison": f"{a}-{bb}", "kind": kind, "finding": f,
                                 "seed_ci": val[a][0] - val[bb][0], "seed_ci2": val[a][1] - val[bb][1], "ens_ci": val[a][2] - val[bb][2]})
            if (b + 1) % 500 == 0:
                print(f"  [{name}] bootstrap {b + 1}/{args.n_boot}", flush=True)
        boot = pd.DataFrame(recs)
        p0 = per[per.window == w0]
        so = p0[p0.seed != "ens"]
        for (cmp, kind, f), bd in boot.groupby(["comparison", "kind", "finding"], sort=False):
            a, bb = cmp.split("-")
            est_s = so[(so.cond == a) & (so.finding == f)].missed.mean() - so[(so.cond == bb) & (so.finding == f)].missed.mean()
            e = p0[(p0.seed == "ens") & (p0.finding == f)].set_index("cond").missed
            comp_rows.append({"set": name, "comparison": cmp, "kind": kind, "finding": f, "window": w0,
                              "seed_estimate": est_s, "seed_ci_low": bd.seed_ci.quantile(0.025), "seed_ci_high": bd.seed_ci.quantile(0.975),
                              "seed_ci2_low": bd.seed_ci2.quantile(0.025), "seed_ci2_high": bd.seed_ci2.quantile(0.975),
                              "ens_estimate": e[a] - e[bb], "ens_ci_low": bd.ens_ci.quantile(0.025), "ens_ci_high": bd.ens_ci.quantile(0.975)})

        # 화면 요약
        title = "내부 test (환자 단위 CI)" if name == "internal" else "VinDr test (영상 단위 CI)"
        print(f"\n==================== {title}: 순위 기반 자동 문턱, 하위 {int(RATE * 100)}% referral")
        for f in SETS[name]:
            print(f"\n[{f}] 양성 {int(data[name][(list(seeds)[0], seeds[list(seeds)[0]][0])][f]['y'].sum())}건")
            for c in seeds:
                txt = []
                for w in args.windows:
                    d = per[(per.window == w) & (per.cond == c) & (per.finding == f)]
                    s_ = d[d.seed != "ens"]
                    e = d[d.seed == "ens"].iloc[0]
                    txt.append(f"W{w}: missed {s_.missed.mean():.3f}±{s_.missed.std():.3f} (ref {s_.referral.mean():.3f}) ens {e.missed:.3f}")
                print(f"  {c:13s} " + " | ".join(txt))
        cr = pd.DataFrame([r for r in comp_rows if r["set"] == name])
        print(f"\n비교 (W{w0}, missed 차이 = 앞 − 뒤, 음수면 앞 조건이 덜 놓침, ✓ = CI가 0을 포함하지 않음)")
        for r in cr.itertuples():
            s2 = " ✓" if (r.seed_ci2_low > 0 or r.seed_ci2_high < 0) else ""
            se = " ✓" if (r.ens_ci_low > 0 or r.ens_ci_high < 0) else ""
            print(f"  {r.finding:16s} {r.comparison:24s} seed평균 {r.seed_estimate:+.3f} ci2 [{r.seed_ci2_low:+.3f}, {r.seed_ci2_high:+.3f}]{s2}"
                  f" | 앙상블 {r.ens_estimate:+.3f} ci [{r.ens_ci_low:+.3f}, {r.ens_ci_high:+.3f}]{se}")
    pd.DataFrame(comp_rows).to_csv(out / "comparisons.csv", index=False)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
