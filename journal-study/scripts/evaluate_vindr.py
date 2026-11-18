"""D단계 3: VinDr-CXR test 외부 검증 평가 (ANALYSIS_PLAN.md 9절, 결과를 보기 전에 고정).

지표 (내부 test와 같은 정의)
    temperature와 referral 기준은 내부 val에서 정한 값을 그대로 쓴다 (VinDr에 맞춰 다시 정하지 않음)
    주 지표: referral 25%에서 missed-positive rate / 주요 보조: AUROC / 그 밖: ECE, Brier, AURC, workload@민감도 95·98%
    seed 평균과 앙상블(seed 확률 평균) 둘 다
    보조: VinDr 자신의 확신도 분포로 정확히 25%를 보내는 경우(label 없이 정할 수 있는 기준)의 missed (missed_local@25)
신뢰구간 (bootstrap 2000회, VinDr는 환자 ID가 없어 영상 단위)
    seed 평균: ci  = 영상만 다시 뽑음 / ci2 = 영상 + 조건별 seed를 함께 다시 뽑음 (주 판정)
    앙상블   : ci  = 영상만 / ci2 = 영상 + 앙상블 구성 seed를 함께 다시 뽑음 (주 판정)
비열등: fs − ellipse의 AUROC 차이 ci2 하한 > −0.02
pointing game (hybrid3 조건): 박스가 있는 양성 영상에서 finding별 attention이 가장 강한 칸이 박스와 겹치면 hit,
    보조로 박스 안 attention 비율(energy)

출력  ~/X-ray/results/eval_vindr/
    per_run.csv, summary.csv, comparisons.csv, pointing.csv, pointing_comparisons.csv, referral_curve.csv

사용법  python evaluate_vindr.py [--n-boot 2000]
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from evaluate import referral_arrays, referral_metrics, run_metrics
from vindr_common import (COMPARISONS, CONDITIONS, FINDINGS_EVAL, FINDINGS_MODEL, HYBRID3, NI_MARGIN, RESULTS,
                          SENSITIVITY, VINDR_DIR, run_dir)

EVAL = FINDINGS_EVAL + list(SENSITIVITY)          # pleural, cardiac, pleural_effusion
MODEL_OUT = {f: SENSITIVITY.get(f, f) for f in EVAL}  # 평가 label → 모델 출력
CURVE = np.round(np.arange(0, 0.91, 0.05), 2)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ------------------------------------------------------------ 불러오기
def load(labels, allow_missing):
    runs, seeds = {}, {}
    for c, (_, _, ss) in CONDITIONS.items():
        ok = []
        for s in ss:
            d = run_dir(c, s)
            if not (d / "preds_vindr.csv").exists():
                if allow_missing:
                    continue
                raise SystemExit(f"VinDr 예측이 없습니다: {d}. predict_vindr.py를 먼저 실행하세요.")
            v = pd.read_csv(d / "preds_val.csv")
            t = pd.read_csv(d / "preds_vindr.csv").set_index("image_id").loc[labels.image_id].reset_index()
            temp = json.load(open(d / "temperature.json"))
            runs[(c, s)] = {}
            for f in EVAL:
                m = MODEL_OUT[f]
                runs[(c, s)][f] = dict(pv=sigmoid(v[f"logit_{m}"].values / temp[m]), yv=v[f"{m}_t3"].values,
                                       pt=sigmoid(t[f"logit_{m}"].values / temp[m]), yt=labels[f].values,
                                       pt_raw=sigmoid(t[f"logit_{m}"].values))
            ok.append(s)
        if ok:
            seeds[c] = ok
    for c, ss in seeds.items():  # 앙상블: 보정된 확률의 seed 평균
        runs[(c, "ens")] = {f: {k: (np.mean([runs[(c, s)][f][k] for s in ss], axis=0) if k.startswith("p")
                                    else runs[(c, ss[0])][f][k]) for k in ["pv", "yv", "pt", "yt", "pt_raw"]}
                            for f in EVAL}
    return runs, seeds


# ------------------------------------------------------------ 빠른 가중 AUROC
class AUC:
    """점수가 고정된 경우: 정렬·동점 묶음을 미리 계산해 bootstrap 가중치마다 O(N)으로 AUROC."""

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
        below = np.cumsum(n) - n
        return float((p * (below + 0.5 * n)).sum() / (tp * tn))


def auc_w(score, y, w):
    return AUC(score, y)(w)


# ------------------------------------------------------------ bootstrap
def bootstrap(runs, seeds, n_boot, rng):
    conds = sorted({c for a, b, _ in COMPARISONS for c in (a, b) if c in seeds})
    n = len(runs[(conds[0], seeds[conds[0]][0])][EVAL[0]]["yt"])
    pre_auc = {(c, s, f): AUC(runs[(c, s)][f]["pt"], runs[(c, s)][f]["yt"]) for c in conds for s in seeds[c] + ["ens"] for f in EVAL}
    pre_ref = {(c, s, f): referral_arrays(runs[(c, s)][f]["pv"], runs[(c, s)][f]["pt"], runs[(c, s)][f]["yt"], 0.25)
               for c in conds for s in seeds[c] + ["ens"] for f in EVAL}
    recs = []
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        pick = {c: rng.choice(seeds[c], len(seeds[c])) for c in conds}
        vals = {}
        for c in conds:
            for f in EVAL:
                per = {s: (referral_metrics(pre_ref[(c, s, f)], w)["missed_pos_rate"], pre_auc[(c, s, f)](w)) for s in seeds[c]}
                allm = np.array([per[s] for s in seeds[c]])
                pk = np.array([per[s] for s in pick[c]])
                # 앙상블: 전체 seed / 다시 뽑은 seed로 다시 평균
                e_all = (referral_metrics(pre_ref[(c, "ens", f)], w)["missed_pos_rate"], pre_auc[(c, "ens", f)](w))
                pv = np.mean([runs[(c, s)][f]["pv"] for s in pick[c]], axis=0)
                pt = np.mean([runs[(c, s)][f]["pt"] for s in pick[c]], axis=0)
                yt = runs[(c, "ens")][f]["yt"]
                e_pk = (referral_metrics(referral_arrays(pv, pt, yt, 0.25), w)["missed_pos_rate"], auc_w(pt, yt, w))
                vals[(c, f)] = dict(seed=allm.mean(0), seed2=pk.mean(0), ens=np.array(e_all), ens2=np.array(e_pk))
        for a, bb, kind in COMPARISONS:
            if a not in seeds or bb not in seeds:
                continue
            for f in EVAL:
                va, vb = vals[(a, f)], vals[(bb, f)]
                for j, metric in enumerate(["missed_pos_rate@25", "auroc"]):
                    recs.append({"b": b, "comparison": f"{a}-{bb}", "finding": f, "metric": metric,
                                 "seed_ci": va["seed"][j] - vb["seed"][j], "seed_ci2": va["seed2"][j] - vb["seed2"][j],
                                 "ens_ci": va["ens"][j] - vb["ens"][j], "ens_ci2": va["ens2"][j] - vb["ens2"][j]})
        if (b + 1) % 200 == 0:
            print(f"  bootstrap {b + 1}/{n_boot}", flush=True)
    return pd.DataFrame(recs)


# ------------------------------------------------------------ pointing game
def box_cover(boxes, g):
    """512 좌표 박스들의 합집합을 g×g 칸 비율로 (칸마다 박스가 덮는 면적 비율)."""
    m = np.zeros((512, 512), bool)
    for r in boxes.itertuples():
        m[int(np.floor(r.y_min_512)):int(np.ceil(r.y_max_512)), int(np.floor(r.x_min_512)):int(np.ceil(r.x_max_512))] = True
    k = 512 // g
    return m.reshape(g, k, g, k).mean(axis=(1, 3))


def pointing(seeds, labels, boxes, n_boot, rng):
    rows, per_img = [], {}
    for fi, f in enumerate(["pleural", "cardiac"]):
        ids = labels.image_id[(labels[f] == 1) & labels.image_id.isin(boxes.loc[boxes.finding == f, "image_id"])].tolist()
        pos = {i: k for k, i in enumerate(labels.image_id)}
        idx = np.array([pos[i] for i in ids])
        bf = boxes[boxes.finding == f].groupby("image_id")
        cover = None
        for c in HYBRID3:
            if c not in seeds:
                continue
            sm_all = []
            for s in seeds[c]:
                att = np.load(run_dir(c, s) / "att_vindr.npy").astype(np.float32)[idx, FINDINGS_MODEL.index(f)]
                g = att.shape[-1]
                if cover is None:
                    cover = np.stack([box_cover(bf.get_group(i), g) for i in ids])
                flat = att.reshape(len(ids), -1)
                sm = np.exp(flat - flat.max(1, keepdims=True))
                sm /= sm.sum(1, keepdims=True)
                sm_all.append(sm)
                hit = cover.reshape(len(ids), -1)[np.arange(len(ids)), flat.argmax(1)] > 0
                energy = (sm * cover.reshape(len(ids), -1)).sum(1)
                per_img[(c, s, f)] = (hit.astype(float), energy)
            ens = np.mean(sm_all, axis=0)
            per_img[(c, "ens", f)] = ((cover.reshape(len(ids), -1)[np.arange(len(ids)), ens.argmax(1)] > 0).astype(float),
                                      (ens * cover.reshape(len(ids), -1)).sum(1))
            h = [per_img[(c, s, f)][0].mean() for s in seeds[c]]
            e = [per_img[(c, s, f)][1].mean() for s in seeds[c]]
            rows.append({"cond": c, "finding": f, "n_images": len(ids), "hit_mean": np.mean(h), "hit_sd": np.std(h, ddof=1),
                         "energy_mean": np.mean(e), "energy_sd": np.std(e, ddof=1),
                         "hit_ens": per_img[(c, "ens", f)][0].mean(), "energy_ens": per_img[(c, "ens", f)][1].mean()})
    # fs − 다른 hybrid3 조건 (영상 + seed bootstrap)
    comp = []
    for f in ["pleural", "cardiac"]:
        for other in [c for c in HYBRID3 if c != "fs" and c in seeds]:
            if "fs" not in seeds:
                break
            n = len(per_img[("fs", seeds["fs"][0], f)][0])
            d = {"hit": [], "energy": []}
            for _ in range(n_boot):
                ii = rng.integers(0, n, n)
                pa, pb = rng.choice(seeds["fs"], len(seeds["fs"])), rng.choice(seeds[other], len(seeds[other]))
                for j, m in enumerate(["hit", "energy"]):
                    d[m].append(np.mean([per_img[("fs", s, f)][j][ii].mean() for s in pa])
                                - np.mean([per_img[(other, s, f)][j][ii].mean() for s in pb]))
            for j, m in enumerate(["hit", "energy"]):
                est = (np.mean([per_img[("fs", s, f)][j].mean() for s in seeds["fs"]])
                       - np.mean([per_img[(other, s, f)][j].mean() for s in seeds[other]]))
                comp.append({"comparison": f"fs-{other}", "finding": f, "metric": m, "estimate": est,
                             "ci2_low": np.quantile(d[m], 0.025), "ci2_high": np.quantile(d[m], 0.975)})
    return pd.DataFrame(rows), pd.DataFrame(comp)


# ------------------------------------------------------------ 메인
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--name", default="eval_vindr")
    ap.add_argument("--allow-missing", action="store_true", help="동작 확인용")
    args = ap.parse_args()
    out = RESULTS / args.name
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    labels = pd.read_csv(VINDR_DIR / "labels.csv")
    boxes = pd.read_csv(VINDR_DIR / "boxes.csv")
    choice = json.load(open(VINDR_DIR / "preprocess_choice.json"))["choice"]
    runs, seeds = load(labels, args.allow_missing)
    print(f"[{out.name}] VinDr test {len(labels)}장, 전처리 {choice}, 조건 {len(seeds)}개")
    print("  양성:", {f: int(labels[f].sum()) for f in EVAL})

    # 1. run별 지표
    def metrics_row(r):
        m = run_metrics(r)
        loc = referral_metrics(referral_arrays(r["pt"], r["pt"], r["yt"], 0.25))
        m["missed_local@25"], m["referral_local@25"] = loc["missed_pos_rate"], loc["referral_actual"]
        return m
    rows = [{"cond": c, "seed": s, "finding": f, **metrics_row(runs[(c, s)][f])}
            for c in seeds for s in seeds[c] + ["ens"] for f in EVAL]
    per_run = pd.DataFrame(rows)
    per_run.to_csv(out / "per_run.csv", index=False)
    metrics = [m for m in per_run.columns if m not in ["cond", "seed", "finding"]]
    so = per_run[per_run.seed != "ens"]
    so.groupby(["finding", "cond"])[metrics].agg(["mean", "std"]).to_csv(out / "summary.csv")

    # 2. referral 곡선 (seed 평균)
    curve = []
    for c in seeds:
        for f in EVAL:
            for rate in CURVE:
                ms = [referral_metrics(referral_arrays(runs[(c, s)][f]["pv"], runs[(c, s)][f]["pt"], runs[(c, s)][f]["yt"], rate))
                      for s in seeds[c]]
                curve.append({"cond": c, "finding": f, "rate": rate, **{k: np.mean([m[k] for m in ms]) for k in ms[0]}})
    pd.DataFrame(curve).to_csv(out / "referral_curve.csv", index=False)

    # 3. 비교 + bootstrap
    print(f"영상 단위 bootstrap {args.n_boot}회 ...")
    boot = bootstrap(runs, seeds, args.n_boot, rng)
    comp = []
    for a, b, kind in COMPARISONS:
        if a not in seeds or b not in seeds:
            continue
        for f in EVAL:
            for metric in ["missed_pos_rate@25", "auroc"]:
                bd = boot[(boot.comparison == f"{a}-{b}") & (boot.finding == f) & (boot.metric == metric)]
                for level in ["seed", "ens"]:
                    if level == "seed":
                        est = so[(so.cond == a) & (so.finding == f)][metric].mean() - so[(so.cond == b) & (so.finding == f)][metric].mean()
                    else:
                        g = per_run[(per_run.seed == "ens") & (per_run.finding == f)].set_index("cond")[metric]
                        est = g[a] - g[b]
                    r = {"comparison": f"{a}-{b}", "kind": kind, "finding": f, "metric": metric, "level": level,
                         "estimate": est, "ci_low": bd[f"{level}_ci"].quantile(0.025), "ci_high": bd[f"{level}_ci"].quantile(0.975),
                         "ci2_low": bd[f"{level}_ci2"].quantile(0.025), "ci2_high": bd[f"{level}_ci2"].quantile(0.975)}
                    if kind == "noninferiority" and metric == "auroc":
                        r["noninferior"] = bool(r["ci2_low"] > -NI_MARGIN)
                    comp.append(r)
    comp = pd.DataFrame(comp)
    comp.to_csv(out / "comparisons.csv", index=False)

    # 4. pointing game
    print("pointing game ...")
    pt, ptc = pointing(seeds, labels, boxes, args.n_boot, rng)
    pt.to_csv(out / "pointing.csv", index=False)
    ptc.to_csv(out / "pointing_comparisons.csv", index=False)

    # ------------------------------------------------------ 화면 요약
    pd.set_option("display.width", 220)
    print("\n=== 조건별 결과 (seed 평균 ± SD | 앙상블). missed = referral 25%에서 놓친 양성 비율 (낮을수록 좋음)")
    for f in EVAL:
        tag = " (민감도 분석: effusion만)" if f in SENSITIVITY else ""
        print(f"\n[{f}{tag}] VinDr 양성 {int(labels[f].sum())}건")
        for c in seeds:
            d = so[(so.cond == c) & (so.finding == f)]
            e = per_run[(per_run.cond == c) & (per_run.finding == f) & (per_run.seed == "ens")].iloc[0]
            print(f"  {c:13s} missed {d['missed_pos_rate@25'].mean():.3f} ± {d['missed_pos_rate@25'].std():.3f} "
                  f"(실제 referral {d['referral_actual@25'].mean():.3f}; VinDr 기준 25%면 {d['missed_local@25'].mean():.3f}) | AUROC {d.auroc.mean():.3f} ± {d.auroc.std():.3f} "
                  f"| 앙상블 missed {e['missed_pos_rate@25']:.3f}, AUROC {e.auroc:.3f}")
    print("\n=== 비교 (차이 = 앞 − 뒤, ci2 = 영상+seed bootstrap. missed는 음수, AUROC는 양수가 앞 조건이 좋은 것)")
    for f in EVAL:
        print(f"\n[{f}]")
        for (cmp, kind), g in comp[comp.finding == f].groupby(["comparison", "kind"], sort=False):
            parts = []
            for metric, short in [("missed_pos_rate@25", "missed"), ("auroc", "AUROC")]:
                for level, lv in [("seed", "seed평균"), ("ens", "앙상블")]:
                    r = g[(g.metric == metric) & (g.level == level)].iloc[0]
                    star = " ✓" if (r.ci2_low > 0 or r.ci2_high < 0) else ""
                    parts.append(f"{short}({lv}) {r.estimate:+.3f} [{r.ci2_low:+.3f}, {r.ci2_high:+.3f}]{star}")
            ni = ""
            if kind == "noninferiority":
                rr = g[(g.metric == "auroc") & (g.level == "seed")].iloc[0]
                ni = f" → 비열등(마진 {NI_MARGIN}): {'예' if rr.noninferior else '아니오'}"
            print(f"  {cmp:26s} [{kind}] " + " | ".join(parts[:2]) + "\n" + " " * 30 + " | ".join(parts[2:]) + ni)
    print("\n=== pointing game (박스 있는 양성 영상, attention 최대 칸이 박스와 겹치는 비율 / 박스 안 attention 비율)")
    print(pt.round(3).to_string(index=False))
    print("\nfs − 다른 조건 (영상+seed bootstrap):")
    print(ptc.round(3).to_string(index=False))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
