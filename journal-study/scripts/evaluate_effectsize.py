"""26절: 시선 이득 크기가 왜 소견·병원마다 다른가 (ANALYSIS_PLAN.md 26절, 실행 전에 고정). 학습·예측 없음, CPU만.

A. 이득 크기 설명 분석 (meta-regression)
   단위: 모델 집단 × 외부 데이터 × 소견의 AUROC 이득 (이미 계산된 eval_disc_* 결과)
     모델 집단  imagenet (fs − c1_none, eval_disc_7sites) / nih (nih_fs − nih_c1, eval_disc_nih) /
                convnext (cnx_fs − cnx_c1, eval_disc_cnx) / more (more_fs − more_c1, eval_disc_more_ext)
     소견       흉막, 심장비대, 폐부종, 무기폐, 경화 (하위 label pleural_effusion·pleural_cpa·edema_hi는 중복이라 제외)
     이득의 표준오차 = ci2 폭 / 3.92
   설명 변수 (결과를 보기 전에 고정)
     base_auc   대조 모델(시선 없음)의 그 데이터·소견 AUROC (seed 평균) → 천장 효과
     report     label 방식: PadChest(판독문 기반, 의사 1명) = 1, 그 외(전문의 판정·합의·과반수) = 0
     area       병변 퍼짐: CheXlocalize CheXpert test 전문의 분할의 평균 면적 비율 (소견별 한 값, 20·25-2절 '무작위 기준 energy')
     + 모델 집단 더미 (집단마다 기본 이득이 다를 수 있음)
   모형: random-effects meta-regression (가중치 1/(se² + τ²), τ²는 DerSimonian–Laird 방식)
   가설: M1 base_auc 계수 < 0 (천장 효과), M2 report 계수 < 0 (판독문 label에서 이득이 작음), M3 area 계수 < 0 (퍼진 소견에서 이득이 작음)
         → 95% CI 상한 < 0이면 지지. 같은 영상을 여러 모델이 공유하므로 보조로 데이터 단위 cluster bootstrap CI(2,000회)도 보고
   보조 결과: 같은 분석을 시선 − 다른 환자 시선(fs − fs_shuffled 계열)에도

B. PadChest 엄격 label 민감도 분석 (무기폐)
   PadChest 무기폐 label 중 'laminar atelectasis'(얇은 판상 무기폐, 경미한 소견)만 있는 영상은 제외하고,
   나머지 무기폐 label(atelectasis, lobar, segmental, round, total)이 있는 영상만 양성 → padchest/labels_more_strict_{a,b}.csv
   평가: evaluate_discrimination.py --cond-set more --datasets padchest_a_more_strict padchest_b_more_strict --tag more_strict
   가설 S1 (기술적): 엄격 label에서 more_fs − more_c1 이득이 원래 label보다 큼 → "판독문 label의 노이즈가 이득을 줄였다"를 뒷받침

사용법  python evaluate_effectsize.py strict     (B의 label 파일 만들기)
        python evaluate_effectsize.py meta       (A, 결과 results/effectsize/)
"""
import argparse
import sys

import numpy as np
import pandas as pd

from cal_common import PROC, RESULTS

OUT = RESULTS / "effectsize"
FAMILIES = {  # 집단: (결과 폴더, 주 비교, 보조 비교, 대조 조건)
    "imagenet": ("eval_disc_7sites", "fs-c1_none", "fs-fs_shuffled", "c1_none"),
    "nih": ("eval_disc_nih", "nih_fs-nih_c1", "nih_fs-nih_fs_shuffled", "nih_c1"),
    "convnext": ("eval_disc_cnx", "cnx_fs-cnx_c1", "cnx_fs-cnx_fs_shuffled", "cnx_c1"),
    "more": ("eval_disc_more_ext", "more_fs-more_c1", "more_fs-more_fs_shuffled", "more_c1"),
}
FIND5 = ["pleural", "cardiac", "edema", "atelectasis", "consolidation"]
AREA_SRC = {"pleural": ("localization", "pleural"), "cardiac": ("localization", "cardiac"), "edema": ("localization", "edema"),
            "atelectasis": ("localization_more", "atelectasis"), "consolidation": ("localization_more", "consolidation")}
AT_ALL = {"atelectasis", "laminar atelectasis", "lobar atelectasis", "segmental atelectasis", "round atelectasis", "total atelectasis"}
N_BOOT = 2000


# ------------------------------------------------------------ B
def strict():
    from build_padchest_cache import select
    pc, _ = select(list(range(12)) + [54])
    pc = pc.assign(image_id=pc.ImageID.map(lambda s: s.rsplit(".", 1)[0])).set_index("image_id")
    rows = []
    for b in ["a", "b"]:
        base = pd.read_csv(PROC / f"padchest/labels_{b}.csv")[["image_id", "group"]]
        ls = base.image_id.map(pc.labset).map(lambda s: s if isinstance(s, set) else set())
        at = ls.map(lambda s: s & AT_ALL)
        laminar_only = at.map(lambda s: s == {"laminar atelectasis"})
        pos = at.map(lambda s: len(s - {"laminar atelectasis"}) > 0)
        out = base[~laminar_only.values].assign(atelectasis=pos[~laminar_only.values].astype(int).values)
        out.to_csv(PROC / f"padchest/labels_more_strict_{b}.csv", index=False)
        rows.append({"batch": b, "영상": len(base), "원래 양성": int(at.map(bool).sum()), "laminar만 (제외)": int(laminar_only.sum()),
                     "엄격 양성": int(out.atelectasis.sum()), "평가 영상": len(out)})
    print("=== PadChest 무기폐 엄격 label (laminar만 있는 영상 제외)")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"저장: {PROC / 'padchest/labels_more_strict_a.csv'}, labels_more_strict_b.csv")


# ------------------------------------------------------------ A
def lesion_area():
    area = {}
    for f, (folder, name) in AREA_SRC.items():
        p = RESULTS / folder / "summary.csv"
        if not p.exists():
            print(f"[주의] {p} 없음 → {f} 면적 없음")
            continue
        s = pd.read_csv(p)
        r = s[s.dataset.str.startswith("chexpert_test") & (s.finding == name) & (s.cond == "무작위 기준")]
        if len(r):
            area[f] = float(r.energy.iloc[0])
    return area


def load_units(which):
    units = []
    for fam, (folder, prim, sec, ctrl) in FAMILIES.items():
        d = RESULTS / folder
        if not (d / "comparisons_all.csv").exists():
            print(f"[주의] {d / 'comparisons_all.csv'} 없음 → {fam} 제외")
            continue
        c = pd.read_csv(d / "comparisons_all.csv")
        cmp = prim if which == "primary" else sec
        c = c[(c.comparison == cmp) & (c.metric == "auroc") & c.finding.isin(FIND5)]
        base = {}
        for f in sorted(d.glob("per_run_*.csv")):
            pr = pd.read_csv(f)
            pr = pr[(pr.cond == ctrl) & (pr.seed.astype(str) != "ens")]
            for (ds, fi), g in pr.groupby(["dataset", "finding"]):
                base[(ds, fi)] = g.auroc.mean()
        for r in c.itertuples():
            units.append({"family": fam, "dataset": r.dataset, "site": r.dataset.replace("_more", ""), "finding": r.finding,
                          "gain": r.seed_estimate, "se": max((r.seed_ci2_high - r.seed_ci2_low) / 3.92, 1e-4),
                          "base_auc": base.get((r.dataset, r.finding), np.nan),
                          "report": int(r.dataset.startswith("padchest")), "pos": r.pos})
    return pd.DataFrame(units)


def design(u, cols):
    fams = sorted(u.family.unique())
    X = np.column_stack([np.ones(len(u))] + [u[c].values.astype(float) for c in cols]
                        + [(u.family == f).astype(float).values for f in fams[1:]])
    return X, ["절편"] + cols + [f"집단:{f}" for f in fams[1:]]


def meta_reg(y, se, X):
    """random-effects meta-regression (DerSimonian–Laird 방식 τ²). 계수, 표준오차, τ²."""
    w = 1 / se ** 2
    W = np.diag(w)
    XtWX_inv = np.linalg.pinv(X.T @ W @ X)
    b = XtWX_inv @ X.T @ W @ y
    q = float(((y - X @ b) ** 2 * w).sum())
    k, p = X.shape
    P = W - W @ X @ XtWX_inv @ X.T @ W
    tau2 = max(0.0, (q - (k - p)) / np.trace(P)) if k > p else 0.0
    w2 = 1 / (se ** 2 + tau2)
    W2 = np.diag(w2)
    cov = np.linalg.pinv(X.T @ W2 @ X)
    b2 = cov @ X.T @ W2 @ y
    return b2, np.sqrt(np.diag(cov)), tau2


def fit(u, cols, rng):
    v = u.dropna(subset=cols + ["gain", "se"])
    X, names = design(v, cols)
    b, se, tau2 = meta_reg(v.gain.values, v.se.values, X)
    # 데이터 단위 cluster bootstrap (같은 영상을 여러 모델·소견이 공유)
    sites = v.site.unique()
    boots = []
    for _ in range(N_BOOT):
        pick = rng.choice(sites, len(sites))
        vb = pd.concat([v[v.site == s] for s in pick])
        if vb.report.nunique() < 2 and "report" in cols:
            continue
        Xb, _ = design(vb, cols)
        if Xb.shape[1] != X.shape[1] or np.linalg.matrix_rank(Xb) < Xb.shape[1]:
            continue
        boots.append(meta_reg(vb.gain.values, vb.se.values, Xb)[0])
    boots = np.array(boots)
    rows = []
    for j, n in enumerate(names):
        lo, hi = b[j] - 1.96 * se[j], b[j] + 1.96 * se[j]
        blo, bhi = (np.quantile(boots[:, j], [0.025, 0.975]) if len(boots) else (np.nan, np.nan))
        rows.append({"변수": n, "계수": b[j], "CI 하한": lo, "CI 상한": hi, "boot 하한": blo, "boot 상한": bhi})
    return pd.DataFrame(rows), tau2, len(v), len(boots)


def meta():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    area = lesion_area()
    print("병변 평균 면적 비율 (CheXpert test 전문의 분할): " + ", ".join(f"{f} {a:.3f}" for f, a in area.items()))
    pd.set_option("display.width", 200)
    for which in ["primary", "secondary"]:
        u = load_units(which)
        if u.empty:
            print(f"[주의] {which}: 단위 없음")
            continue
        u["area"] = u.finding.map(area)
        u.to_csv(OUT / f"units_{which}.csv", index=False)
        title = "시선 − 시선 없음" if which == "primary" else "시선 − 다른 환자 시선 (보조)"
        print(f"\n=== [{title}] 단위 {len(u)}개 (모델 집단 × 데이터 × 소견)")
        print(u.groupby(["family", "finding"]).agg(n=("gain", "size"), 이득평균=("gain", "mean"), 기본AUROC=("base_auc", "mean"))
              .round(3).to_string())
        print("\n소견·label 방식별 평균 이득 (집단 합침):")
        print(u.pivot_table(index="finding", columns="report", values="gain", aggfunc="mean")
              .rename(columns={0: "전문의 label", 1: "판독문 label(PadChest)"}).round(3).to_string())
        res = []
        for label, cols in [("다변수 (주)", ["base_auc", "report", "area"]), ("단변수 base_auc", ["base_auc"]),
                            ("단변수 report", ["report"]), ("단변수 area", ["area"])]:
            t, tau2, n, nb = fit(u, cols, rng)
            t.insert(0, "모형", label)
            res.append(t)
            print(f"\n[{label}] 단위 {n}개, τ² {tau2:.2e}, bootstrap 유효 {nb}/{N_BOOT}")
            show = t[~t.변수.str.startswith("집단") & (t.변수 != "절편")].copy()
            for c in ["base_auc", "area"]:   # 0.1 단위로 (AUROC 0.1, 면적 비율 10%p)
                show.loc[show.변수 == c, ["계수", "CI 하한", "CI 상한", "boot 하한", "boot 상한"]] *= 0.1
            show["변수"] = show.변수.replace({"base_auc": "base_auc (0.1 높을 때)", "report": "report (판독문 label)",
                                              "area": "area (면적 10%p 넓을 때)"})
            show["판정"] = np.where(show["CI 상한"] < 0, "지지(음수)", np.where(show["CI 하한"] > 0, "반대(양수)", "불분명"))
            print(show.round(4).to_string(index=False))
        pd.concat(res).to_csv(OUT / f"metareg_{which}.csv", index=False)
        try:
            plot(u, which)
        except Exception as ex:  # noqa: BLE001
            print(f"[주의] 그림 실패: {ex}")
    print(f"\n저장: {OUT}")


def plot(u, which):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mk = {"pleural": "o", "cardiac": "s", "edema": "^", "atelectasis": "D", "consolidation": "P"}
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for (f, rep), g in u.groupby(["finding", "report"]):
        ax.errorbar(g.base_auc, g.gain, yerr=1.96 * g.se, fmt=mk.get(f, "o"), ms=6, alpha=0.75, capsize=0,
                    color="#c0392b" if rep else "#2c7fb8", mfc="white" if rep else None,
                    label=f"{f} ({'report' if rep else 'expert'})")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("baseline AUROC (no-gaze model)")
    ax.set_ylabel("AUROC gain from gaze" + (" (vs no gaze)" if which == "primary" else " (vs shuffled gaze)"))
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / f"gain_vs_baseline_{which}.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["strict", "meta"])
    args = ap.parse_args()
    {"strict": strict, "meta": meta}[args.action]()


if __name__ == "__main__":
    sys.exit(main())
