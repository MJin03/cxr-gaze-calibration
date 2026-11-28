"""24절 준비 (소견 확장): 다른 소견을 학습·외부 평가할 자료가 충분한지 센다. 학습·예측 결과는 보지 않음. CPU만.

후보 소견 (REFLACX phase 2·3 label 이름 → 판독 녹음 단어 어근 → 외부 label 이름)
세는 것
  ① 내부: REFLACX 영상의 양성 수 (확신도 ≥ 3, 판독 여러 명이면 과반수) — split별. EGD 영상은 이 소견 label이 없어 제외
  ② 시선 지도 가능성: train 양성 영상 중 그 소견 단어를 말한 판독이 있는 비율 (소견별 시선 지도의 coverage)
  ③ 외부: CheXpert test·val, VinDr test·train·새 train, PadChest a·b의 양성 수
판단 규칙 (ANALYSIS_PLAN 24절에 고정): train 양성 ≥ 150, val·test 양성 각 ≥ 30, 양성 coverage ≥ 70%,
  외부 병원 2곳 이상에서 각각 양성 ≥ 50 → 확장 후보

사용법  python count_more_findings.py
"""
import glob

import numpy as np
import pandas as pd

from build_fs_heatmaps import PROC, load_words

RAW = PROC.parent / "raw"
CAND = {  # 이름: (REFLACX label, 단어 어근, CheXpert, VinDr, PadChest label 집합)
    "atelectasis": ("Atelectasis", ["atelecta"], "Atelectasis", "Atelectasis",
                    {"atelectasis", "laminar atelectasis", "lobar atelectasis", "segmental atelectasis", "round atelectasis", "total atelectasis"}),
    "consolidation": ("Consolidation", ["consolid"], "Consolidation", "Consolidation", {"consolidation"}),
    "nodule_mass": ("Lung nodule or mass", ["nodul", "mass"], "Lung Lesion", "Nodule/Mass",
                    {"nodule", "mass", "pulmonary mass", "multiple nodules"}),
    "pneumothorax": ("Pneumothorax", ["pneumothora"], "Pneumothorax", "Pneumothorax", {"pneumothorax"}),
    "emphysema": ("High lung volume / emphysema", ["emphysem", "hyperinfl", "hyperexpan"], None, "Emphysema", {"emphysema"}),
    "interstitial": ("Interstitial lung disease", ["interstit"], None, "ILD", {"interstitial pattern"}),
    "groundglass": ("Groundglass opacity", ["ground", "glass"], None, None, {"ground glass pattern"}),
    "hilum": ("Enlarged hilum", ["hil"], None, None, {"hilar enlargement"}),
    "mediastinum": ("Abnormal mediastinal contour", ["mediastin"], "Enlarged Cardiomediastinum", None, {"mediastinal enlargement"}),
    "fracture": ("Acute fracture", ["fractur"], "Fracture", "Rib fracture", {"fracture", "rib fracture"}),
}


def internal():
    reads = pd.read_csv(PROC / "cohort_readings.csv")
    sp = pd.read_csv(PROC / "split_seed42.csv")
    meta = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(RAW / "reflacx/main_data/metadata_phase_*.csv")))])
    ref = reads[reads.dataset == "reflacx"].merge(meta.drop(columns=["dicom_id", "subject_id"], errors="ignore"),
                                                  left_on="reading_id", right_on="id", how="left").merge(sp, on="dicom_id")
    return ref


def main():
    ref = internal()
    rows = []
    train_reads = ref[ref.split == "train"]
    words = {}
    print(f"train 판독 {len(train_reads)}개의 녹음 단어 읽는 중 ...", flush=True)
    for k, r in enumerate(train_reads.itertuples(), 1):
        w = load_words(r)
        words[r.reading_id] = set(w.word) if len(w) else set()
        if k % 500 == 0:
            print(f"  {k}/{len(train_reads)}", flush=True)
    for name, (col, roots, *_rest) in CAND.items():
        if col not in ref.columns:
            print(f"[주의] REFLACX 열 없음: {col}")
            continue
        v = pd.to_numeric(ref[col], errors="coerce").fillna(0)
        g = (v >= 3).groupby([ref.dicom_id, ref.split]).agg(["sum", "size"])
        img = (g["sum"] > g["size"] / 2).astype(int).reset_index()
        cnt = img.groupby("split")[0].sum().to_dict()
        tr = train_reads[pd.to_numeric(train_reads[col], errors="coerce").fillna(0) >= 3]
        cov = np.mean([any(any(t.startswith(r_) for r_ in roots) for t in words[i]) for i in tr.reading_id]) if len(tr) else np.nan
        rows.append({"소견": name, "REFLACX label": col, "train 양성": cnt.get("train", 0), "val 양성": cnt.get("val", 0),
                     "test 양성": cnt.get("test", 0), "양성 coverage": cov})
    t = pd.DataFrame(rows)

    # 외부
    ext = {}
    for split, ds in [("test", "chexpert_test"), ("val", "chexpert_valid")]:
        lab = pd.read_csv(RAW / f"chexlocalize/chexlocalize/CheXpert/{split}_labels.csv")
        lab = lab[lab["Frontal/Lateral"] == "Frontal"] if "Frontal/Lateral" in lab.columns else lab[lab.Path.str.contains("frontal")]
        ext[ds] = {n: int((lab[c[2]] == 1).sum()) if c[2] and c[2] in lab.columns else None for n, c in CAND.items()}
    vt = pd.read_csv(RAW / "vindr-cxr/annotations/image_labels_test.csv")
    ext["vindr_test"] = {n: int((vt.groupby("image_id")[c[3]].max() > 0).sum()) if c[3] and c[3] in vt.columns else None for n, c in CAND.items()}
    vtr = pd.read_csv(RAW / "vindr-cxr/annotations/image_labels_train.csv")
    for ds, lst in [("vindr_train", "vindr_train3000"), ("vindr_train_b", "vindr_train3000b")]:
        ids = pd.read_csv(PROC / lst / "labels.csv").image_id
        sub = vtr[vtr.image_id.isin(ids)]
        ext[ds] = {n: int((sub.groupby("image_id")[c[3]].mean() > 0.5).sum()) if c[3] and c[3] in sub.columns else None for n, c in CAND.items()}
    try:
        from build_padchest_cache import select
        pc, _ = select(list(range(12)) + [54])
        for b, ds in [("a", "padchest_a"), ("b", "padchest_b")]:
            sub = pc[pc.batch == b]
            ext[ds] = {n: int(sub.labset.map(lambda s, S=c[4]: len(s & S) > 0).sum()) for n, c in CAND.items()}
    except Exception as ex:  # noqa: BLE001
        print(f"[주의] PadChest label을 읽지 못함: {ex}")
    e = pd.DataFrame(ext)
    e.index.name = "소견"

    pd.set_option("display.width", 220)
    print("\n=== ① 내부 (REFLACX 영상, 확신도 ≥ 3) + ② 양성 영상의 소견 단어 coverage")
    print(t.round(3).to_string(index=False))
    print("\n=== ③ 외부 양성 수 (없음 = 그 데이터에 해당 label 없음)")
    print(e.astype("Int64").astype(object).where(e.notna(), "없음").to_string())

    print("\n=== 판단 (train ≥ 150, val·test ≥ 30, coverage ≥ 70%, 외부 병원 2곳 이상에서 양성 ≥ 50)")
    hosp = {"CheXpert": ["chexpert_test", "chexpert_valid"], "VinDr": ["vindr_test", "vindr_train", "vindr_train_b"],
            "PadChest": ["padchest_a", "padchest_b"]}
    for _, r in t.iterrows():
        n = r["소견"]
        cov = 0.0 if pd.isna(r["양성 coverage"]) else r["양성 coverage"]
        ok_int = r["train 양성"] >= 150 and r["val 양성"] >= 30 and r["test 양성"] >= 30 and cov >= 0.7
        hs = [h for h, dss in hosp.items() if any((ext.get(d, {}).get(n) or 0) >= 50 for d in dss)]
        ok = ok_int and len(hs) >= 2
        print(f"  {n:14s} 내부 {'충족' if ok_int else '부족'} | 외부 충족 병원 {hs} → {'✅ 확장 후보' if ok else '제외'}")
    out = PROC / "more_findings_counts.csv"
    t.to_csv(out, index=False)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
