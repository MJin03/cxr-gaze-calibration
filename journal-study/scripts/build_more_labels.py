"""25절 소견 확장: 무기폐(atelectasis)·경화(consolidation) label과 셔플 매핑을 만든다 (ANALYSIS_PLAN.md 25절). CPU만, 학습·예측 결과는 보지 않음.

내부 (REFLACX 영상만. EGD 영상에는 이 소견 label이 없음)
    판독의 확신도 ≥ 3이면 그 판독에서 양성, 판독이 여러 명이면 과반수 (24절 count_more_findings.py와 같은 규칙)
    → ~/X-ray/data/processed/more_labels.csv           (dicom_id, subject_id, split, datasets, atelectasis_t3, consolidation_t3, n_readers)
    → ~/X-ray/data/processed/shuffle_map_more_seed42.csv (fs-shuffled 대조용: 이 코호트 train 영상끼리, 다른 환자에게서 일대일)
    → ~/X-ray/data/processed/more_internal_test/labels.csv (내부 test 평가용: image_id = dicom_id, group = 환자)

외부 (24절과 같은 label 이름)
    CheXpert test·val  test_labels.csv / val_labels.csv의 Atelectasis, Consolidation (전문의 label, 정면만)
    VinDr test         image_labels_test.csv (합의 label) > 0
    VinDr train·새 train  image_labels_train.csv, 판독의 3명 과반수
    PadChest a·b       판독문 label 집합 (24절 CAND와 같은 집합)
    → 각 폴더의 labels_more.csv (PadChest는 labels_more_a.csv, labels_more_b.csv)
    평가에 쓰는 소견 = 그 데이터에서 양성 ≥ 30인 것만 (25절에 사전 고정). 30 미만인 소견은 열을 만들지 않음 (개수는 출력)

사용법  python build_more_labels.py
"""
import os

import numpy as np
import pandas as pd

from count_more_findings import CAND, RAW, internal
from build_fs_heatmaps import PROC

MORE = ["atelectasis", "consolidation"]
MIN_POS = int(os.environ.get("MORE_MIN_POS_TEST", 30))  # 동작 확인용 환경변수 (실제 실행에서는 비어 있음 → 30)
SEED = 42


def shuffle_map(train: pd.DataFrame, seed: int) -> pd.DataFrame:
    """일대일 셔플: 모든 영상이 다른 환자의 지도를 정확히 하나씩 받는다 (build_heatmaps.py와 같은 방법)."""
    rng = np.random.default_rng(seed)
    ids = train.dicom_id.values
    subj = train.subject_id.values
    for _ in range(1000):
        perm = rng.permutation(len(ids))
        bad = np.where(subj[perm] == subj)[0]
        for i in bad:
            for j in rng.permutation(len(ids)):
                if subj[perm[j]] != subj[i] and subj[perm[i]] != subj[j]:
                    perm[i], perm[j] = perm[j], perm[i]
                    break
        if (subj[perm] != subj).all():
            return pd.DataFrame({"dicom_id": ids, "donor_dicom_id": ids[perm]})
    raise RuntimeError("셔플 매핑을 만들지 못했습니다")


def build_internal():
    ref = internal()
    out = None
    for f in MORE:
        v = pd.to_numeric(ref[CAND[f][0]], errors="coerce").fillna(0)
        g = (v >= 3).groupby(ref.dicom_id).agg(["sum", "size"])
        col = pd.DataFrame({"dicom_id": g.index, f"{f}_t3": (g["sum"] > g["size"] / 2).astype(int).values,
                            "n_readers": g["size"].values})
        out = col if out is None else out.merge(col.drop(columns="n_readers"), on="dicom_id")
    imgs = pd.read_csv(PROC / "cohort_images.csv")[["dicom_id", "subject_id", "datasets"]]
    sp = pd.read_csv(PROC / f"split_seed{SEED}.csv")[["dicom_id", "split"]]
    out = out.merge(imgs, on="dicom_id").merge(sp, on="dicom_id")
    out = out[["dicom_id", "subject_id", "split", "datasets"] + [f"{f}_t3" for f in MORE] + ["n_readers"]]
    out.to_csv(PROC / "more_labels.csv", index=False)

    reads = pd.read_csv(PROC / "cohort_readings.csv")
    has_gaze = set(reads.loc[reads.gaze_ok.astype(bool), "dicom_id"])
    tr = out[(out.split == "train") & out.dicom_id.isin(has_gaze)].sort_values("dicom_id").reset_index(drop=True)
    smap = shuffle_map(tr, SEED)
    smap.to_csv(PROC / f"shuffle_map_more_seed{SEED}.csv", index=False)
    chk = smap.merge(tr[["dicom_id", "subject_id"]], on="dicom_id").merge(
        tr[["dicom_id", "subject_id"]].rename(columns={"dicom_id": "donor_dicom_id", "subject_id": "donor_subject"}), on="donor_dicom_id")

    te = out[out.split == "test"]
    d = PROC / "more_internal_test"
    d.mkdir(exist_ok=True)
    pd.DataFrame({"image_id": te.dicom_id, "group": te.subject_id, **{f: te[f"{f}_t3"] for f in MORE}}).to_csv(d / "labels.csv", index=False)

    print("=== 내부 (REFLACX 영상, 확신도 ≥ 3 과반수)")
    t = out.groupby("split")[[f"{f}_t3" for f in MORE]].agg(["sum", "size"])
    print(t.to_string())
    print(f"셔플 매핑: train {len(smap)}장, 같은 환자 배정 {int((chk.subject_id == chk.donor_subject).sum())}건, "
          f"donor 중복 {int(smap.donor_dicom_id.duplicated().sum())}건")
    return {"내부 test": {f: int(te[f"{f}_t3"].sum()) for f in MORE}}


def write_ext(name, path, base, pos, counts):
    """base: image_id(+group) 표, pos: 소견 → 양성(0/1) Series (base 순서). 양성 ≥ 30인 소견만 열로."""
    out = base.copy()
    counts[name] = {}
    for f in MORE:
        n = int(pos[f].sum())
        counts[name][f] = n
        if n >= MIN_POS:
            out[f] = pos[f].astype(int).values
    out.to_csv(path, index=False)


def build_external():
    counts = {}
    # CheXpert
    from build_chexpert_cache import image_id as cx_id
    for split, ds in [("test", "chexpert_test"), ("val", "chexpert_valid")]:
        base = pd.read_csv(PROC / ds / "labels.csv")[["image_id", "group"]]
        lab = pd.read_csv(RAW / f"chexlocalize/chexlocalize/CheXpert/{split}_labels.csv")
        lab = lab[lab.Path.str.contains("frontal")].copy()
        lab["image_id"] = lab.Path.map(cx_id)
        m = base.merge(lab.drop_duplicates("image_id"), on="image_id", how="left")
        if m[CAND["atelectasis"][2]].isna().any():
            print(f"[주의] {ds}: 원본 label과 맞지 않는 영상 {int(m[CAND['atelectasis'][2]].isna().sum())}장 (음성 처리)")
        write_ext(ds, PROC / ds / "labels_more.csv", base,
                  {f: (m[CAND[f][2]] == 1) for f in MORE}, counts)
    # VinDr test (합의 label)
    base = pd.read_csv(PROC / "vindr_test/labels.csv")[["image_id"]]
    vt = pd.read_csv(RAW / "vindr-cxr/annotations/image_labels_test.csv").groupby("image_id").max()
    write_ext("vindr_test", PROC / "vindr_test/labels_more.csv", base,
              {f: (base.image_id.map(vt[CAND[f][3]]).fillna(0) > 0) for f in MORE}, counts)
    # VinDr train (판독의 3명 과반수)
    vtr = pd.read_csv(RAW / "vindr-cxr/annotations/image_labels_train.csv")
    for ds, folder in [("vindr_train", "vindr_train3000"), ("vindr_train_b", "vindr_train3000b")]:
        base = pd.read_csv(PROC / folder / "labels.csv")[["image_id"]]
        g = vtr[vtr.image_id.isin(base.image_id)].groupby("image_id")
        n = g.size()
        write_ext(ds, PROC / folder / "labels_more.csv", base,
                  {f: base.image_id.map(g[CAND[f][3]].apply(lambda s: (s > 0).sum()) > n / 2).fillna(False) for f in MORE}, counts)
    # PadChest
    try:
        from build_padchest_cache import select
        pc, _ = select(list(range(12)) + [54])
        pc = pc.assign(image_id=pc.ImageID.map(lambda s: s.rsplit(".", 1)[0])).set_index("image_id")
        for b in ["a", "b"]:
            base = pd.read_csv(PROC / f"padchest/labels_{b}.csv")[["image_id", "group"]]
            ls = base.image_id.map(pc.labset)
            if ls.isna().any():
                print(f"[주의] padchest_{b}: label 집합을 찾지 못한 영상 {int(ls.isna().sum())}장 (음성 처리)")
            ls = ls.map(lambda s: s if isinstance(s, set) else set())
            write_ext(f"padchest_{b}", PROC / f"padchest/labels_more_{b}.csv", base,
                      {f: ls.map(lambda s, S=CAND[f][4]: len(s & S) > 0) for f in MORE}, counts)
    except Exception as ex:  # noqa: BLE001
        print(f"[주의] PadChest label을 만들지 못함: {ex}")
    return counts


def main():
    counts = build_internal()
    counts.update(build_external())
    c = pd.DataFrame(counts).T
    c.index.name = "데이터"
    show = c.astype(object)
    for f in MORE:
        show[f] = [f"{v} {'평가' if v >= MIN_POS else f'제외(<{MIN_POS})'}" for v in c[f]]
    pd.set_option("display.width", 200)
    print(f"\n=== 평가용 양성 수 (양성 ≥ {MIN_POS}인 소견만 평가)")
    print(show.to_string())
    c.to_csv(PROC / "more_eval_counts.csv")
    print(f"\n저장: {PROC / 'more_labels.csv'}, {PROC / f'shuffle_map_more_seed{SEED}.csv'}, {PROC / 'more_internal_test/labels.csv'},"
          f"\n      각 외부 폴더의 labels_more*.csv, {PROC / 'more_eval_counts.csv'}")


if __name__ == "__main__":
    main()
