"""1단계: EGD-CXR와 REFLACX를 하나의 목록으로 합친다.

판독 단위 표와 영상 단위 표 두 개를 만든다. 라벨은 원래 값 그대로 둔다.
양성 기준(확신도 3 이상 등)은 2단계에서 정한다.

출력
    ~/X-ray/data/processed/cohort_readings.csv   판독 1건 = 1행
    ~/X-ray/data/processed/cohort_images.csv     영상 1장 = 1행

라벨 열 (원래 값)
    pleural_raw   EGD: pleural_effusion_or_thickening (0/1)
                  REFLACX phase 1: max(Pleural effusion, Pleural thickening) (0~5)
                  REFLACX phase 2, 3: Pleural abnormality (0~5)
    cardiac_raw   EGD: enlarged_cardiac_silhouette / REFLACX: Enlarged cardiac silhouette
    edema_raw     EGD: pulmonary_edema__hazy_opacity / REFLACX: Pulmonary edema
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path.home() / "X-ray"
RAW = ROOT / "data/raw"
OUT = ROOT / "data/processed"
IMAGE_DIRS = [RAW / "mimic-cxr-jpg/files", RAW / "mimic-cxr-jpg-egdcxr"]
CLASS3 = ["Normal", "CHF", "pneumonia"]


def index_images() -> dict:
    """dicom_id -> jpg 경로. 두 영상 폴더를 모두 찾는다."""
    index = {}
    for d in IMAGE_DIRS:
        for p in d.rglob("*.jpg"):
            index.setdefault(p.stem, str(p))
    return index


def ids_from_path(path: str) -> tuple:
    m = re.search(r"p(\d+)/s(\d+)/", path)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def load_egd(images: dict) -> pd.DataFrame:
    m = pd.read_csv(RAW / "egd-cxr/master_sheet.csv")
    fix_ids = set(pd.read_csv(RAW / "egd-cxr/fixations.csv", usecols=["DICOM_ID"])["DICOM_ID"])
    subj_study = m["path"].map(ids_from_path)
    df = pd.DataFrame({
        "dataset": "egd",
        "reading_id": "EGD_" + m["dicom_id"],
        "phase": 0,
        "dicom_id": m["dicom_id"],
        "subject_id": [s for s, _ in subj_study],
        "study_id": [t for _, t in subj_study],
        "gaze_file": str(RAW / "egd-cxr/fixations.csv"),
        "gaze_ok": m["dicom_id"].isin(fix_ids),
        "label_scale": "binary",
        "pleural_raw": m["pleural_effusion_or_thickening"],
        "cardiac_raw": m["enlarged_cardiac_silhouette"],
        "edema_raw": m["pulmonary_edema__hazy_opacity"],
        "class3": m[CLASS3].idxmax(axis=1).where(m[CLASS3].sum(axis=1) == 1),
    })
    mismatch = (df["subject_id"] != m["patient_id"]).sum()
    if mismatch:
        print(f"[주의] EGD patient_id와 경로의 subject_id가 다른 행: {mismatch}개")
    return df


def load_reflacx(mimic_meta: pd.DataFrame) -> pd.DataFrame:
    base = RAW / "reflacx/main_data"
    parts = []
    for ph in [1, 2, 3]:
        m = pd.read_csv(base / f"metadata_phase_{ph}.csv")
        pleural = m[["Pleural effusion", "Pleural thickening"]].max(axis=1) if ph == 1 else m["Pleural abnormality"]
        gaze = m["id"].map(lambda i: str(base / i / "fixations.csv"))
        parts.append(pd.DataFrame({
            "dataset": "reflacx",
            "reading_id": m["id"],
            "phase": ph,
            "dicom_id": m["dicom_id"],
            "subject_id": m["subject_id"],
            "gaze_file": gaze,
            "gaze_ok": (~m["eye_tracking_data_discarded"].astype(bool)) & gaze.map(lambda p: Path(p).exists()),
            "label_scale": "certainty_0_5",
            "pleural_raw": pleural,
            "cardiac_raw": m["Enlarged cardiac silhouette"],
            "edema_raw": m["Pulmonary edema"],
            "class3": None,
        }))
    df = pd.concat(parts, ignore_index=True)
    df = df.merge(mimic_meta[["dicom_id", "study_id"]], on="dicom_id", how="left")
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    images = index_images()
    mimic_meta = pd.read_csv(RAW / "mimic-cxr-jpg/mimic-cxr-2.0.0-metadata.csv.gz",
                             usecols=["dicom_id", "subject_id", "study_id", "ViewPosition"])

    readings = pd.concat([load_egd(images), load_reflacx(mimic_meta)], ignore_index=True)
    readings["image_path"] = readings["dicom_id"].map(images)
    readings["image_ok"] = readings["image_path"].notna()
    readings = readings.merge(mimic_meta[["dicom_id", "ViewPosition"]], on="dicom_id", how="left")
    cols = ["dataset", "reading_id", "phase", "dicom_id", "subject_id", "study_id", "ViewPosition",
            "image_path", "image_ok", "gaze_file", "gaze_ok", "label_scale",
            "pleural_raw", "cardiac_raw", "edema_raw", "class3"]
    readings = readings[cols]
    readings.to_csv(OUT / "cohort_readings.csv", index=False)

    img = (readings.groupby("dicom_id")
           .agg(subject_id=("subject_id", "first"), study_id=("study_id", "first"),
                ViewPosition=("ViewPosition", "first"), image_path=("image_path", "first"),
                image_ok=("image_ok", "first"),
                datasets=("dataset", lambda s: "+".join(sorted(set(s)))),
                n_readings=("reading_id", "size"), n_gaze=("gaze_ok", "sum"))
           .reset_index())
    img.to_csv(OUT / "cohort_images.csv", index=False)

    # 요약
    print("=== 판독 단위")
    print(readings.groupby(["dataset", "phase"]).agg(판독=("reading_id", "size"), 시선있음=("gaze_ok", "sum"),
                                                    영상있음=("image_ok", "sum")).to_string())
    print("\n=== 영상 단위")
    print("  전체 영상:", len(img), "/ 영상 파일 없음:", int((~img["image_ok"]).sum()))
    print("  데이터셋별:", img["datasets"].value_counts().to_dict())
    print("  판독 수별 영상:", img["n_readings"].value_counts().sort_index().to_dict())
    print("  촬영 방향(ViewPosition):", img["ViewPosition"].value_counts(dropna=False).to_dict())
    egd_s = set(readings.loc[readings.dataset == "egd", "subject_id"])
    ref_s = set(readings.loc[readings.dataset == "reflacx", "subject_id"])
    print("\n=== 환자")
    print("  전체 환자:", img["subject_id"].nunique(), f"(EGD {len(egd_s)}, REFLACX {len(ref_s)}, 두 데이터셋 모두 {len(egd_s & ref_s)})")
    print("  환자당 영상 수:", img.groupby("subject_id").size().value_counts().sort_index().to_dict())
    print("\n=== 라벨 원래 값 분포 (판독 단위)")
    for lab in ["pleural_raw", "cardiac_raw", "edema_raw"]:
        for ds in ["egd", "reflacx"]:
            vc = readings.loc[readings.dataset == ds, lab].value_counts(dropna=False).sort_index()
            print(f"  {lab:12s} {ds:8s}", {int(k) if pd.notna(k) else "NaN": int(v) for k, v in vc.items()})
    print(f"\n저장: {OUT / 'cohort_readings.csv'}\n      {OUT / 'cohort_images.csv'}")


if __name__ == "__main__":
    main()
