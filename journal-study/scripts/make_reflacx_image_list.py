"""REFLACX 영상의 MIMIC-CXR-JPG 경로 목록을 만든다. EGD-CXR와 겹치는 영상, 이미 받은 영상은 뺀다."""
import glob
from pathlib import Path
import pandas as pd

ROOT = Path.home() / "X-ray"
RAW = ROOT / "data/raw"
ARCHIVE_REFLACX = ROOT / "archive/kim_mj_sens_X-ray/data/reflacx/1.0.0/main_data"
OUT = ROOT / "data/lists/reflacx_jpg_paths.txt"

meta_files = sorted(glob.glob(str(RAW / "reflacx/main_data/metadata_phase_[123].csv")))
if len(meta_files) < 3:
    meta_files = sorted(glob.glob(str(ARCHIVE_REFLACX / "metadata_phase_[123].csv")))
if not meta_files:
    raise SystemExit("REFLACX metadata_phase_*.csv 를 찾지 못했습니다.")
reflacx = pd.concat([pd.read_csv(f, usecols=["id", "dicom_id"]) for f in meta_files], ignore_index=True)
print("REFLACX 메타데이터:", [Path(f).name for f in meta_files])
print("  판독 수:", len(reflacx), "/ 고유 영상 수:", reflacx["dicom_id"].nunique())

meta_path = RAW / "mimic-cxr-jpg/mimic-cxr-2.0.0-metadata.csv.gz"
if not meta_path.exists():
    raise SystemExit(f"{meta_path} 가 없습니다. MIMIC-CXR-JPG 메타데이터를 먼저 받으세요.")
mimic = pd.read_csv(meta_path, usecols=["dicom_id", "subject_id", "study_id"])
egd_ids = set(pd.read_csv(RAW / "egd-cxr/master_sheet.csv", usecols=["dicom_id"])["dicom_id"])

merged = pd.Series(reflacx["dicom_id"].unique(), name="dicom_id").to_frame().merge(mimic, on="dicom_id", how="left")
missing = merged["subject_id"].isna().sum()
if missing:
    raise SystemExit(f"MIMIC-CXR-JPG metadata에서 찾지 못한 dicom_id가 {missing}개 있습니다.")
merged["subject_id"] = merged["subject_id"].astype(int).astype(str)
merged["study_id"] = merged["study_id"].astype(int).astype(str)
merged["path"] = ("files/p" + merged["subject_id"].str[:2] + "/p" + merged["subject_id"]
                  + "/s" + merged["study_id"] + "/" + merged["dicom_id"] + ".jpg")

overlap = merged["dicom_id"].isin(egd_ids)
todo = merged[~overlap]
already = todo["path"].map(lambda p: (RAW / "mimic-cxr-jpg" / p).exists())
todo = todo[~already]
OUT.parent.mkdir(parents=True, exist_ok=True)
todo["path"].to_csv(OUT, index=False, header=False)
print("  EGD-CXR와 겹치는 영상:", int(overlap.sum()))
print("  이미 받은 영상:", int(already.sum()))
print("  새로 받을 영상:", len(todo), "→", OUT)
