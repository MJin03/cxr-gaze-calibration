"""2단계: 판독 단위 원래 라벨을 영상 단위 양성/음성으로 바꾼다.

규칙
    EGD-CXR (0/1)          : 1이면 양성
    REFLACX (확신도 0~5)    : 확신도 >= 기준이면 그 판독은 양성
                             기준 3 = 주 분석, 기준 4 = 민감도 분석
    판독의 여러 명(109장)   : 과반수(5명 중 3명 이상)가 양성이면 양성
    EGD와 REFLACX 둘 다     : REFLACX 판독의 확신도 라벨을 따른다 (10장)

입력  ~/X-ray/data/processed/cohort_readings.csv, cohort_images.csv
출력  ~/X-ray/data/processed/cohort_images_labeled.csv
"""
from pathlib import Path

import pandas as pd

PROC = Path.home() / "X-ray/data/processed"
LABELS = {"pleural": "흉막 이상", "cardiac": "심장비대", "edema": "폐부종"}
THRESHOLDS = [3, 4]


def reading_positive(df: pd.DataFrame, lab: str, t: int) -> pd.Series:
    raw = df[f"{lab}_raw"]
    return ((df["dataset"] == "egd") & (raw == 1)) | ((df["dataset"] == "reflacx") & (raw >= t))


def main() -> None:
    reads = pd.read_csv(PROC / "cohort_readings.csv")
    imgs = pd.read_csv(PROC / "cohort_images.csv")

    # EGD와 REFLACX가 둘 다 있는 영상은 REFLACX 판독만 라벨에 쓴다
    has_ref = set(reads.loc[reads.dataset == "reflacx", "dicom_id"])
    use = reads[~((reads.dataset == "egd") & reads.dicom_id.isin(has_ref))].copy()

    for lab in LABELS:
        for t in THRESHOLDS:
            use["pos"] = reading_positive(use, lab, t)
            g = use.groupby("dicom_id")["pos"].agg(["sum", "size"])
            imgs = imgs.merge((g["sum"] > g["size"] / 2).astype(int).rename(f"{lab}_t{t}"),
                              left_on="dicom_id", right_index=True, how="left")
            if t == 3:
                imgs = imgs.merge((g["sum"] / g["size"]).rename(f"{lab}_reader_frac"),
                                  left_on="dicom_id", right_index=True, how="left")
    egd_class = reads.loc[reads.dataset == "egd", ["dicom_id", "class3"]]
    imgs = imgs.merge(egd_class, on="dicom_id", how="left")
    imgs.to_csv(PROC / "cohort_images_labeled.csv", index=False)

    n = len(imgs)
    print(f"=== 영상 단위 양성 수 (전체 {n}장)")
    rows = []
    for lab, name in LABELS.items():
        for t in THRESHOLDS:
            c = f"{lab}_t{t}"
            rows.append({"소견": name, "기준": f">={t}", "양성": int(imgs[c].sum()),
                         "유병률": f"{imgs[c].mean():.1%}",
                         "EGD만": int(imgs.loc[imgs.datasets == "egd", c].sum()),
                         "REFLACX": int(imgs.loc[imgs.datasets != "egd", c].sum()),
                         "AP 유병률": f"{imgs.loc[imgs.ViewPosition == 'AP', c].mean():.1%}",
                         "PA 유병률": f"{imgs.loc[imgs.ViewPosition == 'PA', c].mean():.1%}",
                         "test 예상(15%)": round(imgs[c].sum() * 0.15)})
    print(pd.DataFrame(rows).to_string(index=False))

    multi = imgs[imgs.n_readings == 5]
    print(f"\n=== 판독의 5명이 읽은 영상 {len(multi)}장: 판독의 간 일치 (기준 >=3)")
    for lab, name in LABELS.items():
        f = multi[f"{lab}_reader_frac"]
        print(f"  {name}: 5명 모두 일치 {((f == 0) | (f == 1)).mean():.0%}, "
              f"과반 양성 {int((f > 0.5).sum())}장, 의견 갈림(2:3 또는 3:2) {int(f.isin([0.4, 0.6]).sum())}장")

    both = imgs[["pleural_t3", "cardiac_t3"]].sum(axis=1)
    print(f"\n=== 두 소견 동시 양성 (기준 >=3): {int((both == 2).sum())}장, 둘 다 음성: {int((both == 0).sum())}장")
    print(f"\n저장: {PROC / 'cohort_images_labeled.csv'}")


if __name__ == "__main__":
    main()
