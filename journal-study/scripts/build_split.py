"""3단계: 환자 단위 train/val/test 분할을 만든다.

규칙
    - 같은 환자의 영상은 모두 같은 세트에 들어간다 (EGD-CXR와 REFLACX에 함께 있는 환자 포함)
    - 판독의 5명이 읽은 영상(109장)의 환자는 test에 고정한다
      → test 기준 라벨의 품질을 높이고, 판독의 간 불일치 분석을 test에서 할 수 있다
    - 나머지 환자는 (흉막 양성 여부, 심장비대 양성 여부, 데이터셋, AP 포함 여부)로 층화해 나눈다
    - 목표 비율: 환자 기준 70 / 15 / 15

입력  ~/X-ray/data/processed/cohort_images_labeled.csv
출력  ~/X-ray/data/processed/split_seed{SEED}.csv   (dicom_id, subject_id, split)

사용법  python build_split.py [--seed 42] [--no-multi-to-test]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROC = Path.home() / "X-ray/data/processed"
MIN_STRATUM = 10


def patient_table(img: pd.DataFrame) -> pd.DataFrame:
    return img.groupby("subject_id").agg(
        pleural=("pleural_t3", "max"), cardiac=("cardiac_t3", "max"),
        has_ap=("ViewPosition", lambda s: int((s == "AP").any())),
        datasets=("datasets", lambda s: "+".join(sorted(set("+".join(s).split("+"))))),
        multi=("n_readings", lambda s: int((s >= 5).any())),
        n_images=("dicom_id", "size"))


def strata(p: pd.DataFrame) -> pd.Series:
    """세밀한 층 → 인원이 적으면 (흉막, 심장비대) 층 → 그래도 적으면 하나로 합친다."""
    coarse = p["pleural"].astype(str) + p["cardiac"].astype(str)
    fine = coarse + "_" + p["datasets"] + "_ap" + p["has_ap"].astype(str)
    s = fine.where(fine.map(fine.value_counts()) >= MIN_STRATUM, coarse)
    return s.where(s.map(s.value_counts()) >= MIN_STRATUM, "rare")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-multi-to-test", action="store_true")
    args = ap.parse_args()

    img = pd.read_csv(PROC / "cohort_images_labeled.csv")
    pt = patient_table(img)
    n = len(pt)
    n_test, n_val = round(n * 0.15), round(n * 0.15)

    fixed = pt[pt["multi"] == 1].index if not args.no_multi_to_test else pt.index[:0]
    rest = pt.drop(index=fixed)
    rest_test = max(n_test - len(fixed), 0)

    train_val, test_extra = train_test_split(rest.index, test_size=rest_test, random_state=args.seed,
                                             stratify=strata(rest))
    tv = rest.loc[train_val]
    train, val = train_test_split(tv.index, test_size=n_val, random_state=args.seed, stratify=strata(tv))

    split_of = pd.Series("train", index=pt.index)
    split_of[val] = "val"
    split_of[list(fixed) + list(test_extra)] = "test"
    img["split"] = img["subject_id"].map(split_of)
    out = PROC / f"split_seed{args.seed}.csv"
    img[["dicom_id", "subject_id", "split"]].to_csv(out, index=False)

    # 검증
    assert img.groupby("subject_id")["split"].nunique().max() == 1, "한 환자가 여러 세트에 들어감"
    assert img["split"].notna().all()

    print(f"=== 분할 요약 (seed {args.seed}, 5명 판독 영상 test 고정: {not args.no_multi_to_test})")
    rows = []
    for s in ["train", "val", "test"]:
        d = img[img.split == s]
        rows.append({"세트": s, "환자": d.subject_id.nunique(), "영상": len(d),
                     "흉막 양성": int(d.pleural_t3.sum()), "흉막 %": f"{d.pleural_t3.mean():.1%}",
                     "심장비대 양성": int(d.cardiac_t3.sum()), "심장비대 %": f"{d.cardiac_t3.mean():.1%}",
                     "폐부종 양성": int(d.edema_t3.sum()),
                     "AP %": f"{(d.ViewPosition == 'AP').mean():.1%}",
                     "EGD 영상": int(d.datasets.str.contains('egd').sum()),
                     "5명 판독": int((d.n_readings >= 5).sum())})
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\n환자 겹침 검사 통과 (전체 환자 {n}명)")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
