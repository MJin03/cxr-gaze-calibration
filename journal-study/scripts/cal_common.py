"""11절 평가 공용 설정: 조건, 비교, 외부 데이터 목록 (ANALYSIS_PLAN.md 11절, 예측 전에 고정).

조건 (seed 0~9)
    gaze           fs + 시선 난이도 smoothing ε_max 0.05 (제안)
    gaze_shuffled  fs + 다른 환자 시선 난이도 ε_max 0.1 (대조)
    certainty      fs + 판독의 확신도 난이도 ε_max 0.02 (비교)
    uniform        fs + 일반 smoothing ε_max 0.05 (비교)
    fs             smoothing 없는 fs (8절)
    c1_none        C1 none (avg, gaze 없음)
주 비교: gaze − uniform, gaze − gaze_shuffled / 보조: gaze − certainty, gaze − fs, gaze − c1_none

11-2절 A (난이도 예측 헤드 + 영상별 logit 축소, μ는 val로 선택: 2026-11-20 확정)
    a_gaze           정답 = 시선 난이도, μ 3 (제안)
    a_gaze_shuffled  정답 = 다른 환자 시선 난이도, μ 0.3 (대조)
    a_certainty      정답 = 판독의 확신도 난이도, μ 1 (비교)
    a_free           같은 구조, 난이도 정답 없음 (비교)
주 비교: a_gaze − a_free, a_gaze − a_gaze_shuffled / 보조: a_gaze − a_certainty, a_gaze − gaze, a_gaze − fs, a_gaze − c1_none
"""
import os
from pathlib import Path

HOME = Path.home() / "X-ray"
PROC, RUNS, RESULTS = HOME / "data/processed", HOME / "runs", HOME / "results"
HYB = "bb-imagenet_lr0.0001_sch-cosine_pool-hybrid3"
AVG = "bb-imagenet_lr0.0001_sch-cosine_pool-avg"
S10 = list(range(10))
FINDINGS_MODEL = ["pleural", "cardiac", "edema"]

# 조건 → (gaze, run 태그)
CONDITIONS = {
    "gaze":          ("fs", f"{HYB}_lam0.3_sm-gaze0.05"),
    "gaze_shuffled": ("fs", f"{HYB}_lam0.3_sm-gaze-shuffled0.1"),
    "certainty":     ("fs", f"{HYB}_lam0.3_sm-certainty0.02"),
    "uniform":       ("fs", f"{HYB}_lam0.3_sm-uniform0.05"),
    "fs":            ("fs", f"{HYB}_lam0.3"),
    "c1_none":       ("none", f"{AVG}_lam0"),
    "a_gaze":          ("fs", f"{HYB}_lam0.3_diff-gaze3"),
    "a_gaze_shuffled": ("fs", f"{HYB}_lam0.3_diff-gaze-shuffled0.3"),
    "a_certainty":     ("fs", f"{HYB}_lam0.3_diff-certainty1"),
    "a_free":          ("fs", f"{HYB}_lam0.3_diff-free0"),
}
COMPARISONS = [
    ("gaze", "uniform", "primary(11)"),
    ("gaze", "gaze_shuffled", "primary(11)"),
    ("gaze", "certainty", "secondary(11)"),
    ("gaze", "fs", "secondary(11)"),
    ("gaze", "c1_none", "secondary(11)"),
    ("a_gaze", "a_free", "primary(11-2)"),
    ("a_gaze", "a_gaze_shuffled", "primary(11-2)"),
    ("a_gaze", "a_certainty", "secondary(11-2)"),
    ("a_gaze", "gaze", "secondary(11-2)"),
    ("a_gaze", "fs", "secondary(11-2)"),
    ("a_gaze", "c1_none", "secondary(11-2)"),
]
# B (attention 엔트로피 불확실성) 대상: finding별 attention이 있는 조건
UNCERTAINTY_CONDS = ["a_gaze", "a_gaze_shuffled", "a_certainty", "a_free", "gaze", "fs"]
INTERNAL_VAL = "internal_val"  # B의 기준 분포(내부 val)용 예측 이름 (predict_external.py --dataset internal_val)

# 평가 데이터: 이름 → (영상 폴더, label 파일, finding 목록과 사용할 모델 출력)
DATASETS = {
    "vindr_test": dict(img=PROC / "vindr_test", labels=PROC / "vindr_test/labels.csv",
                       findings={"pleural": "pleural", "cardiac": "cardiac", "pleural_effusion": "pleural"}),
    "vindr_train": dict(img=PROC / "vindr_train3000", labels=PROC / "vindr_train3000/labels.csv",
                        findings={"pleural": "pleural", "cardiac": "cardiac", "pleural_effusion": "pleural"}),
    # CheXpert (CheXlocalize 묶음, 전문의 label, 정면 영상만). label 열 → 모델 출력
    "chexpert_test": dict(img=PROC / "chexpert_test", labels=PROC / "chexpert_test/labels.csv",
                          findings={"pleural": "pleural", "cardiac": "cardiac", "edema": "edema",
                                    "pleural_effusion": "pleural"}),
    "chexpert_valid": dict(img=PROC / "chexpert_valid", labels=PROC / "chexpert_valid/labels.csv",
                           findings={"pleural": "pleural", "cardiac": "cardiac", "edema": "edema",
                                     "pleural_effusion": "pleural"}),
}



def train_prevalence():
    """내부 train 양성 비율 (C의 EM 시작값). 모델 출력 이름 → 비율."""
    import pandas as pd
    lab = pd.read_csv(PROC / "cohort_images_labeled.csv")
    split = pd.read_csv(PROC / "split_seed42.csv")
    tr = lab[lab.dicom_id.isin(split.loc[split.split == "train", "dicom_id"])]
    return {f: float(tr[f"{f}_t3"].mean()) for f in FINDINGS_MODEL}


_TEST_SUFFIX = os.environ.get("VINDR_TEST_SUFFIX", "")  # 동작 확인용 (실제 실행에서는 비어 있음)


def run_dir(cond: str, seed: int) -> Path:
    gaze, tag = CONDITIONS[cond]
    return RUNS / f"findings_gaze-{gaze}_t3_seed{seed}_{tag}{_TEST_SUFFIX}"
