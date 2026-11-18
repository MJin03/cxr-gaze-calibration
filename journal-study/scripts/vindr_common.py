"""D단계(VinDr-CXR 외부 검증) 공용 설정. ANALYSIS_PLAN.md 9절과 같은 내용을 코드로 고정한다.

여기 적힌 조건·비교·지표를 결과를 보기 전에 정하고, 결과를 본 뒤에는 바꾸지 않는다.
"""
import os
from pathlib import Path

HOME = Path.home() / "X-ray"
PROC, RUNS, RESULTS = HOME / "data/processed", HOME / "runs", HOME / "results"
VINDR = HOME / "data/raw/vindr-cxr"
VINDR_DIR = PROC / "vindr_test"          # 전처리 결과 (영상, label, 박스, 전처리 선택)
FINDINGS_MODEL = ["pleural", "cardiac", "edema"]   # 모델 출력 순서
FINDINGS_EVAL = ["pleural", "cardiac"]             # VinDr 평가 대상 (폐부종은 양성 0장이라 평가하지 않음)
SENSITIVITY = {"pleural_effusion": "pleural"}      # 민감도 분석: label → 사용하는 모델 출력

HYB = "bb-imagenet_lr0.0001_sch-cosine_pool-hybrid3"
AVG = "bb-imagenet_lr0.0001_sch-cosine_pool-avg"
S10, S5 = list(range(10)), list(range(5))

# 조건 이름 → (gaze, run 태그, seed 목록). run 폴더 = findings_gaze-{gaze}_t3_seed{s}[_{태그}]
CONDITIONS = {
    "fs":          ("fs",          f"{HYB}_lam0.3", S10),
    "c1_none":     ("none",        f"{AVG}_lam0",   S10),
    "fs_shuffled": ("fs-shuffled", f"{HYB}_lam0.3", S10),
    "real_global": ("real",        f"{HYB}_lam1",   S10),
    "hyb_none":    ("none",        f"{HYB}_lam0",   S10),
    "ellipse":     ("ellipse",     f"{HYB}_lam0.1", S10),
    "pre_none":     ("none",     "", S5),   # 1~7절 사전 고정 설정
    "pre_real":     ("real",     "", S5),
    "pre_shuffled": ("shuffled", "", S5),
}
HYBRID3 = ["fs", "fs_shuffled", "real_global", "hyb_none", "ellipse"]   # attention 지도가 있는 조건 (pointing game)

# (비교 대상, 기준, 종류). 종류: primary / secondary / noninferiority
COMPARISONS = [
    ("fs", "c1_none", "primary"),
    ("fs", "fs_shuffled", "primary"),
    ("fs", "real_global", "secondary"),
    ("fs", "hyb_none", "secondary"),
    ("fs", "ellipse", "noninferiority"),
    ("pre_real", "pre_none", "secondary"),
    ("pre_real", "pre_shuffled", "secondary"),
]
NI_MARGIN = 0.02            # 비열등 마진: AUROC 차이 CI 하한 > −0.02
PRIMARY_METRIC = "missed_pos_rate@25"
KEY_METRICS = ["missed_pos_rate@25", "auroc"]
RATES = [0.10, 0.25, 0.30, 0.50]
SENS_TARGETS = [0.95, 0.98]


_TEST_SUFFIX = os.environ.get("VINDR_TEST_SUFFIX", "")  # 동작 확인용 (실제 실행에서는 비어 있음)


def run_dir(cond: str, seed: int) -> Path:
    gaze, tag, _ = CONDITIONS[cond]
    return RUNS / (f"findings_gaze-{gaze}_t3_seed{seed}" + (f"_{tag}{_TEST_SUFFIX}" if tag else ""))
