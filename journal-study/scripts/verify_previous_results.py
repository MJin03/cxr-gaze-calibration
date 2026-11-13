"""이전 연구(IWAIT 2027) 보고 수치를 원본 예측값으로 다시 계산해 대조한다.

학습이나 추론은 하지 않는다. 보관본의 예측 CSV와 temperature 값만 읽는다.

확인 항목
1. 조건별 raw 지표(Accuracy, Macro F1, Macro AUROC, ECE, Brier, NLL)가 보고값과 같은지
2. 보고된 temperature로 보정한 지표가 보고값과 같은지
3. 운영점 threshold가 두 계산 방식(quantile / 순위) 중 어느 것과 일치하는지
4. 예측 파일 간 test 사례 구성, prediction 열과 argmax 일치 여부

사용법
    python verify_previous_results.py [results 폴더]
기본 results 폴더:
    ~/X-ray/archive/kim_mj_sens_X-ray/gaze_calibration_research/results
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

DEFAULT_RESULTS = Path.home() / "X-ray/archive/kim_mj_sens_X-ray/gaze_calibration_research/results"
PROB_COLS = ["prob_Normal", "prob_CHF", "prob_Pneumonia"]
N_BINS = 15
EPS = 1e-12

# 예측 파일 접두어 -> 보고 표의 조건 이름
ARMS = {
    "baseline": "Baseline",
    "gaze_original": "Gaze-Original",
    "shuffled_gaze": "Shuffled-Gaze",
    "random_gaze": "Random-Gaze",
    "lesion_supervised": "Lesion-Supervised",
}

# previous-study/results/reported 의 보고값
REPORTED_RAW = {
    "Baseline": dict(Accuracy=0.670732, Macro_F1=0.656323, Macro_AUROC=0.867083, ECE=0.055293, Brier=0.404887, NLL=0.668854),
    "Gaze-Original": dict(Accuracy=0.652439, Macro_F1=0.656791, Macro_AUROC=0.84625, ECE=0.059363, Brier=0.448311, NLL=0.717421),
    "Shuffled-Gaze": dict(Accuracy=0.689024, Macro_F1=0.673437, Macro_AUROC=0.825912, ECE=0.164286, Brier=0.485149, NLL=0.848986),
    "Random-Gaze": dict(Accuracy=0.658537, Macro_F1=0.638235, Macro_AUROC=0.823529, ECE=0.08781, Brier=0.454969, NLL=0.72969),
    "Lesion-Supervised": dict(Accuracy=0.640244, Macro_F1=0.588846, Macro_AUROC=0.82297, ECE=0.125479, Brier=0.47477, NLL=0.766754),
}
REPORTED_TS = {
    "Baseline": dict(Macro_AUROC=0.867222, ECE=0.06996, Brier=0.404856, NLL=0.66341),
    "Gaze-Original": dict(Macro_AUROC=0.845816, ECE=0.06728, Brier=0.447637, NLL=0.719606),
    "Shuffled-Gaze": dict(Macro_AUROC=0.825457, ECE=0.089979, Brier=0.44848, NLL=0.73356),
    "Random-Gaze": dict(Macro_AUROC=0.823691, ECE=0.08931, Brier=0.451085, NLL=0.726215),
    "Lesion-Supervised": dict(Macro_AUROC=0.821174, ECE=0.0923, Brier=0.461256, NLL=0.747687),
}
TOL = 5e-4  # 보고값 반올림을 고려한 허용 오차


def load_preds(path: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = pd.read_csv(path)
    probs = df[PROB_COLS].to_numpy(dtype=float)
    labels = df["label"].to_numpy(dtype=int)
    return probs, labels, df


def apply_temperature(probs: np.ndarray, t: float) -> np.ndarray:
    # softmax(z / T) 는 softmax(log p / T) 와 같다 (상수항은 소거됨)
    logits = np.log(np.clip(probs, EPS, 1.0)) / t
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def ece_score(probs: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS) -> float:
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    onehot = np.eye(probs.shape[1])[labels]
    pred = probs.argmax(axis=1)
    return dict(
        Accuracy=float((pred == labels).mean()),
        Macro_F1=float(f1_score(labels, pred, average="macro")),
        Macro_AUROC=float(roc_auc_score(labels, probs, multi_class="ovr", average="macro")),
        ECE=ece_score(probs, labels),
        Brier=float(((probs - onehot) ** 2).sum(axis=1).mean()),
        NLL=float(-np.log(np.clip(probs[np.arange(len(labels)), labels], EPS, 1.0)).mean()),
        Errors=int((pred != labels).sum()),
        N=int(len(labels)),
    )


def threshold_quantile(conf: np.ndarray, coverage: float) -> float:
    return -np.inf if coverage >= 1.0 else float(np.quantile(conf, 1.0 - coverage))


def threshold_rank(conf: np.ndarray, coverage: float) -> float:
    if coverage >= 1.0:
        return -np.inf
    ordered = np.sort(conf)[::-1]
    idx = max(math.ceil(len(conf) * coverage) - 1, 0)
    return float(ordered[idx])


def compare(computed: dict, reported: dict) -> pd.DataFrame:
    rows = []
    for key, rep in reported.items():
        val = computed[key]
        rows.append(dict(metric=key, computed=round(val, 6), reported=rep,
                         diff=round(val - rep, 6), match=abs(val - rep) <= TOL))
    return pd.DataFrame(rows)


def main() -> None:
    results = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DEFAULT_RESULTS
    test_dir, val_dir = results / "test_predictions", results / "validation_predictions"
    temps = pd.read_csv(results / "decision_support_temperatures.csv").set_index("Arm")["Temperature"]
    pd.set_option("display.width", 160)

    all_ok = True
    test_ids = {}
    print(f"results 폴더: {results}\n")

    for prefix, arm in ARMS.items():
        probs, labels, df = load_preds(test_dir / f"{prefix}_test_predictions.csv")
        test_ids[arm] = set(df["dicom_id"])
        argmax_ok = bool((df["prediction"].to_numpy() == probs.argmax(axis=1)).all())
        raw = metrics(probs, labels)
        ts = metrics(apply_temperature(probs, float(temps[arm])), labels)

        print(f"=== {arm}  (n={raw['N']}, 오류 {raw['Errors']}건, T={temps[arm]:.4f}, prediction=argmax: {argmax_ok})")
        raw_cmp, ts_cmp = compare(raw, REPORTED_RAW[arm]), compare(ts, REPORTED_TS[arm])
        print("[raw]");  print(raw_cmp.to_string(index=False))
        print("[temperature scaled]");  print(ts_cmp.to_string(index=False))
        print()
        all_ok &= bool(raw_cmp["match"].all() and ts_cmp["match"].all() and argmax_ok)

    base_ids = test_ids["Baseline"]
    same = {arm: ids == base_ids for arm, ids in test_ids.items()}
    print("test 사례 구성이 Baseline과 같은지:", same, "\n")

    # 운영점 threshold 계산 방식 확인
    ops = pd.read_csv(results / "decision_support_operating_points.csv")
    print("=== 운영점 threshold: 보고값 vs 두 계산 방식 (validation, temperature 적용 확률)")
    rows = []
    for prefix, arm in ARMS.items():
        vpath = val_dir / f"{prefix}_validation_predictions.csv"
        if not vpath.exists() or arm not in set(ops["Arm"]):
            continue
        vprobs, _, _ = load_preds(vpath)
        vconf = apply_temperature(vprobs, float(temps[arm])).max(axis=1)
        for _, r in ops[ops["Arm"] == arm].iterrows():
            cov = float(r["Target_Coverage"])
            rep = float(r["Confidence_Threshold"])
            q, k = threshold_quantile(vconf, cov), threshold_rank(vconf, cov)
            rows.append(dict(Arm=arm, Target=cov, reported=rep, quantile=q, rank=k,
                             match_quantile=bool(np.isclose(rep, q, atol=1e-6) or (np.isinf(rep) and np.isinf(q))),
                             match_rank=bool(np.isclose(rep, k, atol=1e-6) or (np.isinf(rep) and np.isinf(k))),
                             Errors_Total=int(r["Errors_Total"])))
    th = pd.DataFrame(rows)
    print(th.to_string(index=False))
    print()

    # 운영점 표의 전체 오류 수가 예측 파일과 맞는지
    err_from_preds = {arm: metrics(*load_preds(test_dir / f"{p}_test_predictions.csv")[:2])["Errors"] for p, arm in ARMS.items()}
    err_from_ops = ops.groupby("Arm")["Errors_Total"].first().to_dict()
    print("전체 오류 수 (예측 파일 vs 운영점 표):")
    for arm in ARMS.values():
        if arm in err_from_ops:
            flag = "OK" if err_from_ops[arm] == err_from_preds[arm] else "불일치"
            print(f"  {arm}: {err_from_preds[arm]} vs {err_from_ops[arm]}  {flag}")
    print()
    print("요약: 분류·보정 지표 전부 일치" if all_ok else "요약: 일부 지표 불일치 — 위 표의 match=False 행 확인")


if __name__ == "__main__":
    main()
