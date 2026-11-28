"""19절 학습: 강한 모델 (torchxrayvision densenet121-res224-nih, NIH ChestX-ray14로만 사전학습) 위에서 시선 효과 반복.

조건 (기존 조건과 같은 구조, backbone만 NIH 사전학습)
    nih_c1          avg, 시선 없음 (λ0)           ← c1_none
    nih_hyb_none    hybrid3, 시선 없음 (λ0)       ← hyb_none
    nih_fs          hybrid3, 소견별 시선 fs       ← fs
    nih_fs_shuffled hybrid3, 다른 환자의 fs 시선  ← fs_shuffled (nih_fs와 같은 lr·λ, 따로 조정 안 함)
설정 탐색 (내부 val만, seed 0~2): lr ∈ {1e-4, 3e-5} (세 조건 모두), λ ∈ {0.1, 0.3, 1.0} (nih_fs만)
선택 규칙 (기존과 같음): val NLL(temperature 전 val 분류 loss) seed 평균 최저, 같으면 val AUROC(흉막·심장비대 평균) 높은 쪽
→ results/nih_choice.json 기록 → 선택된 설정으로 seed 0~9 (탐색 때 만든 seed 0~2 재사용)

23절 (2026-11-28): --backbone convnext_t 이면 같은 절차를 ConvNeXt-Tiny(ImageNet-1K)로 반복.
    조건 이름 cnx_c1 / cnx_hyb_none / cnx_fs / cnx_fs_shuffled, 선택 기록 results/cnx_choice.json

사용법  python sweep_nih.py all [--backbone nih|convnext_t]   # 탐색 → 선택 → 최종 (끊겨도 다시 실행하면 이어서)
        python sweep_nih.py report [--backbone ...]          # 탐색 결과 표만
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

HOME = Path.home() / "X-ray"
RUNS, RESULTS = HOME / "runs", HOME / "results"
TRAIN = Path(__file__).parent / "train.py"
LRS = [1e-4, 3e-5]
LAMS = [0.1, 0.3, 1.0]
TUNE_SEEDS, FINAL_SEEDS = [0, 1, 2], list(range(10))
BB = {"nih": ("nih", "nih_choice.json"), "convnext_t": ("cnx", "cnx_choice.json")}  # backbone → (조건 접두어, 선택 기록)
BACKBONE = ["nih"]
SPEC = {}  # 조건 → (gaze, pool, 탐색할 λ 목록). main()에서 접두어로 채움


def set_backbone(bb):
    BACKBONE[0] = bb
    p = BB[bb][0]
    SPEC.clear()
    SPEC.update({f"{p}_c1": ("none", "avg", [0.0]), f"{p}_hyb_none": ("none", "hybrid3", [0.0]),
                 f"{p}_fs": ("fs", "hybrid3", LAMS)})


def tag(lr, pool, lam):
    return f"bb-{BACKBONE[0]}_lr{lr:g}_sch-cosine_pool-{pool}_lam{lam:g}"


def rdir(gaze, lr, pool, lam, s):
    return RUNS / f"findings_gaze-{gaze}_t3_seed{s}_{tag(lr, pool, lam)}"


EXTRA = []


def train(gaze, lr, pool, lam, s, label):
    d = rdir(gaze, lr, pool, lam, s)
    if (d / "preds_test.csv").exists():
        print(f"  건너뜀 (이미 있음): {label}", flush=True)
        return
    print(f"\n[{label}] 시작", flush=True)
    cmd = [sys.executable, str(TRAIN), "--task", "findings", "--gaze", gaze, "--seed", str(s), "--backbone", BACKBONE[0],
           "--lr", str(lr), "--schedule", "cosine", "--pool", pool, "--gaze-weight", str(lam), "--tag", tag(lr, pool, lam)] + EXTRA
    if (d / "best.pt").exists():
        cmd.append("--overwrite")
    r = subprocess.run(cmd)
    print(f"[{label}] " + ("완료" if r.returncode == 0 else f"실패 (returncode {r.returncode})"), flush=True)


def tune_rows():
    rows = []
    for c, (g, pool, lams) in SPEC.items():
        for lr in LRS:
            for lam in lams:
                for s in TUNE_SEEDS:
                    d = rdir(g, lr, pool, lam, s)
                    if (d / "val_metrics.json").exists():
                        m = json.load(open(d / "val_metrics.json"))
                        h = pd.read_csv(d / "history.csv")
                        rows.append({"cond": c, "lr": lr, "lam": lam, "seed": s, "val_nll": h.val_cls_loss.min(),
                                     "score": (m["pleural_auroc"] + m["cardiac_auroc"]) / 2, "edema": m["edema_auroc"]})
    return pd.DataFrame(rows)


def choose(r):
    agg = r.groupby(["cond", "lr", "lam"]).agg(n=("seed", "size"), val_nll=("val_nll", "mean"), score=("score", "mean"),
                                               edema=("edema", "mean")).reset_index()
    agg = agg.sort_values(["cond", "val_nll", "score"], ascending=[True, True, False])
    best = {c: agg[agg.cond == c].iloc[0] for c in SPEC}
    return agg, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["all", "report"])
    ap.add_argument("--extra", default="", help="동작 확인용 train.py 추가 인자")
    ap.add_argument("--backbone", choices=list(BB), default="nih")
    args = ap.parse_args()
    set_backbone(args.backbone)
    P = BB[args.backbone][0]
    EXTRA.extend(args.extra.split())
    n_tune = sum(len(LRS) * len(l) * len(TUNE_SEEDS) for _, _, l in SPEC.values())
    if args.action == "all":
        print(f"=== 1/2 설정 탐색 {n_tune}회 (내부 val만)", flush=True)
        k = 0
        for c, (g, pool, lams) in SPEC.items():
            for lr in LRS:
                for lam in lams:
                    for s in TUNE_SEEDS:
                        k += 1
                        train(g, lr, pool, lam, s, f"탐색 {k}/{n_tune} {c} lr{lr:g} λ{lam:g} seed {s}")
    r = tune_rows()
    if r.empty:
        raise SystemExit("탐색 결과가 없습니다.")
    agg, best = choose(r)
    pd.set_option("display.width", 200)
    print("\n=== 설정 탐색 결과 (seed 0~2 평균). val_nll 낮을수록 좋음, score = val AUROC(흉막·심장비대 평균)")
    print(agg.round(4).to_string(index=False))
    if (agg.n < len(TUNE_SEEDS)).any():
        print("[주의] seed가 덜 끝난 설정이 있음 → all을 다시 실행")
        if args.action == "report":
            return
    cond_map = {}
    for c, (g, pool, _) in SPEC.items():
        b = best[c]
        cond_map[c] = [g, tag(b.lr, pool, b.lam)]
        print(f"  선택 {c:14s} lr {b.lr:g}, λ {b.lam:g} → val_nll {b.val_nll:.4f}, score {b.score:.4f}")
    bf = best[f"{P}_fs"]
    cond_map[f"{P}_fs_shuffled"] = ["fs-shuffled", tag(bf.lr, "hybrid3", bf.lam)]
    print(f"  {P}_fs_shuffled: {P}_fs와 같은 lr {bf.lr:g}, λ {bf.lam:g}")
    if args.action == "report":
        return
    json.dump({"rule": "val NLL seed 0~2 평균 최저, 같으면 score", "conditions": cond_map,
               "choice": {c: {"lr": float(best[c].lr), "lam": float(best[c].lam)} for c in SPEC}},
              open(RESULTS / BB[args.backbone][1], "w"), indent=2, ensure_ascii=False)
    print(f"  기록: {RESULTS / BB[args.backbone][1]}")

    print("\n=== 2/2 최종 학습 (seed 0~9, 탐색 때 만든 seed 0~2는 재사용)", flush=True)
    finals = [(c, s) for c in [f"{P}_c1", f"{P}_hyb_none", f"{P}_fs", f"{P}_fs_shuffled"] for s in FINAL_SEEDS]
    for k, (c, s) in enumerate(finals, 1):
        g, t = cond_map[c]
        pool = "avg" if "pool-avg" in t else "hybrid3"
        lr = float(t.split("_lr")[1].split("_")[0])
        lam = float(t.split("_lam")[1])
        train(g, lr, pool, lam, s, f"최종 {k}/{len(finals)} {c} seed {s}")
    print(f"\n최종 학습 끝. 다음: 외부 예측 (predict_external.py --conds {P}_c1 {P}_hyb_none {P}_fs {P}_fs_shuffled)")


if __name__ == "__main__":
    main()
