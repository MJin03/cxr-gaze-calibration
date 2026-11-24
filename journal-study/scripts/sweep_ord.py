"""16절 학습: 순서형 확신도 출력 (≥3 기존 + ≥4·≥5). 기존 fs·c1_none과 같은 설정에 --ordinal만 추가.

조건 (cal_common.ORD_CONDITIONS)
    ord_fs  hybrid3, fs 시선 λ0.3, --ordinal
    ord_c1  avg, 시선 없음 (λ0), --ordinal
seed 0~9, ordinal 가중치 1 (사전 고정, 조정 없음). 끊긴 run은 처음부터 다시.

사용법
    python sweep_ord.py run [--seeds 0 1 2] [--conds ord_fs]
    python sweep_ord.py report        # val 지표: ≥3 AUROC (기존 fs·c1_none과 비교), 순서형 temperature
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cal_common import ORD_CONDITIONS, RUNS, run_dir

TRAIN = Path(__file__).parent / "train.py"
ARGS = {
    "ord_fs": ["--gaze", "fs", "--pool", "hybrid3", "--gaze-weight", "0.3"],
    "ord_c1": ["--gaze", "none", "--pool", "avg", "--gaze-weight", "0"],
}
BASE_OF = {"ord_fs": "fs", "ord_c1": "c1_none"}


def do_run(args):
    jobs = [(c, s) for c in args.conds for s in args.seeds]
    print(f"순서형 학습 {len(jobs)}회: {args.conds} × seed {args.seeds}", flush=True)
    for k, (c, s) in enumerate(jobs, 1):
        d = run_dir(c, s)
        if (d / "preds_test.csv").exists():
            print(f"[{k}/{len(jobs)}] 건너뜀 (이미 있음): {c} seed {s}", flush=True)
            continue
        print(f"\n[{k}/{len(jobs)}] {c} seed {s} 시작", flush=True)
        tag = ORD_CONDITIONS[c][1]
        cmd = [sys.executable, str(TRAIN), "--task", "findings", "--seed", str(s), "--backbone", "imagenet",
               "--lr", "0.0001", "--schedule", "cosine", "--ordinal", "--ordinal-weight", "1",
               "--tag", tag] + ARGS[c] + args.extra.split()
        if (d / "best.pt").exists():
            print("  끊긴 run 발견 → 처음부터 다시 학습 (--overwrite)", flush=True)
            cmd.append("--overwrite")
        r = subprocess.run(cmd)
        print(f"[{k}/{len(jobs)}] {c} seed {s} " + ("완료" if r.returncode == 0 else f"실패 (returncode {r.returncode})"), flush=True)


def do_report(args):
    rows = []
    for c in args.conds:
        for cond, lab in [(c, "순서형"), (BASE_OF[c], "기존")]:
            for s in args.seeds:
                d = run_dir(cond, s)
                if not (d / "val_metrics.json").exists():
                    continue
                m = json.load(open(d / "val_metrics.json"))
                t = json.load(open(d / "temperature.json"))
                rows.append({"조건": c, "모델": lab, "seed": s,
                             **{f"AUROC_{f}": m[f"{f}_auroc"] for f in ["pleural", "cardiac", "edema"]},
                             **{f"T_{f}": t[f] for f in ["pleural", "cardiac", "edema"]},
                             **{f"T_{k}": v for k, v in t.items() if "_t" in k}})
    if not rows:
        print("결과가 없습니다.")
        return
    r = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    g = r.groupby(["조건", "모델"]).agg(n=("seed", "size"), **{c: (c, "mean") for c in r.columns if c.startswith(("AUROC", "T_"))})
    print("=== 16절 val 결과 (seed 평균). ≥3 AUROC는 기존 모델과 비교 (판별력 유지 확인용)")
    print(g.round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["run", "report"])
    ap.add_argument("--conds", nargs="*", default=list(ORD_CONDITIONS))
    ap.add_argument("--seeds", type=int, nargs="*", default=list(range(10)))
    ap.add_argument("--extra", default="")
    args = ap.parse_args()
    do_run(args) if args.action == "run" else do_report(args)


if __name__ == "__main__":
    main()
