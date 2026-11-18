"""11절 실행·요약: fs 모델(hybrid3, λ0.3) 위에 난이도 가중 label smoothing을 얹는다.

조건 (smoothing 종류)
    gaze           시선 난이도 (제안 방법, A)
    gaze-shuffled  다른 환자의 시선 난이도 (대조, B)
    certainty      판독의 확신도 난이도 (비교, C)
    uniform        모든 영상 같은 강도 (비교, D)
강도 ε_max {0.05, 0.1, 0.2} × seed 0~2 → 조건마다 고름 → seed 0~9
선택 규칙 (모든 조건 동일, val만 사용): val NLL(= temperature 전 val 분류 loss, 정답 label 그대로)의 seed 평균이 가장 낮은 ε_max.
    같으면 val AUROC(흉막·심장비대 평균)가 높은 쪽

사용법
    python sweep_smooth.py run
    python sweep_smooth.py report
    python sweep_smooth.py final --choice gaze=0.1 gaze-shuffled=0.1 certainty=0.2 uniform=0.05
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HOME = Path.home() / "X-ray"
RUNS = HOME / "runs"
TRAIN = Path(__file__).parent / "train.py"
BASE = "bb-imagenet_lr0.0001_sch-cosine_pool-hybrid3_lam0.3"
SMOOTHS = ["gaze", "gaze-shuffled", "certainty", "uniform"]
EPS = [0.05, 0.1, 0.2]


def tag(sm, eps):
    return f"{BASE}_sm-{sm}{eps:g}"


def run_dir(sm, eps, seed):
    return RUNS / f"findings_gaze-fs_t3_seed{seed}_{tag(sm, eps)}"


def jobs_for(args):
    if args.action == "final":
        cfg = [(it.split("=")[0], float(it.split("=")[1])) for it in args.choice]
        return cfg, args.seeds or list(range(10))
    return [(sm, e) for sm in SMOOTHS for e in EPS], args.seeds or [0, 1, 2]


def do_run(args):
    cfg, seeds = jobs_for(args)
    jobs = [(c, s) for c in cfg for s in seeds]
    print(f"설정 {len(cfg)}개 × seed {len(seeds)}개 = {len(jobs)}회")
    for k, ((sm, e), s) in enumerate(jobs, 1):
        if (run_dir(sm, e, s) / "preds_test.csv").exists():
            print(f"[{k}/{len(jobs)}] 건너뜀 (이미 있음): {sm} ε{e:g} seed {s}")
            continue
        print(f"\n[{k}/{len(jobs)}] {sm} ε{e:g} seed {s}", flush=True)
        cmd = [sys.executable, str(TRAIN), "--task", "findings", "--gaze", "fs", "--seed", str(s),
               "--backbone", "imagenet", "--lr", "0.0001", "--schedule", "cosine", "--pool", "hybrid3",
               "--gaze-weight", "0.3", "--smooth", sm, "--smooth-max", str(e), "--tag", tag(sm, e)] + args.extra.split()
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"  [오류] 실패 (returncode {r.returncode}). 다음으로 넘어갑니다.")


def row(d, name, eps, s):
    m = json.load(open(d / "val_metrics.json"))
    h = pd.read_csv(d / "history.csv")
    return {"smooth": name, "eps": eps, "seed": s, "val_nll": h.val_cls_loss.min(),
            "score": (m["pleural_auroc"] + m["cardiac_auroc"]) / 2, "pleural": m["pleural_auroc"],
            "cardiac": m["cardiac_auroc"], "edema": m["edema_auroc"],
            "ece_raw": np.mean([m[f"{f}_ece_raw"] for f in ["pleural", "cardiac", "edema"]]),
            "ece_ts": np.mean([m[f"{f}_ece_ts"] for f in ["pleural", "cardiac", "edema"]]),
            "best_epoch": m["best_epoch"]}


def do_report(args):
    seeds = args.seeds or [0, 1, 2]
    rows = []
    for s in seeds:  # 기준: smoothing 없는 fs
        d = RUNS / f"findings_gaze-fs_t3_seed{s}_{BASE}"
        if (d / "val_metrics.json").exists():
            rows.append(row(d, "none (fs 그대로)", 0.0, s))
    for sm in SMOOTHS:
        for e in EPS:
            for s in seeds:
                d = run_dir(sm, e, s)
                if (d / "val_metrics.json").exists():
                    rows.append(row(d, sm, e, s))
    if not rows:
        print("결과가 없습니다.")
        return
    r = pd.DataFrame(rows)
    agg = r.groupby(["smooth", "eps"]).agg(n=("seed", "size"), val_nll=("val_nll", "mean"), nll_sd=("val_nll", "std"),
                                           score=("score", "mean"), pleural=("pleural", "mean"), cardiac=("cardiac", "mean"),
                                           edema=("edema", "mean"), ece_raw=("ece_raw", "mean"), ece_ts=("ece_ts", "mean"),
                                           best_epoch=("best_epoch", "mean")).reset_index()
    agg = agg.sort_values(["smooth", "val_nll", "score"], ascending=[True, True, False])
    pd.set_option("display.width", 220)
    print(f"=== 11절 val 결과 (seed {seeds} 평균). val_nll = temperature 전 val 분류 loss (낮을수록 좋음), score = 흉막·심장비대 AUROC 평균")
    print(agg.round(4).to_string(index=False))
    out = HOME / "results/sweep_smooth.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out, index=False)
    print("\n=== 조건별 선택 (val_nll 최저, 같으면 score 높은 쪽)")
    for sm in SMOOTHS:
        d = agg[agg.smooth == sm]
        if len(d):
            b = d.iloc[0]
            print(f"  {sm:14s} ε_max={b.eps:g} → val_nll {b.val_nll:.4f} ± {b.nll_sd:.4f}, score {b.score:.4f} (n={b.n})")
    print(f"\n저장: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["run", "report", "final"])
    ap.add_argument("--choice", nargs="*", default=[], help="final: 종류=ε_max")
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--extra", default="", help='train.py에 그대로 넘길 인자. 예: "--max-images 64 --epochs 1"')
    args = ap.parse_args()
    if args.action == "report":
        do_report(args)
    else:
        if args.action == "final" and not args.choice:
            raise SystemExit("final에는 --choice 종류=ε_max 가 필요합니다.")
        do_run(args)


if __name__ == "__main__":
    main()
