"""11-2절 실행·요약: fs 모델(hybrid3, λ0.3) + 난이도 예측 헤드와 영상별 logit 축소.

조건 (난이도 예측의 정답)
    gaze           시선 난이도 d (제안)
    gaze-shuffled  다른 환자의 d (대조)
    certainty      판독의 확신도 난이도 u (비교)
    free           같은 구조, 난이도 지도 학습 없음 (구조 효과만, μ 없음)
강도 μ {0.1, 0.3, 1.0} × seed 0~2 → 조건마다 고름 → seed 0~9
선택 규칙 (sweep_smooth.py와 같음): val NLL(temperature 전) seed 평균 최저, 같으면 val AUROC(흉막·심장비대) 높은 쪽

사용법
    python sweep_diff.py run
    python sweep_diff.py report
    python sweep_diff.py final --choice gaze=0.3 gaze-shuffled=0.3 certainty=0.3 free=0
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
TARGETS = ["gaze", "gaze-shuffled", "certainty"]
MUS = [0.1, 0.3, 1.0]


def tag(t, mu):
    return f"{BASE}_diff-{t}{mu:g}"


def run_dir(t, mu, seed):
    return RUNS / f"findings_gaze-fs_t3_seed{seed}_{tag(t, mu)}"


def configs(args):
    if args.action == "final":
        return [(it.split("=")[0], float(it.split("=")[1])) for it in args.choice], args.seeds or list(range(10))
    return [(t, m) for t in TARGETS for m in args.mus] + [("free", 0.0)], args.seeds or [0, 1, 2]


def do_run(args):
    cfg, seeds = configs(args)
    jobs = [(c, s) for c in cfg for s in seeds]
    print(f"설정 {len(cfg)}개 × seed {len(seeds)}개 = {len(jobs)}회")
    for k, ((t, mu), s) in enumerate(jobs, 1):
        if (run_dir(t, mu, s) / "preds_test.csv").exists():
            print(f"[{k}/{len(jobs)}] 건너뜀 (이미 있음): {t} μ{mu:g} seed {s}")
            continue
        print(f"\n[{k}/{len(jobs)}] {t} μ{mu:g} seed {s}", flush=True)
        cmd = [sys.executable, str(TRAIN), "--task", "findings", "--gaze", "fs", "--seed", str(s),
               "--backbone", "imagenet", "--lr", "0.0001", "--schedule", "cosine", "--pool", "hybrid3",
               "--gaze-weight", "0.3", "--diff-target", t, "--diff-weight", str(mu), "--tag", tag(t, mu)] + args.extra.split()
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"  [오류] 실패 (returncode {r.returncode}). 다음으로 넘어갑니다.")


def row(d, name, mu, s):
    m = json.load(open(d / "val_metrics.json"))
    h = pd.read_csv(d / "history.csv")
    return {"target": name, "mu": mu, "seed": s, "val_nll": h.val_cls_loss.min(),
            "score": (m["pleural_auroc"] + m["cardiac_auroc"]) / 2, "pleural": m["pleural_auroc"],
            "cardiac": m["cardiac_auroc"], "edema": m["edema_auroc"],
            "ece_raw": np.mean([m[f"{f}_ece_raw"] for f in ["pleural", "cardiac", "edema"]]),
            "ece_ts": np.mean([m[f"{f}_ece_ts"] for f in ["pleural", "cardiac", "edema"]]),
            "best_epoch": m["best_epoch"],
            "train_diff": h.loc[h.epoch == m["best_epoch"], "train_diff"].item() if "train_diff" in h else np.nan}


def do_report(args):
    seeds = args.seeds or [0, 1, 2]
    rows = []
    for s in seeds:
        d = RUNS / f"findings_gaze-fs_t3_seed{s}_{BASE}"
        if (d / "val_metrics.json").exists():
            rows.append(row(d, "none (fs 그대로)", 0.0, s))
        d = RUNS / f"findings_gaze-fs_t3_seed{s}_{BASE}_sm-gaze0.05"
        if (d / "val_metrics.json").exists():
            rows.append(row(d, "참고: smoothing gaze 0.05", 0.0, s))
    for t, mu in [(t, m) for t in TARGETS for m in args.mus] + [("free", 0.0)]:
        for s in seeds:
            d = run_dir(t, mu, s)
            if (d / "val_metrics.json").exists():
                rows.append(row(d, t, mu, s))
    if not rows:
        print("결과가 없습니다.")
        return
    r = pd.DataFrame(rows)
    agg = r.groupby(["target", "mu"]).agg(n=("seed", "size"), val_nll=("val_nll", "mean"), nll_sd=("val_nll", "std"),
                                          score=("score", "mean"), pleural=("pleural", "mean"), cardiac=("cardiac", "mean"),
                                          edema=("edema", "mean"), ece_raw=("ece_raw", "mean"), ece_ts=("ece_ts", "mean"),
                                          best_epoch=("best_epoch", "mean"), train_diff=("train_diff", "mean")).reset_index()
    agg = agg.sort_values(["target", "val_nll", "score"], ascending=[True, True, False])
    pd.set_option("display.width", 220)
    print(f"=== 11-2절 val 결과 (seed {seeds} 평균). val_nll = temperature 전 val 분류 loss, score = 흉막·심장비대 AUROC 평균")
    print(agg.round(4).to_string(index=False))
    out = HOME / "results/sweep_diff.csv"
    agg.to_csv(out, index=False)
    print("\n=== 조건별 선택 (val_nll 최저, 같으면 score 높은 쪽)")
    for t in TARGETS + ["free"]:
        d = agg[agg.target == t]
        if len(d):
            b = d.iloc[0]
            print(f"  {t:14s} μ={b.mu:g} → val_nll {b.val_nll:.4f} ± {b.nll_sd:.4f}, score {b.score:.4f} (n={b.n})")
    print(f"\n저장: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["run", "report", "final"])
    ap.add_argument("--choice", nargs="*", default=[])
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--mus", type=float, nargs="*", default=MUS, help="μ 범위 (확장 시 run --mus 0.03 3.0)")
    ap.add_argument("--extra", default="")
    args = ap.parse_args()
    if args.action == "report":
        do_report(args)
    else:
        if args.action == "final" and not args.choice:
            raise SystemExit("final에는 --choice 정답=μ 가 필요합니다.")
        do_run(args)


if __name__ == "__main__":
    main()
