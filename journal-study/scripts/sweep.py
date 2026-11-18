"""개선 단계 실행·요약 도구 (ANALYSIS_PLAN.md 8절).

단계
    A  공통 학습 설정: Baseline(gaze none, avg pool)으로 backbone × 학습률을 고른다 (입력 512)
    A2 입력 크기 추가 확인: CheXpert 가중치는 224로 사전학습 → 224 입력과 더 높은 학습률 확인
    B  gaze 설정: A에서 고른 backbone·학습률로
         none     : pool {avg, attn}
         real     : pool {avg, attn} × λ {0.03, 0.1, 0.3, 1.0}
         shuffled : real과 같은 범위
    C  최종: B에서 조건별로 고른 설정을 seed 0~9로
    C3 finding별 시선 (2026-11-18 추가): pool hybrid3로
         none        : λ 0
         real        : 전체 시선 지도를 세 attention에 동일 적용 (real-global), λ {0.1, 0.3, 1.0}
         fs          : finding별 시선 지도 (--window 초, 기본 3), λ {0.1, 0.3, 1.0}
         fs-shuffled : 다른 환자의 finding별 시선 지도, λ {0.1, 0.3, 1.0}
         ellipse     : 판독의가 그린 병변 타원 (참고 비교), λ {0.1, 0.3, 1.0}
       고른 뒤 seed 0~9는 --stage C --choice none=hybrid3,0 fs=hybrid3,λ ... 로 실행

선택 기준 (모든 조건 동일): val AUROC(흉막 + 심장비대 평균)의 seed 평균이 가장 높은 설정.
같으면 val 분류 loss가 낮은 쪽.

사용법
    python sweep.py run    --stage A
    python sweep.py report --stage A
    python sweep.py run    --stage B --backbone chexpert --lr 3e-5
    python sweep.py report --stage B --backbone chexpert --lr 3e-5
    python sweep.py run    --stage C --backbone chexpert --lr 3e-5 --choice none=avg,0 real=attn,0.3 shuffled=attn,0.3
    python sweep.py run    --stage C3 --backbone imagenet --lr 1e-4
    python sweep.py report --stage C3 --backbone imagenet --lr 1e-4
이미 결과가 있는 run은 건너뛴다.
"""
import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HOME = Path.home() / "X-ray"
RUNS = HOME / "runs"
TRAIN = Path(__file__).parent / "train.py"
SCHEDULE = "cosine"


FS_GAZE = ("fs", "fs-shuffled")


def tag(bb, lr, pool, lam, size=512, gaze="", window=3.0):
    t = f"bb-{bb}_lr{lr:g}_sch-{SCHEDULE}_pool-{pool}_lam{lam:g}"
    t += f"_sz{size}" if size != 512 else ""
    return t + (f"_w{window:g}" if gaze in FS_GAZE and window != 3.0 else "")


def configs(args):
    """(gaze, backbone, lr, pool, lam) 목록과 seed 목록."""
    if args.stage == "A":
        cfg = [("none", bb, lr, "avg", 0.0, 512) for bb in ["imagenet", "chexpert"] for lr in [1e-4, 3e-5]]
        return cfg, args.seeds or [0, 1, 2]
    if args.stage == "A2":
        cfg = [("none", "imagenet", 1e-4, "avg", 0.0, 224), ("none", "chexpert", 1e-4, "avg", 0.0, 224),
               ("none", "chexpert", 3e-4, "avg", 0.0, 224), ("none", "chexpert", 3e-4, "avg", 0.0, 512)]
        return cfg, args.seeds or [0, 1, 2]
    if args.stage == "B":
        cfg = [("none", args.backbone, args.lr, p, 0.0, args.size) for p in ["avg", "attn"]]
        cfg += [(g, args.backbone, args.lr, p, lam, args.size) for g in ["real", "shuffled"]
                for p in ["avg", "attn"] for lam in [0.03, 0.1, 0.3, 1.0]]
        return cfg, args.seeds or [0, 1, 2]
    if args.stage == "C3":
        cfg = [("none", args.backbone, args.lr, "hybrid3", 0.0, args.size)]
        cfg += [(g, args.backbone, args.lr, "hybrid3", lam, args.size)
                for g in ["real", "fs", "fs-shuffled", "ellipse"] for lam in [0.1, 0.3, 1.0]]
        return cfg, args.seeds or [0, 1, 2]
    cfg = []
    for item in args.choice:
        g, rest = item.split("=")
        pool, lam = rest.split(",")
        cfg.append((g, args.backbone, args.lr, pool, float(lam), args.size))
    return cfg, args.seeds or list(range(10))


def run_dir(g, bb, lr, pool, lam, size, seed, window=3.0):
    return RUNS / f"findings_gaze-{g}_t3_seed{seed}_{tag(bb, lr, pool, lam, size, g, window)}"


def do_run(args):
    cfg, seeds = configs(args)
    jobs = [(c, s) for c in cfg for s in seeds]
    print(f"단계 {args.stage}: 설정 {len(cfg)}개 × seed {len(seeds)}개 = {len(jobs)}회")
    for k, ((g, bb, lr, pool, lam, size), s) in enumerate(jobs, 1):
        t = tag(bb, lr, pool, lam, size, g, args.window)
        if (run_dir(g, bb, lr, pool, lam, size, s, args.window) / "preds_test.csv").exists():
            print(f"[{k}/{len(jobs)}] 건너뜀 (이미 있음): {g} {t} seed {s}")
            continue
        print(f"\n[{k}/{len(jobs)}] {g} {t} seed {s}", flush=True)
        cmd = [sys.executable, str(TRAIN), "--task", "findings", "--gaze", g, "--seed", str(s),
               "--backbone", bb, "--lr", str(lr), "--schedule", SCHEDULE, "--pool", pool,
               "--gaze-weight", str(lam), "--img-size", str(size), "--tag", t,
               "--fs-window", str(args.window)] + args.extra.split()
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"  [오류] 이 run이 실패했습니다 (returncode {r.returncode}). 다음으로 넘어갑니다.")


def do_report(args):
    cfg, seeds = configs(args)
    if args.stage == "A2":  # A와 A2를 한 표에서 비교
        a = argparse.Namespace(**{**vars(args), "stage": "A"})
        cfg = configs(a)[0] + cfg
    rows = []
    for (g, bb, lr, pool, lam, size) in cfg:
        for s in seeds:
            d = run_dir(g, bb, lr, pool, lam, size, s, args.window)
            if not (d / "val_metrics.json").exists():
                continue
            m = json.load(open(d / "val_metrics.json"))
            h = pd.read_csv(d / "history.csv")
            rows.append({"gaze": g, "backbone": bb, "lr": lr, "size": size, "pool": pool, "lam": lam, "seed": s,
                         "val_score": (m["pleural_auroc"] + m["cardiac_auroc"]) / 2,
                         "pleural": m["pleural_auroc"], "cardiac": m["cardiac_auroc"], "edema": m["edema_auroc"],
                         "val_loss": h.val_cls_loss.min(), "best_epoch": m["best_epoch"],
                         "train_kl": h.loc[h.epoch == m["best_epoch"], "train_gaze_kl"].item()})
    if not rows:
        print("결과가 없습니다.")
        return
    r = pd.DataFrame(rows)
    keys = ["gaze", "backbone", "lr", "size", "pool", "lam"]
    agg = r.groupby(keys).agg(n=("seed", "size"), score=("val_score", "mean"), score_sd=("val_score", "std"),
                              pleural=("pleural", "mean"), cardiac=("cardiac", "mean"), edema=("edema", "mean"),
                              val_loss=("val_loss", "mean"), best_epoch=("best_epoch", "mean"),
                              train_kl=("train_kl", "mean")).reset_index()
    agg = agg.sort_values(["gaze", "score", "val_loss"], ascending=[True, False, True])
    pd.set_option("display.width", 220)
    agg["lr"] = agg["lr"].map(lambda x: f"{x:g}")
    print(f"=== 단계 {args.stage} val 결과 (seed 평균, score = 흉막·심장비대 AUROC 평균)")
    print(agg.round(4).to_string(index=False))
    out = HOME / f"results/sweep_stage{args.stage}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out, index=False)
    print("\n=== 조건별 선택 (score 최고, 같으면 val loss 낮은 쪽)")
    for g, d in agg.groupby("gaze"):
        b = d.iloc[0]
        print(f"  {g:9s} backbone={b.backbone} lr={b.lr} size={b['size']} pool={b.pool} λ={b.lam:g} → score {b.score:.4f} ± {b.score_sd:.4f} (n={b.n})")
    print(f"\n저장: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["run", "report"])
    ap.add_argument("--stage", choices=["A", "A2", "B", "C", "C3"], required=True)
    ap.add_argument("--backbone", choices=["imagenet", "chexpert"])
    ap.add_argument("--lr", type=float)
    ap.add_argument("--size", type=int, default=512, help="단계 B·C의 입력 크기 (A2에서 고른 값)")
    ap.add_argument("--choice", nargs="*", default=[])
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--window", type=float, default=3.0, help="fs 조건의 창 길이(초). 민감도 분석은 1.5, 5")
    ap.add_argument("--extra", default="", help='train.py에 그대로 넘길 인자. 예: "--max-images 64 --epochs 1"')
    args = ap.parse_args()
    if args.stage in ("B", "C", "C3") and (args.backbone is None or args.lr is None):
        raise SystemExit("단계 B·C·C3는 --backbone 과 --lr 이 필요합니다 (단계 A에서 고른 값).")
    do_run(args) if args.action == "run" else do_report(args)


if __name__ == "__main__":
    main()
