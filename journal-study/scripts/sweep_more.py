"""25절 학습: 소견 확장 (무기폐·경화, REFLACX 영상만). 기존 조건과 같은 설정, 새 조정 없음 (ANALYSIS_PLAN.md 25절).

조건 (cal_common.MORE_CONDITIONS)
    more_c1           pool avg, 시선 없음 (기존 c1_none과 같은 설정)
    more_hyb_none     pool hybrid3, 시선 없음
    more_fs           pool hybrid3, 소견별 시선 지도 (heatmaps_fs2_w3_128), λ 0.3
    more_fs_shuffled  pool hybrid3, 다른 환자의 소견별 시선 지도, λ 0.3
공통: DenseNet-121 ImageNet, 512, lr 1e-4, cosine, 20 epoch, patience 5, seed 0~9
순서: seed마다 네 조건을 차례로 (중간에 멈춰도 조건끼리 seed 수가 비슷하게). 끊긴 run은 처음부터 다시.

사용법  python sweep_more.py run [--seeds 0 1] [--conds more_fs]
"""
import argparse
import subprocess
import sys
from pathlib import Path

from cal_common import _TEST_SUFFIX, MORE_CONDITIONS, run_dir

TRAIN = Path(__file__).parent / "train.py"
SPEC = {  # 조건 → (gaze, pool, λ)
    "more_c1": ("none", "avg", "0"),
    "more_hyb_none": ("none", "hybrid3", "0"),
    "more_fs": ("fs", "hybrid3", "0.3"),
    "more_fs_shuffled": ("fs-shuffled", "hybrid3", "0.3"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["run"])
    ap.add_argument("--conds", nargs="*", default=list(MORE_CONDITIONS))
    ap.add_argument("--seeds", type=int, nargs="*", default=list(range(10)))
    ap.add_argument("--extra", default="", help="동작 확인용 추가 인자")
    args = ap.parse_args()
    jobs = [(c, s) for s in args.seeds for c in args.conds]
    print(f"소견 확장 학습 {len(jobs)}회: {args.conds} × seed {args.seeds}", flush=True)
    for k, (c, s) in enumerate(jobs, 1):
        d = run_dir(c, s)
        if (d / "preds_test.csv").exists():
            print(f"[{k}/{len(jobs)}] 건너뜀 (이미 있음): {c} seed {s}", flush=True)
            continue
        print(f"\n[{k}/{len(jobs)}] {c} seed {s} 시작", flush=True)
        gaze, pool, lam = SPEC[c]
        cmd = [sys.executable, str(TRAIN), "--task", "findings", "--finding-set", "more", "--gaze", gaze, "--seed", str(s),
               "--backbone", "imagenet", "--lr", "0.0001", "--schedule", "cosine", "--pool", pool, "--gaze-weight", lam,
               "--tag", MORE_CONDITIONS[c][1] + _TEST_SUFFIX] + args.extra.split()
        if (d / "best.pt").exists():
            print("  끊긴 run 발견 → 처음부터 다시 학습 (--overwrite)", flush=True)
            cmd.append("--overwrite")
        r = subprocess.run(cmd)
        print(f"[{k}/{len(jobs)}] {c} seed {s} " + ("완료" if r.returncode == 0 else f"실패 (returncode {r.returncode})"), flush=True)


if __name__ == "__main__":
    main()
