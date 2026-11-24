"""22절 학습: 시선 출처별. fs(hybrid3, λ0.3)와 같은 설정에서 EGD(판독의 1명) 또는 REFLACX(판독의 5명) 영상에만 시선 지도를 줌.

조건 (cal_common.SRC_CONDITIONS): fs_src_egd, fs_src_reflacx. 비교 기준: hyb_none(0%), fs(전체), 17절 무작위 비율 곡선
seed 0~9. 끊긴 run은 처음부터 다시.

사용법  python sweep_gazesource.py run [--seeds 0 1] [--conds fs_src_egd]
"""
import argparse
import subprocess
import sys
from pathlib import Path

from cal_common import SRC_CONDITIONS, run_dir

TRAIN = Path(__file__).parent / "train.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["run"])
    ap.add_argument("--conds", nargs="*", default=list(SRC_CONDITIONS))
    ap.add_argument("--seeds", type=int, nargs="*", default=list(range(10)))
    ap.add_argument("--extra", default="")
    args = ap.parse_args()
    jobs = [(c, s) for c in args.conds for s in args.seeds]
    print(f"시선 출처별 학습 {len(jobs)}회: {args.conds} × seed {args.seeds}", flush=True)
    for k, (c, s) in enumerate(jobs, 1):
        d = run_dir(c, s)
        if (d / "preds_test.csv").exists():
            print(f"[{k}/{len(jobs)}] 건너뜀 (이미 있음): {c} seed {s}", flush=True)
            continue
        print(f"\n[{k}/{len(jobs)}] {c} seed {s} 시작", flush=True)
        src = c.replace("fs_src_", "")
        cmd = [sys.executable, str(TRAIN), "--task", "findings", "--gaze", "fs", "--seed", str(s), "--backbone", "imagenet",
               "--lr", "0.0001", "--schedule", "cosine", "--pool", "hybrid3", "--gaze-weight", "0.3",
               "--gaze-source", src, "--tag", SRC_CONDITIONS[c][1]] + args.extra.split()
        if (d / "best.pt").exists():
            print("  끊긴 run 발견 → 처음부터 다시 학습 (--overwrite)", flush=True)
            cmd.append("--overwrite")
        r = subprocess.run(cmd)
        print(f"[{k}/{len(jobs)}] {c} seed {s} " + ("완료" if r.returncode == 0 else f"실패 (returncode {r.returncode})"), flush=True)


if __name__ == "__main__":
    main()
