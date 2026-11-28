"""23절 ConvNeXt 진행 상황을 1분마다 한 줄로 보여 준다 (학습 → 예측 → 판별력 평가 → 위치 평가). Ctrl+C로 끝.

남은 시간: 최근에 끝난 것들 사이 간격(최대 5개 평균)으로 계산 → 다른 작업과 GPU를 나눠 쓰던 시간의 영향을 덜 받음
사용법  python monitor_cnx.py [--every 60]
"""
import argparse
import importlib
import time
from datetime import datetime, timedelta

import cal_common

N_TRAIN = 61   # 탐색 30 (c1 6, hyb_none 6, fs 18) + 최종 새 run 31 (seed 0~2 재사용)
DS7 = ["chexpert_test", "vindr_train", "vindr_train_b", "chexpert_valid", "vindr_test", "padchest_a", "padchest_b"]
LOGS = cal_common.RESULTS / "logs"


def last_line(f):
    try:
        lines = [x.strip() for x in f.read_text(errors="ignore").replace("\r", "\n").splitlines() if x.strip()]
        return lines[-1][:90] if lines else ""
    except OSError:
        return ""


def hm(sec):
    sec = int(max(sec, 0))
    return f"{sec // 3600}시간 {sec % 3600 // 60:02d}분"


def eta(times, n, total):
    if n >= total:
        return "곧 끝"
    if len(times) < 2:
        return "계산 중"
    k = min(5, len(times) - 1)
    rate = (times[-1] - times[-1 - k]) / k
    rem = rate * (total - n) - (time.time() - times[-1])
    rem = max(rem, 0)
    return f"{hm(rem)} (끝 약 {(datetime.now() + timedelta(seconds=rem)):%m-%d %H:%M}, 1개 약 {rate / 60:.0f}분)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=60)
    args = ap.parse_args()
    t_start = time.time()
    while True:
        importlib.reload(cal_common)   # cnx_choice.json이 학습 중간에 생김
        tr = sorted(p.stat().st_mtime for p in cal_common.RUNS.glob("*bb-convnext_t*/preds_test.csv"))
        cnx = list(cal_common.CNX_CONDITIONS)
        runs = [cal_common.run_dir(c, s) for c in cnx for s in cal_common.S10]
        preds = [d / f"preds_{ds}.csv" for d in runs for ds in DS7]
        pr = sorted(p.stat().st_mtime for p in preds if p.exists())
        pooled = cal_common.RESULTS / "eval_disc_cnx/pooled.csv"
        loc = cal_common.RESULTS / "localization/summary.csv"
        if len(tr) < N_TRAIN or not cnx:
            msg = f"단계 1/4 학습 | {len(tr)}/{N_TRAIN} | 남은 {eta(tr, len(tr), N_TRAIN)} | 마지막: {last_line(LOGS / 'cnx_pred.txt')}"
        elif len(pr) < len(preds):
            msg = (f"단계 2/4 외부 7곳 예측 | {len(pr)}/{len(preds)} | 남은 {eta(pr, len(pr), len(preds))}"
                   f" | 마지막: {last_line(LOGS / 'cnx_pred.txt')}")
        elif not pooled.exists() or pooled.stat().st_mtime < pr[-1]:
            msg = f"단계 3/4 판별력 평가 (약 30~40분) | 마지막: {last_line(LOGS / 'cnx_eval.txt')}"
        elif not loc.exists() or loc.stat().st_mtime < pooled.stat().st_mtime:
            msg = f"단계 4/4 위치 평가 (약 20~30분) | 마지막: {last_line(LOGS / 'cnx_loc.txt')}"
        else:
            print(f"[{datetime.now():%H:%M}] 모두 끝 → logs/cnx_eval.txt, logs/cnx_loc.txt 보내 주세요", flush=True)
            break
        print(f"[{datetime.now():%H:%M}] {msg} (모니터 {hm(time.time() - t_start)})", flush=True)
        time.sleep(args.every)


if __name__ == "__main__":
    main()
