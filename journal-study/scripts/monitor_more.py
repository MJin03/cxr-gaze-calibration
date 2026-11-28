"""25절 진행 상황을 1분마다 한 줄로 보여 준다 (학습 → 예측 → 평가). Ctrl+C로 끝.

사용법  python monitor_more.py [--every 60]
"""
import argparse
import time
from datetime import datetime, timedelta

from cal_common import DATASETS, MORE_CONDITIONS, PROC, RESULTS, S10, run_dir

LOGS = RESULTS / "logs"
STEPS = [("학습", "more_train.txt"), ("예측", "more_pred.txt"), ("평가", "more_eval.txt")]


def last_line(f):
    try:
        lines = [x.strip() for x in f.read_text(errors="ignore").replace("\r", "\n").splitlines() if x.strip()]
        return lines[-1][:90] if lines else ""
    except OSError:
        return ""


def hm(sec):
    sec = int(max(sec, 0))
    return f"{sec // 3600}시간 {sec % 3600 // 60:02d}분"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=60)
    args = ap.parse_args()
    runs = [run_dir(c, s) for s in S10 for c in MORE_CONDITIONS]
    while True:
        dsets = [k for k, v in DATASETS.items() if k.endswith("_more") and v["findings"]]
        if not dsets:  # label 파일이 아직 없으면 (준비 중) 다시 읽기
            import importlib
            import cal_common
            importlib.reload(cal_common)
            dsets = [k for k, v in cal_common.DATASETS.items() if k.endswith("_more") and v["findings"]]
        now = time.time()
        done_tr = [d for d in runs if (d / "preds_test.csv").exists()]
        preds = [d / f"preds_{k}.csv" for d in runs for k in dsets]
        done_pr = [p for p in preds if p.exists()]
        holm = RESULTS / "eval_disc_more_ext/holm.csv"
        if len(done_tr) < len(runs):
            stage, n, total = 1, len(done_tr), len(runs)
            t_list = sorted((d / "config.json").stat().st_mtime for d in runs if (d / "config.json").exists())
        elif len(done_pr) < len(preds):
            stage, n, total = 2, len(done_pr), len(preds)
            # 내부 test는 파일만 옮겨 금방 끝나므로 속도 계산에서 뺀다
            ext_pr = [p for p in done_pr if "internal_test_more" not in p.name]
            t_list = sorted(p.stat().st_mtime for p in ext_pr) or [now]
            n_ext = len(ext_pr)
            n, total = n_ext, len([p for p in preds if "internal_test_more" not in p.name])
        else:
            stage, n, total = 3, int((RESULTS / "eval_disc_more_internal/pooled.csv").exists()) + int(holm.exists()), 2
            t_list = []
        logf = LOGS / STEPS[stage - 1][1]
        if stage == 1 and not t_list and not (PROC / "fs2_heatmap_index.csv").exists():
            f = max([LOGS / "more_labels.txt", LOGS / "more_maps.txt"], key=lambda x: x.stat().st_mtime if x.exists() else 0)
            msg = f"단계 0/3 준비(CPU: label·시선 지도) | 마지막: {last_line(f)}"
        elif stage == 1 and not t_list:
            msg = f"단계 0/3 다른 GPU 작업 끝나기 대기 | 마지막: {last_line(LOGS / 'more_train.txt')}"
        elif stage == 3 and n == 2 and holm.stat().st_mtime > max(p.stat().st_mtime for p in done_pr):
            print(f"[{datetime.now():%H:%M}] 모두 끝 (학습 {len(runs)}, 예측 {len(preds)}, 평가 2) → logs/more_eval.txt 보내 주세요", flush=True)
            break
        else:
            el = now - t_list[0] if t_list else 0
            rate = el / n if n and t_list else None
            rem = rate * (total - n) if rate else None
            eta = f"{hm(rem)} (끝 약 {(datetime.now() + timedelta(seconds=rem)):%m-%d %H:%M})" if rem is not None else "계산 중"
            msg = (f"단계 {stage}/3 {STEPS[stage - 1][0]} | {n}/{total} | 경과 {hm(el)} | 남은 {eta}"
                   f" | 마지막: {last_line(logf)}")
        print(f"[{datetime.now():%H:%M}] {msg}", flush=True)
        time.sleep(args.every)


if __name__ == "__main__":
    main()
