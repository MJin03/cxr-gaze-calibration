#!/usr/bin/env bash
# 23절 ConvNeXt 나머지 자동 실행 (ANALYSIS_PLAN.md 23절): 학습(sweep_nih.py) 끝나기 대기 → 외부 7곳 예측 → 판별력 평가(S0·H9) → 위치 평가(loc3)
# 로그: ~/X-ray/results/logs/cnx_pred.txt, cnx_eval.txt, cnx_loc.txt
# 진행 확인: python monitor_cnx.py (다른 tmux 창에서)
# 사용법  bash run_cnx.sh
set -euo pipefail
cd "$(dirname "$0")"
L="$HOME/X-ray/results/logs"
mkdir -p "$L"
AM="${CNX_ALLOW_MISSING:-}"   # 동작 확인용 환경변수 (실제 실행에서는 비어 있음)
CONDS="cnx_c1 cnx_hyb_none cnx_fs cnx_fs_shuffled"
DS7="chexpert_test vindr_train vindr_train_b chexpert_valid vindr_test padchest_a padchest_b"

echo "[$(date +%H:%M)] ConvNeXt 학습(sweep_nih.py) 끝나기 대기 (2분 연속 없으면 다음 단계)" | tee "$L/cnx_pred.txt"
n=0
while [ "$n" -lt 2 ]; do
  if pgrep -f "sweep_nih.py" > /dev/null; then n=0; else n=$((n + 1)); fi
  sleep "${CNX_WAIT_SEC:-60}"
done
# 최종 40 run이 다 있는지 확인 (없으면 멈춤: sweep_nih.py all --backbone convnext_t 를 다시 실행하면 이어서 학습)
python - <<'PY'
import os
import sys
from cal_common import CNX_CONDITIONS, S10, run_dir
if not CNX_CONDITIONS:
    sys.exit("results/cnx_choice.json 없음 → 학습이 끝나지 않았습니다")
miss = [(c, s) for c in CNX_CONDITIONS for s in S10 if not (run_dir(c, s) / "preds_test.csv").exists()]
print(f"ConvNeXt 최종 run {40 - len(miss)}/40 확인")
if miss and not os.environ.get("CNX_ALLOW_MISSING"):
    sys.exit(f"끝나지 않은 run {len(miss)}개 (예: {miss[0]}) → sweep_nih.py all --backbone convnext_t 다시 실행")
PY

echo "[$(date +%H:%M)] 1) 외부 7곳 예측" | tee -a "$L/cnx_pred.txt"
for d in $DS7; do
  python predict_external.py --dataset "$d" --conds $CONDS $AM 2>&1 | tee -a "$L/cnx_pred.txt"
done

echo "[$(date +%H:%M)] 2) 판별력 평가 (S0·H9)" | tee "$L/cnx_eval.txt"
python evaluate_discrimination.py --cond-set cnx --datasets $DS7 --tag cnx $AM 2>&1 | tee -a "$L/cnx_eval.txt"

echo "[$(date +%H:%M)] 3) 위치 평가 (기존·NIH·ConvNeXt 모두 다시 계산)" | tee "$L/cnx_loc.txt"
python evaluate_localization.py 2>&1 | tee -a "$L/cnx_loc.txt"
echo "[$(date +%H:%M)] ConvNeXt 모두 끝" | tee -a "$L/cnx_loc.txt"
