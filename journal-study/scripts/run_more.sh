#!/usr/bin/env bash
# 25절 전체 실행 (ANALYSIS_PLAN.md 25절): CPU 준비 → 다른 GPU 작업(23절 ConvNeXt 등) 끝나기 대기 → 학습 → 예측 → 평가
# 로그: ~/X-ray/results/logs/more_labels.txt, more_maps.txt, more_train.txt, more_pred.txt, more_eval.txt
# 진행 확인: python monitor_more.py (다른 tmux 창에서)
# 사용법  bash run_more.sh                 (다른 GPU 작업이 끝나면 시작)
#         MORE_NO_WAIT=1 bash run_more.sh  (기다리지 않고 동시에)
set -euo pipefail
cd "$(dirname "$0")"
L="$HOME/X-ray/results/logs"
mkdir -p "$L"
SEEDS="${MORE_SEEDS:-0 1 2 3 4 5 6 7 8 9}"   # 동작 확인용 환경변수 (실제 실행에서는 비어 있음)
EXTRA="${MORE_EXTRA:-}"
AM="${MORE_ALLOW_MISSING:-}"
CONDS="more_fs more_c1 more_hyb_none more_fs_shuffled"
EXT="chexpert_test_more chexpert_valid_more vindr_test_more vindr_train_more vindr_train_b_more padchest_a_more padchest_b_more"

if [ ! -f "$HOME/X-ray/data/processed/fs2_heatmap_index.csv" ]; then
  echo "[$(date +%H:%M)] 1) label·셔플 매핑 (CPU)"
  python build_more_labels.py 2>&1 | tee "$L/more_labels.txt"
  echo "[$(date +%H:%M)] 2) 무기폐·경화 시선 지도 (CPU)"
  python build_fs_heatmaps.py --finding-set more --windows 3 2>&1 | tee "$L/more_maps.txt"
else
  echo "label·시선 지도 이미 있음 → 건너뜀"
fi

if [ -n "${MORE_NO_WAIT:-}" ]; then  # MORE_NO_WAIT=1 bash run_more.sh → 다른 GPU 작업과 동시에 (GPU 메모리가 충분할 때)
  echo "[$(date +%H:%M)] 대기 없이 바로 시작 (다른 GPU 작업과 동시 실행)" | tee "$L/more_train.txt"
else
  echo "[$(date +%H:%M)] 다른 GPU 작업(ConvNeXt 등) 끝나기 대기 (2분 연속 없으면 시작)" | tee "$L/more_train.txt"
  n=0
  while [ "$n" -lt 2 ]; do
    if pgrep -f "sweep_nih.py|sweep_gazesource.py|sweep_gazefrac.py|predict_external.py|evaluate_localization.py" > /dev/null; then n=0; else n=$((n + 1)); fi
    sleep "${MORE_WAIT_SEC:-60}"
  done
fi

echo "[$(date +%H:%M)] 3) 학습 시작" | tee -a "$L/more_train.txt"
python sweep_more.py run --seeds $SEEDS --extra "$EXTRA" 2>&1 | tee -a "$L/more_train.txt"

echo "[$(date +%H:%M)] 4) 예측 시작" | tee "$L/more_pred.txt"
for d in internal_test_more $EXT; do
  python predict_external.py --dataset "$d" --conds $CONDS $AM 2>&1 | tee -a "$L/more_pred.txt"
done

echo "[$(date +%H:%M)] 5) 평가 시작" | tee "$L/more_eval.txt"
python evaluate_discrimination.py --cond-set more --datasets internal_test_more --tag more_internal $AM 2>&1 | tee -a "$L/more_eval.txt"
python evaluate_discrimination.py --cond-set more --datasets $EXT --tag more_ext $AM 2>&1 | tee -a "$L/more_eval.txt"
echo "[$(date +%H:%M)] 25절 모두 끝" | tee -a "$L/more_eval.txt"
