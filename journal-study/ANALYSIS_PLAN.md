# 분석 계획 (test 평가 전 고정)

작성: 2026-11-17, test set 예측값을 열어보기 전에 작성하고 커밋한다.
이 문서 이후의 변경은 커밋 기록으로 남기고, 논문에는 사전 계획과 사후 분석을 구분해 보고한다.

## 1. 연구 질문

radiologist gaze를 auxiliary loss로 학습한 모델이, 같은 referral 비율에서 AI가 처리한 사례의 놓친 양성(false negative)을 줄이는가.
평가 시 모델 입력은 X-ray 영상뿐이다.

## 2. 비교 조건 (6단계에서 학습 완료)

| 조건 | 설명 |
|---|---|
| none | Baseline, 영상만 |
| real | 실제 gaze heatmap으로 KL auxiliary loss (가중치 0.1) |
| shuffled | 다른 환자의 gaze heatmap (일대일 permutation) |

- 조건마다 seed 0~4, 같은 split(seed 42), 같은 학습 설정
- 라벨 기준: certainty ≥ 3 (주 분석)

## 3. 주 지표 (primary endpoint)

- 대상 finding: pleural abnormality, enlarged cardiac silhouette (pulmonary edema는 보조)
- 확률: finding별 temperature scaling(val에서 fitting) 후 양성 확률 p
- 확신도: confidence = max(p, 1 − p), 불확실한 사례부터 referral
- threshold: val에서 referral 비율이 25%가 되는 confidence threshold를 정하고 test에 고정 적용
- AI 처리 사례의 판정: p ≥ 0.5이면 양성
- **주 지표: missed-positive rate = (AI가 처리했고 음성으로 판정한 실제 양성 수) ÷ (test 전체 양성 수)**
  - 사람이 한 번도 보지 않고 놓치는 양성의 비율. 1 − missed-positive rate = referral 시스템 전체의 sensitivity (referral된 사례는 radiologist가 맞힌다고 가정)
- 함께 보고: false negative 수, AI 처리분 중 false negative 비율, 실제 test referral 비율

## 4. 주 비교

1. real vs none: gaze 학습의 효과
2. real vs shuffled: 영상과 맞는 진짜 gaze만의 효과

- 조건별 값: seed 5개 평균 ± 표준편차
- 차이의 95% 신뢰구간: test 환자 단위 bootstrap 2,000회 (threshold 고정, 같은 resample을 모든 조건·seed에 적용, seed 평균의 차이)

## 5. 보조 지표 (secondary)

- referral 비율 10%, 30%, 50%에서 같은 지표, 0~90% (5% 간격) referral 곡선
- 고정 sensitivity 방식: val에서 sensitivity 95%, 98%를 유지하는 auto-negative threshold → test에서 AI가 음성으로 처리 가능한 비율(workload reduction)과 실제 sensitivity
- AUROC, ECE (15 bins, temperature scaling 전후), Brier score
- AURC, E-AURC, 오류 탐지 AUROC (uncertainty로 오분류 구분)
- seed 5개 앙상블(확률 평균)의 같은 지표
- pulmonary edema의 같은 지표

## 6. 하위 분석

- AP / PA, landscape / portrait 별 주 지표와 AUROC (threshold는 전체 val 기준 그대로)
- 메타데이터 기준선: 촬영 방향(AP)과 사진 모양(landscape)만으로 train에서 logistic regression → test AUROC
- 5명 판독 test 109장: radiologist 의견 불일치(만장일치 아님)를 모델 불확실성이 구분하는지 (AUROC), AP/PA 구성 함께 보고
- certainty ≥ 4 라벨로 학습·평가한 결과 (민감도 분석, 추후)
- val gaze KL (real vs shuffled)

## 7. 해석 원칙

- 주 지표 결론은 real − none, real − shuffled 차이의 신뢰구간으로 판단한다
- 보조·하위 분석은 탐색적 결과로 보고한다
- test 결과를 본 뒤 학습 설정, 라벨 기준, threshold 방식을 바꾸지 않는다
