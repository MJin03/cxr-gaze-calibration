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

## 8. 추가 분석: 설정 개선 (2026-11-17 추가, 내부 test 확인 이후)

1~7절의 사전 고정 결과를 확인한 뒤 추가한다. 내부 test 결과는 이미 보았으므로 아래 결과는 **사후(post-hoc) 분석**으로 표시해 보고하고,
최종 확인은 한 번도 보지 않은 VinDr-CXR test로 한다. 1~7절의 결과는 그대로 함께 보고한다.

목표: 이전 연구 설정을 따르지 않고, 공정한 조건에서 각 조건이 낼 수 있는 가장 좋은 성능끼리 비교한다.

원칙
- 설정 선택은 val로만 한다. 선택 기준(모든 조건 동일): val AUROC(흉막 + 심장비대 평균)의 seed 평균 최고, 같으면 val 분류 loss 낮은 쪽
- none, real, shuffled는 같은 범위로 조정한다
- 시도한 설정은 모두 보고한다 (results/sweep_stage*.csv)

단계
- A. 공통 학습 설정 (Baseline으로 선택, seed 0~2)
  - backbone: ImageNet DenseNet-121 / CheXpert 사전학습 DenseNet-121 (torchxrayvision densenet121-res224-chex, MIMIC-CXR 미포함)
  - 학습률: 1e-4 / 3e-5, cosine schedule (1 epoch warmup)
- A2. (A 결과 확인 후 추가, 2026-11-17) CheXpert 가중치가 512 입력에서 ImageNet보다 크게 낮고 학습이 느렸음(best epoch 13~15).
  CheXpert 가중치는 224로 사전학습되어 입력 크기 차이가 원인일 수 있어 다음을 추가 확인한다 (seed 0~2):
  ImageNet 224 lr 1e-4, CheXpert 224 lr 1e-4, CheXpert 224 lr 3e-4, CheXpert 512 lr 3e-4.
  A와 A2 전체에서 같은 기준으로 하나를 고른다. 224를 고르면 gaze heatmap은 512에서 평균으로 줄여 쓴다.
- B. gaze 설정 (A에서 고른 backbone·학습률, seed 0~2)
  - none: pool {avg, attn(비지도 attention pooling)}
  - real, shuffled: pool {avg, attn(gaze로 학습한 attention pooling)} × λ {0.03, 0.1, 0.3, 1.0}
- A·A2 결과: ImageNet, 512, 학습률 1e-4, cosine 선택 (val score 0.884). CheXpert는 224에서 0.881로 근접했으나
  선택 기준상 2위이고, 224에서는 gaze가 7×7로 축소되어 이 연구 목적에 맞지 않음
- B 결과 (seed 0~2): 규칙상 선택 none avg (0.884), real avg λ0.03 (0.888), shuffled avg λ0.03 (0.888).
  gaze가 판단 위치에 직접 관여하는 attention pooling에서는 real이 shuffled보다 높음 (λ0.3: 0.879 vs 0.856, λ1.0: 0.878 vs 0.859)
- C. 조건별 설정을 seed 0~9로 학습 → 내부 test 평가 (사후, 1~7절과 같은 지표), seed 10개 앙상블 포함
  - C1 (주 비교): B의 선택 규칙대로 고른 설정 — none avg / real avg λ0.03 / shuffled avg λ0.03
  - C2 (보조 비교, B 결과를 보고 추가): gaze가 판단 위치를 정하는 같은 구조에서 진짜 시선의 정보 효과 확인 —
    none attn / real attn λ0.3 / shuffled attn λ0.3 (λ는 real의 val 최고값, 세 조건에 동일 적용)
- D. VinDr-CXR test 외부 검증 (최종 확인)
- E. 이전 연구(IWAIT 2027), 사전 고정 설정 결과, 개선 설정 결과 비교

통계 보완: 차이의 신뢰구간에 seed 변동을 반영하기 위해 seed와 test 환자를 함께 다시 뽑는 bootstrap을 추가한다 (1~7절 결과에도 함께 적용해 보고).
gaze heatmap sigma는 조정하지 않는다 (모델이 16×16으로 줄여 쓰므로 영향이 작음).
