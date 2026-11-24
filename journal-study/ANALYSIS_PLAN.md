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
- C 결과 (seed 0~9, 내부 test, 환자+seed bootstrap CI 기준):
  - C1: 세 조건 간 유의한 차이 없음
  - C2: 심장비대 missed@25% real−none −0.068 [−0.123, −0.014], AUROC real−none +0.016~0.018 (세 finding),
    흉막 AUROC real−shuffled +0.017 [+0.002, +0.035]. 단 C1 none(avg) 대비(eval_C2x)는 유의한 차이 없음
  - 사전 고정 설정(1~7절)에 환자+seed CI를 적용하면 유의한 차이 없음
- C3. finding별 시선 (finding-specific gaze), 2026-11-18 추가 — C 결과 확인 후, 실행 전에 아래를 고정한다
  - 근거: gaze 지도 하나가 세 finding의 시선을 섞고 있고, 한 곳만 보는 attention pooling은 여러 곳에 생기는 흉막에 불리했음
  - finding별 gaze 지도: 판독 녹취에서 finding 관련 단어를 말하기 시작하기 3초 전부터 말을 마칠 때까지의 고정점만 사용
    (창 길이 3초 고정·튜닝하지 않음, 1.5초·5초는 민감도 분석으로만 보고). 지도 만드는 법은 4-2단계와 같음 (머문 시간 가중, sigma 7, 합 1)
    - 단어: 흉막 effusion, pleural, costophrenic, blunting / 심장비대 cardiomegaly, cardiac, heart, cardiomediastinal /
      폐부종 edema, congestion, vascular, interstitial, kerley
    - 단어 일치는 소문자·문장부호 제거 후 어근으로 시작하는지로 판단 (예: effusions, blunted, pleura, vasculature 포함)
    - 창과 겹치는 고정점만 쓰고 창 안에 머문 시간을 가중치로 함. 512에서 만든 지도를 4×4 합산해 128×128로 저장
    - 부정 표현("no effusion" 등)도 포함 (시선이 가르치는 것은 "어디를 볼지"이고 병 유무는 label이 가르침)
    - 해당 단어가 없는 판독은 그 finding의 gaze loss를 계산하지 않음 (mask). 시선 2개인 영상은 있는 지도끼리 평균
    - 폐부종 단어 보완 (2026-11-18, 지도 생성 후·학습 전): 위 단어(v1)로는 train 폐부종 음성 영상의 6.3%만 지도가 생김
      (정상일 때 "no edema"라고 잘 말하지 않음). 다음 세트를 train·val의 coverage와 병변 타원 겹침만으로 비교해 하나를 고른다
      (학습·test 결과는 보지 않음, 흉막·심장비대 단어는 바꾸지 않음)
      - v2 = v1 + perihilar, hilar, haz(hazy/haziness), cephaliz, overload, fluid
      - v3 = v2 + lung ("lungs are clear"처럼 폐 전체를 보고 하는 말. 폐부종은 폐 전체에 퍼지는 소견이라 볼 위치가 폐 전체)
      - 선택 규칙: 폐부종 타원 안 시선 비율이 전체 시선 지도보다 높게 유지되는 세트 중 음성 coverage가 가장 높은 세트
      - 결과 (train coverage 음성 / 타원 안 비율, 전체 시선 0.382): v1 0.063 / 0.584, v2 0.165 / 0.594, v3 0.625 / 0.579 → 규칙상 v3
    - 지도 확인 (학습 전, train·val): 병변 타원 안에 들어간 시선 비율이 finding별 지도(3초)에서 전체 시선 지도보다 높음
      (흉막 0.409 vs 0.171, 심장비대 0.507 vs 0.285, 폐부종 0.584 vs 0.382; 영상의 84~92%에서 더 높음)
  - 모델: finding별 attention 지도(3장) + finding별 hybrid pooling (전체 평균 특징과 자기 attention 특징을 이어 붙여 판단)
  - 조건: none (λ0) / real-global (기존 전체 시선 지도를 세 attention에 동일 적용) / real-fs (finding별 시선) /
    shuffled-fs (C와 같은 셔플 매핑으로 다른 환자의 finding별 시선) / ellipse (참고 비교: REFLACX 판독의가 그린 병변 타원,
    certainty ≥3, 양성 판독에만 있음, EGD는 mask)
  - 주 비교: real-fs vs none, real-fs vs shuffled-fs. 보조: real-fs vs real-global, real-fs vs ellipse, C1 none(avg) 대비
  - λ 선택: gaze 조건마다 λ {0.1, 0.3, 1.0} × seed 0~2, 위 선택 기준(val)으로 각자 고름 → seed 0~9로 학습 → 내부 test 평가 (사후)
  - C3 λ 결과 (val score, seed 0~2): real-global λ1.0 0.8905, fs λ0.3 0.8919, fs-shuffled λ0.3 0.8792, ellipse λ0.1 0.8965,
    none(hybrid3) 0.8635. real-global·ellipse가 범위 끝에서 골라져 네 gaze 조건 모두 λ 0.03, 3.0을 추가 확인 (2026-11-18)
    → 선택 변화 없음, 모든 선택이 범위 안쪽. 본 실행: none λ0 / real λ1.0 / fs λ0.3 / fs-shuffled λ0.3 / ellipse λ0.1, seed 0~9
  - 평가: eval_C3 (none / fs / fs-shuffled), eval_C3_vsC1 (C1 none avg / fs / fs-shuffled),
    eval_C3_ref (real-global / fs / ellipse: fs−real-global, fs−ellipse)
- D. VinDr-CXR test 외부 검증 (최종 확인)
- E. 이전 연구(IWAIT 2027), 사전 고정 설정 결과, 개선 설정 결과 비교

통계 보완: 차이의 신뢰구간에 seed 변동을 반영하기 위해 seed와 test 환자를 함께 다시 뽑는 bootstrap을 추가한다 (1~7절 결과에도 함께 적용해 보고).
gaze heatmap sigma는 조정하지 않는다 (모델이 16×16으로 줄여 쓰므로 영향이 작음).

## 9. VinDr-CXR 외부 검증 계획 (2026-11-18, VinDr 예측 전에 고정)

VinDr-CXR test는 지금까지 한 번도 예측하지 않았다 (label 개수만 확인). 아래 계획대로 한 번만 실행하고,
결과를 본 뒤 설정·전처리·비교를 바꾸지 않는다. 8절의 내부 test 결과(사후)와 구분해 보고한다.
코드: scripts/vindr_common.py에 조건·비교·마진을 그대로 고정.

1. 대상과 label
- VinDr-CXR test 3,000장 (영상의학과 전문의 5명 합의 label). 변환 실패 영상은 제외하고 개수를 보고
- 흉막 = Pleural effusion 또는 Pleural thickening (259장, 학습의 pleural abnormality와 같은 정의)
- 심장비대 = Cardiomegaly (309장)
- 민감도 분석: 흉막을 Pleural effusion만으로 (111장)
- 폐부종: 양성 0장 → 평가하지 않음

2. 영상 전처리 (label·예측을 보지 않고 결정)
- DICOM VOI LUT 적용, MONOCHROME1 반전, 최솟값~최댓값을 0~255로, 비율 유지 없이 512×512 (BILINEAR, 내부와 같음)
- plain(그대로)과 eq(histogram equalization) 중 내부 train 영상(1,000장 표본)과 평균 히스토그램 L1 거리가 작은 쪽을 사용

3. 모델 (재학습 없음, 저장된 best.pt 그대로)
- fs (hybrid3 λ0.3), c1_none (avg), fs_shuffled (hybrid3 λ0.3), real_global (hybrid3 λ1.0), hyb_none (hybrid3 λ0),
  ellipse (hybrid3 λ0.1): 각 seed 0~9
- 1~7절 사전 고정 설정 none / real / shuffled: 각 seed 0~4
- temperature와 referral 기준(확신도 문턱)은 내부 val에서 정한 값을 그대로 사용

4. 비교
- 주 비교: fs − c1_none (가장 강한 기준선), fs − fs_shuffled (시선 내용 효과)
- 보조: fs − real_global (finding별 분리 효과), fs − hyb_none, 사전 고정 설정의 real − none, real − shuffled
- 비열등: fs − ellipse, AUROC 차이의 ci2 하한이 −0.02보다 크면 "수작업 타원 대비 비열등"

5. 지표
- 주 지표: referral 25%에서 missed-positive rate (흉막, 심장비대). 주요 보조: AUROC
- seed 평균과 앙상블(seed 10개 또는 5개의 보정 확률 평균) 둘 다 보고
- 실제 referral 비율을 함께 보고. 보조로 VinDr 자신의 확신도 분포로 정확히 25%를 보내는 경우(label 불필요)의 missed
- 그 밖: ECE, Brier, AURC, E-AURC, error-detection AUROC, workload@민감도 95·98%, referral 10·30·50%, referral 곡선

6. 신뢰구간 (bootstrap 2000회, 영상 단위: VinDr는 환자 ID가 없음)
- seed 평균: ci = 영상만 / ci2 = 영상 + 조건별 seed 복원추출 (판정은 ci2)
- 앙상블: ci = 영상만 / ci2 = 영상 + 앙상블 구성 seed 복원추출 (판정은 ci2)
- 판정: ci2가 0을 포함하지 않으면 차이 있음. 주 비교 2개 × finding 2개는 각각 보고 (다중 비교 보정 없음, 해석 시 명시)

7. pointing game (hybrid3 조건만, avg 모델은 attention 지도가 없음)
- 박스가 있는 양성 영상에서 finding별 attention(16×16)이 가장 강한 칸이 해당 finding 박스(흉막 = effusion·thickening 박스)와 겹치면 hit
- 보조: 박스 안 attention 비율 (칸별 박스 덮임 비율 × attention softmax의 합)
- seed 평균 ± SD, 앙상블(attention softmax 평균), fs − 다른 조건 차이의 영상+seed bootstrap CI

## 10. 추가 분석 (사후): 순위 기반 자동 문턱 (2026-11-19, VinDr 9절 결과 확인 후, 이 분석 실행 전에 고정)

배경: 9절의 주 지표에서 내부 val로 고정한 확신도 문턱은 VinDr에서 실제로 2.5~12.5%만 보냈다
(VinDr는 정상 영상 68%, 양성 비율이 내부의 1/3 수준). 조건마다 실제 referral 비율이 달라 missed 비교가 공정하지 않았다.
9절 결과(사전 계획)는 그대로 보고하고, 아래는 VinDr 결과를 본 뒤 추가한 사후 분석으로 표시한다.
모델·예측값은 바꾸지 않는다 (재학습·재예측 없음, AUROC·pointing game 결과는 변하지 않음).

규칙: 최근 들어온 영상 W장의 확신도 중 하위 25%에 들면 의사에게 보낸다. 병원별 설정·label이 필요 없다.
- 시작할 때는 그 run의 내부 val 확신도(보정 후)에서 W장을 복원추출해 쓴다
- 영상은 무작위 순서로 들어온다고 보고 시뮬레이션, 순서 20번의 평균 (모든 조건이 같은 순서를 씀)
- 과거 영상의 확신도만 쓰므로 평가 영상 자신을 보고 문턱을 정하지 않는다
- W = 500 (주), 200·1000 (민감도 분석). W와 25%는 조정하지 않는다
- 내부 test와 VinDr test 모두에 같은 규칙 적용 (finding: 내부 흉막·심장비대·폐부종, VinDr 흉막·심장비대·effusion만)
- 지표: 실제 referral 비율, missed-positive rate. 비교는 9절과 같은 목록 (fs − c1_none, fs − fs_shuffled 등)
- 신뢰구간: 내부 환자 단위, VinDr 영상 단위 bootstrap 2000회. seed 평균 ci2(+seed 복원추출), 앙상블 ci(영상/환자만)
- 한계로 적을 점: 업무량(25%)을 지키는 규칙이며 안전(민감도)을 보장하지 않음. 실제 병원에서는 영상 순서가 무작위가 아닐 수 있음.
  처음 W장 동안은 내부 val 분포로 시작하므로 referral이 25%에서 약간 벗어날 수 있음
코드: scripts/evaluate_rolling.py

## 11. 추가 연구: 시선 난이도로 보정 개선 (2026-11-19, 학습 전에 고정)

배경 (사후 확인 결과): 8~10절의 모든 조건에서 temperature scaling 전후 ECE가 seed 변동 안에서 비슷했다
(gaze는 보정을 개선하지 않음, IWAIT 주장 재현 안 됨). 또 내부 val로 맞춘 temperature가 VinDr에서는 ECE를 오히려 키웠다
(흉막 raw 0.027~0.028 → TS 후 0.042~0.044). fs는 "어디를 볼지"를 가르치고 "얼마나 확신할지"는 가르치지 않는다.
→ 시선으로 "얼마나 확신할지"를 가르치는 방법을 새로 검증한다. 내부 test와 VinDr test는 이미 보았으므로
주 판정은 한 번도 보지 않은 CheXpert test에서 한다.

1. 시선 난이도 d (scripts/build_gaze_difficulty.py, train·val만)
- finding별 시선 창(8절 C3과 같은 3초 창, 같은 단어 세트 v3)에서 세 특징: 창 안에 머문 시간 합, 창 개수(다시 말한 횟수),
  고정점 위치의 퍼짐(머문 시간 가중 표준편차, 영상 크기로 나눔)
- (finding, label, 데이터셋) 묶음 안에서 각 특징을 백분위로 → 세 백분위 평균을 다시 백분위로 → d ∈ (0, 1]
  (label 자체와 EGD·REFLACX 녹취 길이 차이가 난이도에 섞이지 않게). 그 finding을 말하지 않은 영상은 d = 0.5
- 판독의 확신도 난이도 u (REFLACX): 양성(확신도 3~5) u = (5 − 확신도)/2, 음성(0~2) u = 확신도/2, EGD만 있는 영상 0.5

2. 모델과 조건 (기반: fs, hybrid3, λ0.3, ImageNet, lr 1e-4, cosine, 512 — 8절과 같음)
- 학습 target = y(1 − ε) + ε/2, ε = ε_max × 난이도 (train 영상만, val·test label과 early stopping·temperature는 정답 label 그대로)
- A gaze (제안): 난이도 = d / B gaze-shuffled (대조): 다른 환자의 d (8절과 같은 셔플 매핑)
  C certainty (비교): 난이도 = u / D uniform (비교): 난이도 = 0.5 (A와 같은 평균 강도의 일반 smoothing)
- 기준: smoothing 없는 fs (8절), C1 none (avg)
- ε_max {0.05, 0.1, 0.2} × seed 0~2. 조건마다 val NLL(temperature 전 val 분류 loss) seed 평균이 가장 낮은 ε_max 선택,
  같으면 val AUROC(흉막·심장비대 평균) 높은 쪽 → seed 0~9로 학습 (scripts/sweep_smooth.py)
- ε 결과 (val, seed 0~2, val NLL): gaze 0.05 0.3264 (SD 0.0036, 전체 최저), gaze-shuffled 0.1 0.3311, certainty 0.05 0.3337,
  uniform 0.05 0.3372, smoothing 없음 0.3309. gaze·certainty·uniform이 범위 아래 끝에서 선택되어 네 조건 모두 ε 0.02 추가 (2026-11-20)
  → gaze 0.05, gaze-shuffled 0.1, uniform 0.05 (모두 범위 안쪽), certainty 0.02 (0.3317, smoothing 없음과 거의 같아 더 줄이면 "없음"에 수렴).
  본 실행: gaze 0.05 / gaze-shuffled 0.1 / certainty 0.02 / uniform 0.05, seed 0~9

3. 평가
- 데이터 변경 (2026-11-20, 예측 전): Stanford AIMI(Redivis)에서 CheXpert test 668장은 현재 제공되지 않음을 확인
  (CheXpert Plus에는 train과 validation 234장만 있음). 주 판정 데이터를 다음 둘로 바꾼다
  - CheXpert validation 234장 (환자 200명, 전문의 3명 합의 label): 다른 나라·다른 병원
  - VinDr-CXR train에서 무작위 3,000장 (image_labels_train.csv의 고유 image_id를 pandas sample(n=3000, random_state=0)으로 선택,
    label은 판독의 3명 과반수): 같은 병원(VinDr test와 같은 기관)이지만 한 번도 보지 않은 영상, 큰 표본
  - 두 데이터 모두에서 같은 방향이면 "효과 있음", 한쪽만이면 그대로 보고
- 데이터 재변경 (2026-11-20, CheXpert 예측 전): 이전 AIMI 사이트(stanfordaimi.azurewebsites.net)의 CheXlocalize 데이터에
  CheXpert test 영상(668장, 환자 500명)과 전문의 label(test_labels.csv: 5명 과반수, val_labels.csv: 3명 합의)이 함께 있음을 확인.
  Redivis CheXpert Plus의 label은 판독문에서 자동 추출한 것이라 쓰지 않는다. 주 판정 데이터를 원래 계획에 가깝게 다시 정한다
  - 주: CheXpert test 668장 (전문의 5명 과반수) + VinDr train 무작위 3,000장 (위와 같음)
  - 보조: CheXpert validation 234장 (전문의 3명 합의, val_labels.csv)
  - CheXpert는 정면(frontal) 영상만 사용. label 대응: 흉막 = Pleural Effusion (CheXpert 전문의 label에 Pleural Other가 있으면 포함),
    심장비대 = Cardiomegaly, 폐부종 = Edema, 민감도 분석 = Pleural Effusion만. 세부는 파일 확인 후 예측 전에 추가 기록
- CheXpert 세부 (2026-11-20, 파일 구조·영상 수만 확인, label 분포·예측 보기 전에 고정. scripts/build_chexpert_cache.py)
  - 정면 판정: test는 경로의 viewN_frontal (668장 중 518장), validation은 Frontal/Lateral 열 = Frontal. 한 검사에 정면이 여러 장이면 모두 사용
  - label: 1만 양성, 나머지(0·빈칸) 음성. 흉막 = Pleural Effusion = 1 또는 Pleural Other = 1, 흉막(민감도) = Pleural Effusion = 1,
    심장비대 = Cardiomegaly = 1, 폐부종 = Edema = 1. 폐부종은 CheXpert에서만 평가 (VinDr에는 해당 label 없음)
  - 영상: JPG를 흑백으로 읽어 그대로(plain)와 histogram equalization(eq) 두 가지를 512×512로 축소(BILINEAR),
    내부 train 1,000장과 평균 히스토그램 L1 거리가 작은 쪽을 사용 (9절 VinDr와 같은 규칙, label·예측은 보지 않음)
  - 신뢰구간: CheXpert는 환자 단위 bootstrap (같은 환자의 여러 영상을 함께 뽑음, labels.csv의 group), VinDr train은 영상 단위
  - CheXlocalize의 분할(segmentation) 정답을 이용한 위치 평가는 이 판정에 넣지 않는다 (하게 되면 사후로 표시)
- (원래 계획) 주 판정 데이터: CheXpert test (전문의 합의 label, 한 번도 보지 않음). label 대응과 세부 사항은 데이터 확인 후,
  예측 전에 이 절에 추가로 고정한다. 보조: VinDr test (새 조건은 처음 평가하지만, 이 방향을 VinDr 보정 결과를 보고 정했으므로 보조),
  내부 test (사후)
- 주 지표: temperature scaling 없이 잰 calibration slope (예측 logit에 대한 로지스틱 재보정 기울기, 1이 이상적)와 NLL
- 보조: calibration intercept, ECE(raw·TS 후), Brier, AUROC(판별력이 떨어지지 않는지), 자동 문턱(10절) missed
- 주 비교: A − D (일반 smoothing 대비 시선 난이도의 효과), A − B (시선 내용 효과).
  보조: A − C, A − fs(smoothing 없음), A − C1 none
- 신뢰구간: 영상(내부는 환자) + seed bootstrap 2000회 (ci2)
- 해석: A가 D와 B보다 calibration slope가 1에 유의하게 가깝고 NLL이 낮으며 AUROC가 떨어지지 않으면
  "시선 난이도가 보정을 개선한다"고 판정. 결과와 관계없이 8~10절의 "fs는 보정을 개선하지 않음"은 그대로 보고

## 11-2. 추가 보정 방법 (2026-11-20, 주 판정 데이터를 보기 전에 고정)

배경: 11절 smoothing은 val에서 가장 좋았지만, 보조 데이터(내부 test, VinDr test)에서 seed 수준으로는
일반 smoothing과 보정 차이가 없었다. 또 다른 병원(VinDr)에서 모든 모델이 확신 부족(slope > 1)이었고, 정도가 영상마다 달랐다.
smoothing은 학습 label만 바꾸므로 test에서 영상별로 확신을 조절하지 못한다.
→ 아래 A를 추가하고, B·C를 보조 분석으로 고정한다. 이 셋까지 val로만 고른 뒤 주 판정 데이터(CheXpert test, VinDr train)에서
11절 smoothing과 함께 한 번에 평가한다. 이후 주 판정 데이터를 본 뒤에 추가하는 분석은 모두 사후로 표시한다.

A. 난이도 예측 헤드 + 영상별 logit 축소 (train.py --diff-target, --diff-weight; sweep_diff.py)
- fs(hybrid3, λ0.3) 위에 finding별 난이도 예측 d̂ = sigmoid(w·특징 + b)를 두고,
  출력 logit = 원래 logit / (1 + softplus(β)·d̂). β는 학습으로 정해지며 test에서 따로 맞추는 값이 없다
- 정답: gaze(시선 난이도 d, 그 finding을 말한 train 영상만) / gaze-shuffled(다른 환자의 d) / certainty(u, REFLACX만) /
  free(같은 구조, 난이도 지도 학습 없음). 난이도 loss = BCE(d̂, 정답) × μ
- μ {0.1, 0.3, 1.0} × seed 0~2, 조건마다 val NLL 최저(같으면 val AUROC) → seed 0~9. 범위 끝에서 선택되면 같은 범위로 한 칸 확장
- μ 결과 (val, seed 0~2, val NLL): gaze 0.1 0.3363 (SD 0.0155) / gaze-shuffled 0.3 0.3298 / certainty 1.0 0.3274 / free 0.3344
  (참고: fs 그대로 0.3309, 11절 smoothing gaze 0.05 0.3264). gaze는 아래 끝, certainty는 위 끝에서 선택
  → 11절과 같이 네 조건 모두 μ 0.03, 3.0 추가 (2026-11-20, 주 판정 데이터 보기 전)
  - 진단: gaze 난이도 loss(BCE)가 0.686~0.690으로, 상수 0.5 예측(ln2 = 0.693)과 거의 같음 (완전 예측 하한 ≈ 0.5).
    셔플은 0.694 → 영상 특징으로 시선 난이도를 거의 예측하지 못함. 결과 해석 시 함께 보고
- 확장 결과 (μ 0.03~3.0): gaze 3.0 0.3224 (SD 0.0078, 전체 최저) / gaze-shuffled 0.3 0.3298 / certainty 1.0 0.3274 / free 0.3344.
  gaze가 다시 위 끝(3.0)에서 선택 → 같은 규칙으로 세 조건 모두 μ 10 추가 (주 판정 데이터 보기 전).
  μ 10이 마지막 확장이며, 10에서 선택되어도 더 넓히지 않고 10을 쓴다 (난이도 loss가 분류 loss의 약 20배가 되는 지점)
  - 진단: μ 3에서 gaze 난이도 loss 0.680 (상수 예측 대비 설명력 약 7%), 셔플 μ 3은 0.694 (설명력 없음)
  - 주의: 5~6개 μ 중 최저값을 고르므로 seed 0~2 val NLL은 낙관적일 수 있음 → seed 0~9 본 학습의 val 결과를 함께 보고
- μ 10 결과: gaze 0.3402, gaze-shuffled 0.3345, certainty 0.3631 → 세 조건 모두 범위 안쪽에서 선택, 확장 종료.
  확정 (2026-11-20): A-gaze μ 3 / A-gaze-shuffled μ 0.3 / A-certainty μ 1 / A-free → seed 0~9 (sweep_diff.py final)
- 본 학습 val 결과 (seed 0~9, val NLL / 흉막·심장비대 AUROC): A-gaze 0.3340 ± 0.0108 / 0.8864, A-free 0.3396 ± 0.0161 / 0.8904,
  A-gaze-shuffled 0.3394 ± 0.0139 / 0.8865, A-certainty 0.3348 ± 0.0100 / 0.8872, fs 그대로 0.3374 / 0.8850, smoothing gaze 0.3345 / 0.8903.
  seed 0~2 값보다 모든 조건이 0.006~0.012 높아짐 (A-gaze 0.3224 → 0.3340) → μ 선택의 낙관 편향과 seed 0~2가 전반적으로 좋았던 것이 함께 확인됨.
  A-gaze는 val NLL이 가장 낮지만 차이가 seed 표준편차보다 작고, AUROC는 A-free보다 0.004 낮음.
  (report의 "조건별 선택" 줄은 seed 수가 다른 행이 섞여 있어 쓰지 않음. μ는 위 확정값 유지)
- 주 비교: A-gaze − A-free (난이도 지도 학습의 효과), A-gaze − A-gaze-shuffled (시선 내용의 효과)
  보조: A-gaze − A-certainty, A-gaze − smoothing gaze(11절), A-gaze − fs, A-gaze − C1 none
- 지표·판정은 11절과 같음 (temperature 없는 calibration slope와 NLL, ci2)

B. attention 퍼짐을 불확실성으로 (보조, 학습 없음)
- hybrid3 모델의 finding별 attention softmax의 정규화 엔트로피(0~1)를 불확실성 점수로 쓴다
- 비교 점수: 확신도만(1 − max(p, 1 − p)) / 엔트로피만 / 둘의 순위 평균 (가중치 조정 없음)
- 지표: 오류 탐지 AUROC, AURC, 자동 문턱(10절, W500) 25% referral에서의 missed
- 대상: A 조건들, smoothing gaze, fs (외부 데이터 예측 시 attention 저장: predict_external.py)
- 세부 (2026-11-20, 주 판정 데이터 예측 전 고정. scripts/evaluate_uncertainty.py)
  - 확률은 temperature scaling 후 값(10절과 같음), 오류 = (p ≥ 0.5) ≠ label
  - 엔트로피: 그 finding의 attention logit(16×16) softmax 엔트로피 / log 256
  - 순위 평균의 기준 분포와 자동 문턱 시작값: 같은 run의 내부 val 영상 점수 (predict_external.py --dataset internal_val, label 사용 없음)
  - 자동 문턱: 최근 500장 중 점수 상위 25%면 referral, 무작위 순서 20번 평균. 같은 (finding, 점수)는 모든 조건이 같은 순서
  - 앙상블: 확률·엔트로피 각각 seed 평균 후 점수 계산
  - 비교: 같은 조건 안에서 순위 평균 − 확신도, 엔트로피 − 확신도 (ci2). 조건 간 주장에는 쓰지 않음

C. 병원별 label 없는 재보정 (보조, gaze와 무관한 운영 방법)
- 새 병원의 label 없는 test 영상 전체로 양성 비율을 EM(Saerens 2002)으로 추정해 확률을 조정
  (내부 train 양성 비율을 사전확률로 시작). 조정 전후의 calibration slope, intercept, NLL을 보고
- 모든 조건에 같은 방식으로 적용하며, "gaze가 보정을 개선한다"는 주장의 근거로 쓰지 않는다
- 세부 (2026-11-20, 예측 전 고정. evaluate_calibration.py): temperature 없는 확률에 적용, 시작값 π0 = 내부 train 양성 비율(finding별),
  수렴 1e-8 또는 1000회, 추정 π는 0.001~0.999로 제한. 조정은 logit + logit(π) − logit(π0)인 상수 이동이라
  calibration slope와 AUROC는 그대로이고 intercept, NLL, ECE, Brier만 바뀜

평가 데이터와 순서 (2026-11-20 고정): 주 = CheXpert test(정면 518장, 환자 단위), VinDr train 3,000장(영상 단위, 판독의 3명 과반수:
흉막 241, effusion 136, 심장비대 367, 전처리 eq) / 보조 = CheXpert validation(정면 202장), VinDr test, 내부 test.
11절 smoothing과 11-2절 A·B·C를 같은 실행에서 한 번에 평가하고, 이후 추가 분석은 사후로 표시한다

## 11-3. 주 판정 결과 (2026-11-20, CheXpert test 518장 + VinDr train 3,000장, 한 번 평가)

판정 기준(11절 3, 11-2절): seed 평균 ci2. 11절 smoothing과 11-2절 A 모두 "시선이 보정을 개선한다" 기준 미충족.
- 11절 gaze − uniform (주): 두 데이터 모든 finding에서 NLL·|slope−1| 차이 없음
- 11절 gaze − gaze-shuffled (주): |slope−1| 개선 (CheXpert 심장비대 −0.337, 부종 −0.285 / VinDr 흉막 −0.199, 심장비대 −0.328, effusion −0.298),
  VinDr 흉막 NLL −0.014. 단 셔플은 ε 0.1, gaze·uniform은 ε 0.05라 smoothing 강도 차이로 설명되고, 같은 강도의 uniform과는 차이 없음
- 11-2절 a_gaze − a_free (주): 대부분 차이 없음, CheXpert 심장비대 |slope−1| +0.277 (나빠짐). 앙상블은 모든 finding에서 slope가 1에서 더 멀어짐
- 11-2절 a_gaze − a_gaze-shuffled (주): 보정 차이 없음, VinDr 흉막 AUROC +0.018
- 모든 모델이 두 데이터에서 확신 부족(slope > 1). 확신을 줄이는 방법(smoothing, logit 축소)은 외부에서 방향이 맞지 않음
- 판별력: smoothing gaze − C1 none AUROC CheXpert 흉막 +0.018, effusion +0.018 / VinDr 흉막 +0.025, effusion +0.023 (ci2 유의)
  → 8절 VinDr test의 fs 판별력 개선이 새 외부 데이터 두 곳에서 재현됨 (보조 비교)
- B (보조): attention 엔트로피의 오류 탐지 AUROC 0.25~0.55 (0.5 아래, 예상과 반대 방향). 순위 평균은 확신도만보다 모든 경우 나쁨.
  엔트로피 분포가 내부 val과 달라 referral 비율이 25%를 넘어(최대 0.40) missed 비교는 해석 불가
- C (보조): EM 추정 양성 비율이 크게 틀림 (VinDr 심장비대 0.001 vs 실제 0.122), NLL이 모든 조건에서 나빠짐.
  새 병원에서 모델 보정 자체가 바뀌어(확신 부족) label shift 가정이 성립하지 않음
- 보조 데이터 (2026-11-20): VinDr test·내부 test에서도 같은 양상. 11절 gaze − uniform 모두 n.s.,
  a_gaze − a_free는 내부 흉막 |slope−1| −0.121만 유의(그 외 n.s., 앙상블은 대부분 악화).
  smoothing gaze − C1 none AUROC: VinDr test 흉막 +0.024, effusion +0.018 / 내부 흉막 +0.015, 심장비대 +0.016 (ci2 유의)
  → 흉막 판별력 개선은 내부 + 외부 3곳 모두에서 유의. 내부 slope는 0.82~1.28로 1 근처
- B 보조 (VinDr test, 2026-11-20): 주 판정과 같음. 엔트로피 오류 탐지 AUROC 0.22~0.52, 순위 평균 < 확신도 (모든 조건·finding 유의)
- 코드 수정: C 표의 intercept(EM 후)가 발산하는 계산 문제(citl 뉴턴 단계 제한 없음) 수정. NLL·ECE·Brier와 주 비교에는 영향 없음

## 11-4. 사후 진단: 외부 확신 부족의 원인 (2026-11-20, scripts/diagnose_underconfidence.py, 이미 본 데이터 사용)

- 내부 test slope 0.82~1.13 (1 근처) vs 외부 1.0~1.48 → 외부에서 확신 부족이 생김
- 내부 test 중 판독의 5명 영상 109장: 같은 예측의 slope가 과반수 label 기준 1.26~2.01, 판독의 비율 label 기준 0.93~1.17 (부종 0.68, 양성 7건)
  → 모델은 "판독의 1명이 양성이라고 할 확률"을 배움 (학습 label 대부분이 판독의 1명). 여러 명 합의 label로 재면 확신 부족으로 보임.
  모의 실험(판독의 1명 기준으로 완벽 보정된 모델)에서도 3명 과반수 기준 slope 1.55, 비율 기준 1.00
- VinDr train: 과반수 → 비율 label로 바꾸면 흉막 1.17→1.02, effusion 1.48→1.34, 심장비대 1.20→1.16 (흉막은 대부분 설명, 나머지는 일부)
  → 남은 부분은 병원 변화로 봄. 만장일치 영상 slope 1.2~1.8, 의견 갈린 영상 0.4~0.9
- 앙상블: 확률 평균은 seed 평균보다 slope를 0.2~0.5 키움, logit 평균은 그 증가를 약 절반으로 줄임
- 해석: 외부 확신 부족의 주원인은 "학습 label(판독의 1명)과 평가 label(여러 명 합의)의 정의 차이" + 일부 병원 변화 + 확률 평균 앙상블

## 11-5. 새 확인용 데이터 (2026-11-20, 다운로드 전 고정)

- VinDr-CXR train에서 기존 3,000장(vindr_train_3000.txt)을 뺀 나머지 고유 image_id 중 pandas sample(n=3000, random_state=1)
  → ~/X-ray/data/lists/vindr_train_3000b.txt. label·전처리 규칙은 VinDr train 3,000장과 같음 (판독의 3명 과반수, VinDr test 전처리)
- 용도: 11-4절 진단에서 나온 개선 방법(판독 방식 보정 등)의 확인용. 방법과 모든 값을 내부 데이터로 정해 이 절에 고정한 뒤 한 번만 평가.
  CheXpert validation도 아직 평가하지 않았으므로 같은 시점에 확인용(보조)으로 함께 평가
- 다운로드 완료 (2026-11-21): 3,000장 모두 받음 (첫 시도에서 55장이 연결 문제로 빠져 같은 목록으로 다시 받음, 덜 받은 파일 없음)

## 11-6. 판독 방식 보정 (reader-protocol recalibration) — 확인 평가 전 고정 (2026-11-20, scripts/reader_recalibration.py)

동기: 11-4절 진단 (이미 본 데이터로 착안 → 착안 자체는 사후임을 논문에 명시). 방법과 모든 값은 아래로 고정하고 이후 바꾸지 않는다.
바꾸면 새 확인용 데이터가 다시 필요하다.

방법
- q = temperature scaling 후 확률 (내부 val, 대부분 판독의 1명 label에 맞춘 확률)
- 판독의 k명의 투표를 beta-binomial로 모형화: 판독의 1명의 양성 확률 θ ~ Beta(평균 q, 판독의 간 상관 ρ)
  보정 확률 = P(k명 중 과반수 양성) = Σ_{v > k/2} BetaBinomial(v; k, q(1−ρ)/ρ, (1−q)(1−ρ)/ρ)
- ρ: 내부 REFLACX 판독의 5명 영상(내부 test 109장, 확신도 ≥ 3 투표)에서 모델 q가 주어졌을 때 투표 가능도 최대.
  조건·seed·finding마다 따로, 격자 {0.001, 0.005, 0.01, 0.02, …, 0.99}. 외부 label 사용 없음
- k (데이터 label 방식, 데이터 설명 기준): VinDr train·새 VinDr train 3명 과반수(직접 계산), CheXpert validation 3명(다수결),
  CheXpert test 5명 과반수, VinDr test 5명 합의(투표가 아닌 합의라 k=5는 근사, 사후 참고용)
- 앙상블: seed별 보정 확률의 logit 평균
- 모의 실험 확인: ρ = 0.3인 판독의 투표를 만들면 ρ 추정 0.30, 3명 과반수 기준 slope가 TS 1.15 → 보정 0.99 (독립 가정 0.71로 과보정)

평가 (확인용, 한 번만)
- 확인 데이터: 주 = 새 VinDr train 3,000장(vindr_train_b), 보조 = CheXpert validation 202장. 사후 참고: CheXpert test, VinDr train, VinDr test
- 조건: c1_none, fs, gaze(smoothing), uniform, a_gaze, a_free. 주 조건 = fs, c1_none
- 비교 (같은 조건 안): 보정 − TS, 보정 − raw (주), 독립 가정(ρ=0) − TS (민감도). 지표: NLL, |slope − 1| (주), ECE (보조)
- 신뢰구간: 영상(CheXpert는 환자) + seed bootstrap 2000회 (ci2), 앙상블은 영상만
- 판정: vindr_train_b에서 fs와 c1_none 모두, 흉막·심장비대 각각 "보정 − TS"의 NLL과 |slope − 1|이 함께 유의하게 음수면
  그 finding에서 "개선". 반대 방향으로 유의하면 "악화". 나머지는 "판단 불가". AUROC는 단조 변환이라 바뀌지 않음

### 11-6 확인 결과 (2026-11-22, 새 VinDr train 2,997장, 규칙대로 한 번 평가)
- 판정 (fs, c1_none, 보정 − TS, seed ci2):
  - 흉막: NLL 유의하게 감소 (c1 −0.0079, fs −0.0078), |slope−1|은 n.s. (c1 −0.081 [−0.163, +0.025], fs −0.105 [−0.194, +0.004]) → "판단 불가"
  - 심장비대: NLL 유의하게 증가 (c1 +0.0219, fs +0.0261), |slope−1| n.s. → "악화"
- 보정 − raw: 흉막·effusion 차이 거의 없음 → 흉막에서의 이득은 내부 temperature가 만든 확신 부족(TS slope 1.17~1.39 vs raw 1.01~1.18)을
  되돌린 정도이며, raw보다 낫지 않음
- 심장비대: 보정 후 intercept가 커짐 (c1 +0.92 → +1.36). VinDr가 심장비대를 모델 예측보다 자주 양성으로 판정하는데
  과반수 변환은 확률을 더 낮추므로 악화 → 판독의 수 차이가 아니라 병원별 판독 기준 차이가 주원인으로 보임
- 독립 가정(ρ=0): effusion은 크게 개선(slope 1.0~1.1), 흉막은 과보정(0.70~0.84), 심장비대는 크게 악화 → finding마다 방향이 달라 일반 해법이 아님
- 참고: 3,000장 중 2,997장 평가. 3장은 DICOM 픽셀 데이터가 잘린 파일(다운로드 중 손상)이라 변환 단계에서 자동 제외
  (1d36704f…, 7959604f…, 4522ac24…). label과 무관한 기술적 손상이고 0.1%라 다시 받지 않고 제외한 채 보고
  (확인 데이터를 다시 평가하지 않기 위함)
- 보조 확인 (CheXpert validation 202장, k=3): 같은 방향. 흉막 보정 − TS: NLL n.s. (c1 +0.006, fs +0.000), |slope−1| fs −0.222 ✓, c1 n.s. → 판단 불가.
  심장비대 NLL 악화 ✓ (c1 +0.036, fs +0.037). 폐부종: slope는 1에 가까워짐(fs 1.67 → 1.07) but intercept 악화(+0.92 → +1.93)로 NLL 악화 ✓.
  CheXpert validation은 Pleural Other 양성이 없어 흉막 = effusion
- 해석: 보정은 확률의 퍼짐(slope)은 1 쪽으로 옮기지만, 외부 병원이 모델 예측보다 양성을 더 많이 판정하는 기저율 차이(intercept)를 키움.
  외부 오보정 = (1) 판독 방식에 따른 퍼짐 차이 (label 없이 일부 교정 가능) + (2) 병원별 기저율·판독 기준 차이 (label 필요)
- 결론: label 없는 판독 방식 보정은 외부 보정을 일관되게 개선하지 못함. 외부 오보정은 판독의 수뿐 아니라 병원별 판독 기준 차이에서 옴

## 12. 판별력 외부 검증 확장 (2026-11-22, 이 평가 실행 전에 고정. scripts/evaluate_discrimination.py)

목적: 9절(VinDr test 한 곳)의 핵심 비교를 외부 데이터 여러 곳에서 확인하고 합친다. 이미 학습된 모델만 사용 (재학습·조정 없음)
- 조건: fs, c1_none, fs_shuffled, real_global, hyb_none, ellipse (8·9절과 같은 run, seed 0~9)
- 비교: fs − c1_none, fs − fs_shuffled (주) / fs − real_global, fs − hyb_none (보조) / fs − ellipse (비열등, AUROC 마진 0.02)
- 지표: AUROC (주, temperature 무관), 10절 자동 문턱 missed (보조, W500, 하위 25%, 내부 val 시작, 순서 20번)
- 데이터: CheXpert test, VinDr train, 새 VinDr train(2,997장), CheXpert validation, VinDr test.
  finding: 흉막, 심장비대, effusion(흉막 출력), 폐부종(CheXpert만)
  - 이미 본 정도: VinDr test는 9절에서 이 비교를 이미 봄. 나머지는 smoothing gaze·A 조건만 봤고 fs_shuffled·real_global·hyb_none·ellipse는 처음
- 신뢰구간: 환자(CheXpert)/영상(VinDr) + seed bootstrap 2000회 (ci2), 앙상블은 영상만
- 통합: 데이터별 seed 평균 차이와 ci2 폭으로 구한 표준오차로 random-effects(DerSimonian-Laird) 통합, I² 함께 보고.
  VinDr 3개는 같은 병원이므로 "독립 병원 2곳(VinDr, CheXpert)"임을 명시하고, 병원별(VinDr 3개 통합 / CheXpert 2개 통합) 결과도 함께 제시
- 해석: 주 비교 2개가 통합 추정에서 유의하고 데이터별 방향이 일관되면 "외부에서 일관된 시선 효과". 비열등은 통합 하한 > −0.02

### 12절 결과 (2026-11-22, 외부 5곳: CheXpert test·validation, VinDr train·새 train·test)
- AUROC 통합 (random-effects, seed 평균 차이 [95% CI]):
  - fs − c1_none: 흉막 +0.018 [+0.008, +0.029]✓, 심장비대 +0.018 [+0.009, +0.028]✓, effusion +0.011✓, 폐부종(CheXpert 2곳) +0.020✓
  - fs − fs_shuffled: 흉막 +0.027✓, 심장비대 +0.025✓, effusion +0.023✓ (VinDr·CheXpert 병원별 통합 모두 유의), 폐부종 +0.009 (n.s.)
  - fs − real_global: 흉막 +0.000 (차이 없음), 심장비대 +0.025✓ (병원별 모두 유의) → finding별 분리는 심장비대에서만 이득
  - fs − hyb_none: 흉막 +0.033✓, 심장비대 +0.034✓
  - fs − ellipse: 흉막 +0.009 [+0.001, +0.016] 비열등✓(통합 하한 > 0), 심장비대 +0.005 비열등✓, effusion 비열등✓, 폐부종 −0.007 (하한 −0.024, 비열등 아님)
  - I² 0~33%: 데이터 간 이질성 낮음. CheXpert만 통합 시 흉막 fs − c1 +0.008 (n.s.), 나머지 주 비교는 병원별로도 유의
- 자동 문턱 missed (보조): 흉막 fs − c1 −0.034✓, fs − shuffled −0.031✓, 심장비대 fs − shuffled −0.031✓.
  폐부종은 fs가 더 놓침(+0.084✓): fs의 폐부종 확률이 외부에서 확신 부족(slope 1.67)이라 0.5 문턱 아래로 떨어지는 양성이 많기 때문 (순위 지표 AUROC는 개선)
- 결론: 시선 학습의 판별력 개선과 "시선 내용" 효과가 독립 병원 2곳·데이터 5곳에서 일관되게 재현됨. 시선은 의사 타원 표시와 비열등
- 11-6 사후 참고 (보정 − TS): 흉막·effusion은 CheXpert test·VinDr train·VinDr test에서 대부분 NLL·slope 함께 개선(✓),
  심장비대·폐부종은 모든 데이터에서 NLL 악화. 단 raw 대비로는 이득이 거의 없음(새 VinDr·CheXpert validation 기준)

## 13. 정보 수준별 베이지안 재보정 (2026-11-22, 실행 전에 고정. scripts/bayes_recalibration.py)

동기 (사후 착안): 11절~11-6절에서 label 없는 보정은 모두 실패. 외부 오보정 = 퍼짐 + 기저율. 병원이 가진 정보(없음 / 유병률 / label n장)에 따라
방법을 따로 고르는 대신, 하나의 베이지안 틀로 정보가 들어오는 만큼 갱신한다. 학습·모델 변경 없음.
- 모형: p = sigmoid(a + b·u), 사전 a ~ N(0, 1²), b ~ N(1, 0.5²) (정보 없으면 기준 확률 그대로). MAP 추정
  - u(기준 logit): reader = 11-6절 판독 방식 보정 확률의 logit (대상의 판독 방식 k, 내부 추정 ρ) / ts = 내부 TS logit (비교)
  - 유병률 가능도: 대상 영상 평균 예측 확률 ~ N(π_기록, (0.25·π_기록)²) / label 가능도: Bernoulli
- 현실적 평가: 같은 병원의 다른 묶음을 "과거 기록"으로 사용 (평가 묶음의 정답은 쓰지 않음)
  VinDr(train, 새 train, test) 6쌍, CheXpert(validation, test) 2쌍. π_기록 = 기록 묶음 양성 비율, label n장 = 기록 묶음에서 무작위 추출(100번)
- 방법: 정보 없음 raw·ts·reader / 유병률 prev_ts·prev_reader (+ π × 0.5·0.75·1.25·1.5 민감도) /
  label n ∈ {10, 25, 50, 100, 200}: ts_refit·platt·bayes_ts·bayes_reader·bayes_reader_prev / 참고 하한 oracle(대상 전체로 맞춘 platt)
- 조건 fs, c1_none / finding 흉막, 심장비대, effusion, 폐부종(CheXpert) / 지표 NLL(주), |slope−1|, intercept, ECE
- 판정 (쌍·조건·seed 평균, finding별):
  H1 유병률: prev_reader의 NLL이 raw보다 낮고, 쌍의 과반수에서 낮으면 "유병률 하나로 개선"
  H2 label 효율: n ≤ 50에서 bayes_reader가 platt과 ts_refit보다 NLL 낮으면 "적은 label에서 베이지안 틀이 유리",
     bayes_reader < bayes_ts이면 "판독 방식 보정이 좋은 사전 믿음"
  H3 견고성: π × 0.75, 1.25에서도 prev_reader가 raw보다 NLL 낮은지
- 한계: VinDr 3묶음은 label 방식이 다름(train 3명 투표 / test 5명 합의) → 기록과 대상의 정의 차이가 그대로 포함됨(현실적 오차로 봄)

### 13절 결과 (2026-11-24, VinDr 6쌍 + CheXpert 2쌍, 규칙대로 한 번 평가)

NLL (쌍·조건·seed 평균). gap 회복 = (raw − prev_reader) / (raw − oracle)

| 병원 · finding | raw | prev_reader | oracle | gap 회복 | H1 쌍 | H3 (×0.75 / ×1.25) |
|---|---|---|---|---|---|---|
| VinDr · pleural | 0.2036 | 0.1986 | 0.1955 | 62% | 6/6 ✓ | ✓ / ✓ |
| VinDr · cardiac | 0.2946 | 0.2636 | 0.2522 | 73% | 6/6 ✓ | ✓ / ✓ |
| VinDr · effusion | 0.1234 | 0.0884 | 0.0850 | 91% | 6/6 ✓ | ✓ / ✓ |
| CheXpert · pleural | 0.3843 | 0.3996 | 0.3523 | −48% | 0/2 ✗ | ✗ / ✗ |
| CheXpert · cardiac | 0.5285 | 0.4691 | 0.4540 | 80% | 2/2 ✓ | ✓ / ✓ |
| CheXpert · edema | 0.3589 | 0.3447 | 0.3197 | 36% | 2/2 ✓ | ✓ / ✓ |
| CheXpert · effusion | 0.3826 | 0.4011 | 0.3488 | −55% | 0/2 ✗ | ✗ / ✗ |

- H1 (유병률 하나로 개선): 7개 중 5개 충족. CheXpert pleural·effusion은 악화 (기록 묶음 validation 3명 합의 → 대상 test 5명 과반수: 기록과 대상의 label 정의가 달라 π_기록이 맞지 않는 것으로 추정, 사전에 적은 한계)
- H2 label 효율: n ≤ 50에서 bayes_reader < platt·ts_refit 21칸 중 19칸 충족 (예외: CheXpert pleural·effusion n=50에서 ts_refit이 약간 우세).
  platt·ts_refit은 label 10~25장에서 raw보다 크게 나쁨 (NLL 최대 2~3배) → 적은 label로 다시 맞추면 오히려 해로움. 베이지안 틀은 이 손해를 막음
- H2 사전 믿음: bayes_reader < bayes_ts는 VinDr effusion만 충족 → 판독 방식 보정이 TS보다 좋은 사전 믿음이라는 근거 없음
- H3 견고성: H1을 충족한 5개 모두 π × 0.75·1.25에서도 raw보다 좋음. × 0.5·1.5에서는 일부 악화 → 유병률 오차 ±25% 안에서 견고
- 유병률 + label (bayes_reader_prev): label을 더해도 prev_reader 대비 이득이 작음 (200장에서 oracle에 근접). 즉 이득의 대부분은 유병률에서 옴
- 부가 관찰: prev_reader는 NLL·기저율을 고치지만 cardiac에서 |slope−1|이 커짐 (VinDr 0.21→0.50) → 퍼짐(slope)은 유병률로 고쳐지지 않음
- 정리 (프로토콜): 정보 없음 → raw 유지 (TS·reader는 일관된 이득 없음) / 같은 label 방식의 유병률 있음 → prev_reader (gap의 62~91% 회복) /
  label 있음 → bayes_reader_prev (platt·ts_refit은 100장 미만에서 쓰지 않음) / ⚠️ 기록과 대상의 label 정의가 다르면 유병률이 오히려 해로울 수 있음
- PadChest(14절, a↔b 같은 label 방식)가 이 프로토콜의 최종 확인 (H1 충족 예상)

## 14. PadChest 세 번째 병원 확인 (2026-11-22, 영상 다운로드·예측 전에 고정. scripts/build_padchest_cache.py)

- 데이터: PadChest (스페인 San Juan 병원, BIMCV). 메타데이터(label CSV)만 먼저 받아 아래 규칙을 정함. 영상·모델 예측은 아직 보지 않음
- 대상: MethodLabel = Physician(의사가 판독문에서 직접 label), Projection ∈ {PA, AP, AP_horizontal}, 소아(PED) 제외
- zip: 의사 label 정면 영상이 있는 0~12, 54번 중 **0~11, 54번** (13개, 약 282GB). 12번은 22.5GB에 306장뿐이라 제외 (label·결과와 무관한 효율 기준)
- 제외: label 해석 불가·빈 label, "exclude" 포함, label이 "unchanged"뿐인 영상 (판독문이 소견을 나열하지 않아 음성이 불확실)
- label 대응 (VinDr·CheXpert와 같은 정의를 따름)

| 평가 label | PadChest label | 비고 |
|---|---|---|
| pleural_effusion | pleural effusion, loculated pleural effusion | 민감도 분석 (VinDr·CheXpert와 같음) |
| pleural | pleural_effusion + pleural thickening, apical pleural thickening, calcified pleural thickening, pleural plaques, calcified pleural plaques, hydropneumothorax, empyema, pleural mass | 주 결과 (VinDr: effusion 또는 thickening, CheXpert: Effusion 또는 Pleural Other) |
| cardiac | cardiomegaly | 주 결과 |
| edema | pulmonary edema, kerley lines | 보조 결과 (양성 약 100장으로 적음) |
| pleural_cpa | pleural + costophrenic angle blunting | 민감도 분석 |
| edema_hi | edema + heart insufficiency | 민감도 분석 |

- 민감도 분석: "suboptimal study" 영상 제외, "unchanged"가 있는 영상 전부 제외
- reader 수: 판독문 1명 → **k = 1** (11-6 판독 방식 보정에서 단일 판독). 예상: 단일 판독 label이므로 확신 부족(slope > 1)이 VinDr·CheXpert보다 작아야 함 (11-4 진단의 검증)
- 묶음: batch a = zip 0~5, batch b = zip 6~11, 54 → 13절의 같은 병원 다른 묶음 평가(a→b, b→a)에 사용
- 환자 단위 bootstrap (group = PatientID)
- 영상: 16-bit PNG → plain(최솟값~최댓값) / win(CSV의 WindowCenter·Width) / eq(plain에 histogram equalization) → 512×512 직접 축소 → 내부 train과 히스토그램 L1 거리 최소인 방식 선택 (무작위 3,000장, label·예측 사용 안 함)
- 흑백 반전: finalize의 qc_examples.png(MONOCHROME1·2 각 12장)를 사람이 보고 예측 전에 결정 (none / mono1 / all)
- 평가: 12절 판별력(같은 비교·지표, PadChest a·b를 외부 데이터로 추가해 random-effects 통합), 11-6 판독 방식 보정, 13절 베이지안 재보정. 결과를 본 뒤 규칙은 바꾸지 않음

### 14절 준비 결과 (2026-11-22, 예측 전)

- 다운로드·변환: zip 13개, 대상 25,304장 전부 성공 (실패 0), 환자 23,307명. batch a 11,754장 / b 13,550장
- 양성 수 (a / b / 합계): pleural 1,149 / 1,229 / 2,378, pleural_effusion 561 / 610 / 1,171, cardiac 1,100 / 1,322 / 2,422, edema 37 / 60 / 97
- 전처리 선택: eq (내부 train과 히스토그램 L1 0.114; plain 0.458, win 1.817 — CSV window 값이 PNG 밝기와 맞지 않음)
- 흑백 방향: 전부 MONOCHROME2. PadChest 평균 영상(무작위 500장)과 내부 train 평균 영상(500장)의 상관 0.93 → 반전 없음 (invert = none). label·예측 사용 안 함

### 14절 결과 ① 13절 베이지안 재보정 — PadChest a↔b (2026-11-24, 규칙대로 한 번 평가)

| finding | raw | prev_reader | oracle | gap 회복 | H1 쌍 | H3 (×0.75 / ×1.25) |
|---|---|---|---|---|---|---|
| pleural | 0.2721 | 0.2597 | 0.2492 | 54% | 2/2 ✓ | ✓ / ✓ |
| cardiac | 0.2457 | 0.2367 | 0.2324 | 68% | 2/2 ✓ | ✓ / ✓ |
| edema (양성 97장) | 0.0823 | 0.0211 | 0.0199 | 98% | 2/2 ✓ | ✓ / ✓ |
| pleural_effusion | 0.1574 | 0.1047 | 0.1028 | 97% | 2/2 ✓ | ✓ / ✓ |
| pleural_cpa (15절 통일 label) | 0.3070 | 0.2963 | 0.2925 | 74% | 2/2 ✓ | ✓ / ✓ |
| edema_hi (민감도) | 0.1072 | 0.0809 | 0.0757 | 83% | 2/2 ✓ | ✓ / ✓ |

- H1: 6개 모두 충족 (같은 병원·같은 label 방식이면 유병률 하나로 gap의 54~98% 회복) → 13절 결론(유병률은 label 정의가 같을 때 통함)을 세 번째 병원에서 확인
- H2: n ≤ 50에서 bayes_reader < platt·ts_refit 18칸 중 17칸 (예외: edema n=10에서 platt이 근소하게 우세). platt은 label 10~25장에서 slope가 크게 틀어짐(|slope−1| 최대 4)
- H2 사전 믿음: PadChest는 k = 1이라 판독 방식 보정이 TS와 같음(변환 없음) → bayes_reader = bayes_ts (비교 불가, 정상)
- H3: 6개 모두 ×0.75·×1.25에서도 raw보다 좋음
- 15절 범주 범위: pleural_cpa는 pleural보다 NLL이 전반적으로 높음(CPA 둔화를 포함하면 모델이 덜 잡는 양성이 늘어남). 유병률 보정 후 oracle까지 남은 차이는 둘 다 작음(0.0105 / 0.0038) → 범주 범위 차이는 주로 기저율로 나타나고 유병률 보정이 흡수
- 판독 1명 label에서 확신 부족이 작은지(11-4 설명)는 부호가 있는 slope가 필요 → 11-6 판독 보정 평가(reader_pc)에서 판정

### 14절 결과 ② 12절 판별력 — 외부 7곳 (2026-11-24, eval_disc_7sites, 규칙대로 한 번 평가)

통합(전체) random-effects, AUROC 차이 [95% CI]. ✓ = CI가 0 미포함

| 비교 | pleural | cardiac | effusion | edema (n=4) |
|---|---|---|---|---|
| fs − c1_none (주) | +0.013 [+0.008, +0.019] ✓ | +0.020 [+0.014, +0.026] ✓ | +0.002 [−0.006, +0.010] | +0.020 [+0.007, +0.033] ✓ |
| fs − fs_shuffled (주) | +0.019 [+0.009, +0.028] ✓ | +0.019 [+0.014, +0.024] ✓ | +0.014 [+0.004, +0.024] ✓ | +0.007 [−0.004, +0.017] |
| fs − real_global (보조) | +0.002 | +0.021 ✓ | −0.002 | +0.006 |
| fs − hyb_none (보조) | +0.024 ✓ | +0.031 ✓ | +0.020 ✓ | +0.008 |
| fs − ellipse (비열등) | +0.003 비열등✓ | −0.000 비열등✓ | +0.002 비열등✓ | −0.003 비열등✓ |

- PadChest만 (a·b 통합): fs − c1_none pleural +0.011 ✓, cardiac +0.022 ✓, edema +0.020 ✓, effusion −0.007 ✓(fs가 낮음), pleural_cpa +0.011 ✓, edema_hi +0.031 ✓
- 5곳(12절) 대비: pleural·cardiac 주 비교는 세 번째 병원에서도 재현. effusion fs − c1_none은 5곳에서 +0.011 ✓였으나 PadChest에서 반대 방향 → 7곳 통합 유의하지 않음 (I² 56%)
- missed (보조): pleural −0.019 ✓, cardiac −0.027 ✓, effusion +0.003, edema +0.007 (CheXpert +0.084 ✓ 악화 / PadChest −0.046 ✓ 개선, I² 68%)
- 절대 AUROC: PadChest pleural 0.78~0.79로 다른 곳(0.82~0.91)보다 낮음, effusion은 0.94로 높음 → PadChest pleural에 많은 apical pleural thickening(1,048장) 등 범주 범위 차이가 판별력에도 영향 (15절)
- 해석: 소견별 시선(fs)의 판별력 이득은 세 병원·7개 데이터에서 흉막 이상·심장비대·폐부종에 일관(+0.013~+0.020 AUROC), 시선 내용을 섞으면 사라짐(fs − fs_shuffled). 흉수 단독은 병원마다 달라 결론 유보

## 15. label 개념 통일 (2026-11-24, 14절 PadChest 평가 결과를 보기 전에 고정)

동기: 외부 label 차이 = ① 범주 범위(어떤 소견을 포함하나) + ② 판독 방식(같은 이름을 얼마나 엄격하게 붙이나, 11-4).
키워드 통일은 ①만 다룸 (②는 11-6 판독 방식 보정과 16절 순서형 출력이 담당).
- 기준 정의: 내부 학습 label(REFLACX)의 개념 — Pleural abnormality / Enlarged cardiac silhouette / Pulmonary edema
- 단순 키워드 일치는 쓰지 않음 (예: heart → artificial heart valve, effusion → pericardial effusion이 잘못 잡힘). 포함·제외 목록으로 고정

| 통일 개념 | VinDr | CheXpert | PadChest | 제외 |
|---|---|---|---|---|
| 흉막 이상 | Pleural effusion, Pleural thickening | Pleural Effusion, Pleural Other | 14절 pleural + costophrenic angle blunting (= pleural_cpa) | pericardial effusion, fissure thickening, pneumothorax |
| 심장 음영 확대 | Cardiomegaly | Cardiomegaly (Enlarged Cardiomediastinum은 종격동 포함이라 제외) | cardiomegaly (+ pericardial effusion 14장은 무시 가능한 수라 추가 안 함) | heart valve, dextrocardia, surgery heart |
| 폐부종 | (해당 label 없음) | Edema | pulmonary edema, kerley lines | heart insufficiency (진단명) |

- 확인 결과: 현재 사전 고정 대응표가 이미 통일 개념과 같음. 유일한 차이 = PadChest 흉막에 CPA blunting 포함 여부 → 14절 민감도 label pleural_cpa가 곧 통일 label
- 따라서 통일 label 분석 = PadChest에서 pleural 대신 pleural_cpa로 같은 평가(판별력, 판독 방식 보정, 13절)를 한 결과. 범주 범위 차이의 영향 = 두 결과의 차이로 보고
- 출력 이름 표시: 모델 출력(pleural, cardiac, edema)은 통일 개념, 결과 표시는 병원별 이름표로 바꿈 (위 표의 병원 열). 성능과 무관한 표시 기능

## 16. 순서형(ordinal) 확신도 출력 (2026-11-24, 학습 전·14절 PadChest 평가 결과를 보기 전에 고정. scripts/sweep_ord.py)

동기: 외부 오보정의 주원인은 판독 방식 차이(②, 11-4). 학습 label은 "판독의 1명이 확신도 ≥3"인데 병원마다 양성 기준이 다름.
모델이 확신도 문턱별 확률을 직접 내면, 병원 정보 수준에 맞는 문턱을 정답 없이 고를 수 있음.
- 문턱: ≥3(기존 출력), ≥4, ≥5. ≥1·≥2는 train에서 ≥3과 차이가 9장 이하라 제외
  (train 판독 양성 ≥3/≥4/≥5: 흉막 523/452/345, 심장비대 387/358/301, 폐부종 250/218/115)
- 구조: logit_≥4 = logit_≥3 − softplus(g4(x)), logit_≥5 = logit_≥4 − softplus(g5(x)). g = finding별 특징의 선형 헤드
  → 문턱이 높을수록 확률이 항상 낮음(단조), 간격이 영상마다 달라 퍼짐(slope)도 달라질 수 있음. ≥3 출력은 기존 구조 그대로
- label: ≥4·≥5는 REFLACX 판독 확신도(판독 여러 명이면 과반수). EGD 음성은 모든 문턱 음성, EGD 양성은 확신도를 몰라 ≥4·≥5 loss에서 제외
- loss = 기존 loss + 1 × (≥4·≥5 BCE). 가중치 1은 사전 고정, 조정 없음. checkpoint 선택·나머지 설정은 기존과 동일(val ≥3 loss)
- temperature: 출력마다 내부 val에서 따로
- 조건: ord_fs (hybrid3, fs 시선 λ0.3) / ord_c1 (avg, 시선 없음). seed 0~9
- 문턱 선택 규칙 (외부 정답 사용 안 함)
  - 정보 없음: 판독 방식 k에 대한 문턱 t*(k) = 내부 5명 판독 109장에서 "k명 과반수 label"에 대해 TS 확률의 NLL이 가장 낮은 문턱
    (k = 5: 실제 5명 과반수, k = 3: 5명 중 3명 조합 10개의 과반수 평균, k = 1: 판독 1명씩). 조건·finding별, seed 평균 NLL 기준.
    k는 14절·11-6절과 같은 K_PROTOCOL (VinDr train·새 train·CheXpert val = 3, VinDr test·CheXpert test = 5, PadChest = 1)
  - 유병률 있음: 13절과 같은 같은 병원 쌍에서, 대상 영상 평균 예측 확률이 π_기록에 가장 가까운 문턱 (ord_select_prev),
    + 그 문턱 출력에 13절 prev 보정을 더한 것 (ord_select_prev_ts)
- 판정 (외부 7곳, seed 평균, 쌍·데이터별)
  H5 정보 없음: ord_fs[t*(k)]의 NLL이 ord_fs[≥3]와 기존 fs raw보다 낮은 데이터가 과반수 → "판독 방식에 맞춘 문턱이 정보 없는 병원 보정을 개선".
     |slope−1|, intercept도 함께 보고
  H6 유병률: ord_select_prev(_ts)의 NLL이 13절 prev_reader보다 낮은 쌍이 과반수 → "문턱 선택이 직선 보정보다 나음"
  H7 판별력 유지: ord_fs ≥3 AUROC − fs AUROC의 random-effects 통합 차이 CI 하한 > −0.02 (비열등). ord_fs − ord_c1은 보조 (시선 효과 유지)
- 해석: H5·H6 실패여도 "판독 방식 차이는 확신도 문턱으로 설명되지 않음(합의 과정 자체의 효과)"이라는 결과로 보고

## 앞으로의 순서 (2026-11-22 결정: 현재 결과 최대한 강화 + 다른 구조로 반복(B))
1. 12절 판별력 외부 검증 확장 + 통합 (완료)
2. 13절 정보 수준별 베이지안 재보정 (2026-11-22 사용자 결정: 프로토콜 한계 해결 권장 순서 1)
3. 순서형 출력(판독 확신도 기준별 확률) 학습 → 정보 없는 병원 개선 시도 (권장 순서 2, 재학습)
4. PadChest(세 번째 병원)로 프로토콜·B 최종 확인 (권장 순서 3) — 14절, 평가 중
   - 15절 label 개념 통일 (pleural_cpa로 확인), 16절 순서형 출력 학습 (2026-11-24 고정)
5. CheXlocalize 전문의 분할로 위치 평가
6. B: 두 번째 backbone(예: ConvNeXt)으로 c1_none, fs, fs_shuffled 반복. 설정은 내부 val로만, 세 조건에 같은 탐색 범위.
   외부 데이터는 B 설계에 쓰지 않았으므로 B의 확인 데이터로 사용 가능. 계획은 실행 전에 별도 고정

