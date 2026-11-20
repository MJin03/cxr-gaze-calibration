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
- 코드 수정: C 표의 intercept(EM 후)가 발산하는 계산 문제(citl 뉴턴 단계 제한 없음) 수정. NLL·ECE·Brier와 주 비교에는 영향 없음

