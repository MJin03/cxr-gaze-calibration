# 실험 방법

이 문서는 복구 코드와 당시 보고 기록을 요약한다. 미복구 단계와 구현 차이는 [출처 기록](provenance.md)을 참고한다.

## 데이터 및 분할

- EGD-CXR과 MIMIC-CXR-JPG를 DICOM ID로 연결
- 총 1,083장: Normal 360, CHF 363, Pneumonia 360
- 환자 단위 클래스 층화 train/validation/test 분할, 목표 비율 70/15/15, seed 42
- 보고된 test 사례 164개: Normal 54, CHF 58, Pneumonia 52
- 데이터 및 split CSV 원본은 이 저장소에 포함하지 않음

## 학습 설정

| 항목 | 보고된 설정 |
|---|---|
| Backbone | ImageNet-1K pretrained DenseNet-121 |
| 입력 | 512 × 512, RGB, ImageNet normalization |
| Optimizer | AdamW |
| Learning rate / weight decay | 각각 1e-4 |
| Batch size / epochs | 8 / 10 |
| Classification loss | Cross-entropy |
| Gaze auxiliary loss | KL divergence, 가중치 0.1 |
| Lesion auxiliary loss | BCEWithLogits + Soft Dice, 가중치 0.1 |
| Gaze heatmap | fixation duration 가중, Gaussian sigma 7 |

## 비교 조건

| 조건 | 설명 |
|---|---|
| Baseline | 영상만 이용한 3-class 분류 |
| Gaze-Original | 해당 영상에 대응하는 시선 heatmap supervision |
| Shuffled-Gaze | 다른 환자의 gaze를 배정한 대응 관계 대조군. 초기 derangement 구현도 기록에 보존 |
| Random-Gaze | 무작위 공간 heatmap 대조군 |
| Lesion-Supervised | MIMIC-CXR-Ext-ILS의 병변 mask union을 이용한 추가 supervision |

## 평가

- Discrimination: Accuracy, macro F1, macro one-vs-rest AUROC
- Calibration: ECE (15 bins), multiclass Brier, NLL
- Temperature scaling: validation logits로 fitting 후 test 적용
- Selective prediction: confidence = 최대 class probability, uncertainty = 1-confidence
- AURC: empirical coverage points에서 retained risk의 평균
- E-AURC: observed AURC - ideal AURC
- 배포 후보: Baseline / Gaze-Original / Lesion-Supervised 중 validation AURC 최소 조건. Shuffled/Random은 대조군이므로 제외
- 운영점: validation에서 결정한 threshold를 test에 고정 적용. 목표 coverage와 실제 test coverage는 다를 수 있음
- Error capture: 이관 대상에 포함된 오류 수 / 전체 오류 수
- Error enrichment: error capture rate / 실제 이관율
- Paired stratified bootstrap: 1,000회, 같은 sample index를 비교 조건에 적용
- 운영점 및 오류 탐지 bootstrap: 복구 코드 기준 2,000회

## 해석 범위

Original Gaze는 Baseline 대비 discrimination을 개선하지 못했다. 영상–시선 대응 관계를 깨뜨린 조건은 더 나쁜 calibration과 관련되었다. Baseline의 25% test referral은 전체 오류의 약 48.1%를 포착했으나, 폐렴의 오류 탐지 AUROC 신뢰구간은 0.5를 포함한다.

CI는 고정된 학습 모델에 대한 test-sample uncertainty를 나타내며 반복 학습의 변동성을 나타내지 않는다. 실제 전문가가 오류를 교정한 reader study 또는 임상적 효과를 검증한 실험은 아니다.
