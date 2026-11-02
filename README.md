# Radiologist Gaze Supervision for Chest X-ray Calibration

Research archive for:
“Radiologist Gaze Supervision for Calibration and Selective
Prediction in Chest X-ray Classification.”

## Overview

This study examines radiologist gaze supervision in chest X-ray
classification, focusing on discrimination, predictive calibration,
and uncertainty-based selective prediction.

## Archive Status

This repository contains the original recovered Colab notebook,
a conversation-based reconstruction of subsequent experiment code,
and historical reported results.

The reconstructed script is not a recovered server-original file.
End-to-end reproduction using the original data and checkpoints
has not been verified.

See [provenance](docs/provenance.md) for details.

## Experimental Design

- Dataset: EGD-CXR linked to MIMIC-CXR-JPG
- Cohort: 1,083 images
- Classes: Normal, CHF, Pneumonia
- Patient-level train/validation/test split
- Backbone: DenseNet-121
- Conditions: Baseline, Original Gaze, Shuffled Gaze,
  Random Gaze, Lesion-Supervised

## Main Findings

- Original gaze supervision did not improve discrimination
  over the image-only baseline.
- Disrupted image–gaze correspondence was associated with
  poorer calibration in the reported experiment.
- At a 25% test referral rate, baseline uncertainty-based referral
  captured approximately 48.1% of model errors.
- Error detection was weaker for Pneumonia.

These findings are based on the reported training run;
independent training-run variability was not evaluated.

## Reported Results

Historical result tables are available in
[results/reported](results/reported/).

## Repository Structure

- `code/`: recovered notebook and reconstructed experiment script
- `results/reported/`: historical reported metrics
- `docs/`: methodology, provenance, and figures

## Data Access

Dataset files are not included.
Obtain access through the respective dataset providers and
follow their data-use requirements.

---

# 흉부 X-ray 예측 보정을 위한 영상의학과 전문의 시선 지도학습

다음 연구의 코드와 결과를 보관하는 저장소입니다.

“Radiologist Gaze Supervision for Calibration and Selective
Prediction in Chest X-ray Classification.”

## 연구 개요

흉부 X-ray 분류에서 영상의학과 전문의의 시선 정보를 학습에 활용하고,
분류 성능, 예측 확률의 보정, 불확실성 기반 선택적 예측을 평가한 연구입니다.

## 기록 및 복구 상태

이 저장소에는 확보된 초기 Colab 노트북, 이후 실험 코드를 대화 기록에
기반해 복구한 파일, 당시 보고된 연구 결과가 포함되어 있습니다.

복구 스크립트는 서버에서 직접 회수한 원본 파일이 아닙니다.
당시 데이터와 체크포인트를 사용해 전체 과정을 다시 실행하고
동일한 결과가 재현되는지는 검증하지 않았습니다.

자세한 출처와 복구 범위는 [복구 기록](docs/provenance.md)을 참고하세요.

## 실험 설계

- 데이터셋: EGD-CXR과 MIMIC-CXR-JPG를 연결하여 사용
- 분석 대상: 흉부 X-ray 1,083장
- 분류 클래스: 정상(Normal), 울혈성 심부전(CHF), 폐렴(Pneumonia)
- 데이터 분할: 환자 단위 학습·검증·테스트 세트 분할
- 기본 모델: DenseNet-121
- 실험 조건:
  - Baseline: 영상만 사용하는 기준 모델
  - Original Gaze: 해당 영상에 대응하는 전문의 시선 정보로 지도학습
  - Shuffled Gaze: 영상과 시선 정보의 대응 관계를 섞은 대조군
  - Random Gaze: 무작위로 생성한 시선 지도를 사용하는 대조군
  - Lesion-Supervised: 병변 마스크를 이용한 지도학습

## 주요 결과

- Original Gaze는 영상만 사용하는 Baseline보다 분류 성능을 개선하지 못했습니다.
- 보고된 실험에서 영상과 시선 정보의 대응 관계를 깨뜨린 조건은
  예측 확률의 보정 성능 저하와 관련되었습니다.
- Baseline의 불확실성을 기준으로 테스트 사례의 25%를 전문가 검토 대상으로
  이관했을 때, 전체 모델 오류의 약 48.1%가 이관 대상에 포함되었습니다.
- 폐렴 클래스에서는 오류 탐지 성능이 상대적으로 낮았습니다.

이 결과는 당시 보고된 학습 실행에 근거합니다.
독립적인 반복 학습에 따른 결과의 변동성은 평가하지 않았습니다.

## 보고된 결과

당시 보고된 결과표는 [results/reported](results/reported/)에서 확인할 수 있습니다.

## 저장소 구성

- `code/`: 확보된 초기 노트북과 대화 기반 실험 코드 복구본
- `results/reported/`: 당시 보고된 평가 지표
- `docs/`: 실험 방법, 복구 기록 및 그림

## 데이터 접근

데이터셋 파일은 이 저장소에 포함되어 있지 않습니다.
각 데이터 제공처에서 접근 권한을 확보하고,
해당 데이터의 이용 조건을 따라야 합니다.
