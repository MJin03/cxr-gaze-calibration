# 당시 보고 결과 / Historical reported results

이 폴더의 값은 복구 스크립트의 `ARCHIVAL_*` 상수에서 추출했다. 현재 환경에서 재학습·재평가한 결과가 아니다. 원래 표에서 반올림된 값은 그 정밀도를 유지한다.

| 파일 | 내용 |
|---|---|
| [classification_calibration_raw.csv](classification_calibration_raw.csv) | 5-arm uncalibrated test 성능 |
| [classification_calibration_temperature_scaled.csv](classification_calibration_temperature_scaled.csv) | validation에서 temperature fitting 후 test 평가 |
| [deployable_aurc_ranking.csv](deployable_aurc_ranking.csv) | 실제 후보 3개 조건의 validation/test AURC |
| [baseline_operating_points.csv](baseline_operating_points.csv) | Baseline 목표 coverage 1.0~0.5의 운영점 |
| [classwise_error_detection.csv](classwise_error_detection.csv) | 클래스별 error-detection AUROC 및 95% CI |
| [bootstrap_selective_summary.json](bootstrap_selective_summary.json) | paired 차이, 오류 탐지, 이관 및 enrichment 요약 |

## 지표 및 저장 규칙

- Macro_F1 / Macro_AUROC: 클래스별 지표의 macro average
- ECE / Brier / NLL / AURC / E-AURC: 낮을수록 좋음
- Error-detection AUROC: uncertainty가 오분류를 구분하는 정도. 질병 분류 AUROC와 다름
- Actual_Coverage: test에서 AI가 처리하는 비율
- Defer_Rate: 전문가 검토 대상으로 이관하는 test 비율
- Selective_Accuracy: AI가 처리하는 사례의 정확도
- Error_Capture_Rate: 이관 대상에 포함된 전체 오류의 비율
- Referral_Error_Precision: 이관 대상 중 오류인 비율
- 비율은 0~1 단위. `-inf`는 모든 사례를 accept하는 threshold
- CSV 빈 값은 정의되지 않은 지표. 예: 이관이 0건일 때 referral precision

## 대표 결과

Baseline은 validation AURC 0.2131로 후보 중 가장 낮았다. 목표 coverage 0.70에 대한 threshold 0.5845를 test에 적용했을 때 실제 coverage 0.75, 이관 41/164건, 오류 포착 26/54건, selective accuracy 0.7724였다. Error enrichment는 약 1.93이었다.

전체 5-arm의 모든 coverage 운영점 표는 이 CSV에 포함하지 않았다. 현재 보존한 운영점 상수는 Baseline에 한정된다. Threshold 구현의 불확실성 등은 [출처 기록](../../docs/provenance.md)에 설명한다.
