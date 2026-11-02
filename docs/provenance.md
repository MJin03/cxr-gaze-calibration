# 출처 및 복구 기록

## 보관 기준

최종 기준본은 `code/EGD_CXR_Gaze_Calibration_CONVERSATION_RESTORED_ARCHIVAL.py`이다. 사용자가 최종 제공한 8,189줄의 텍스트를 바이트 변경 없이 복사했다.

- SHA256: `5e8bba5777a465e80e618fc54efb07dcbff618f30b0cdfd2db259073b9a25081`
- 보관 정리일: 2026-11-02
- 성격: 초기 노트북, 대화의 코드·수정 기록과 보고 결과를 바탕으로 한 conversation-based reconstruction
- 서버에서 직접 회수한 최종 스크립트 또는 당시 Git 이력은 아니다.

## 초기 노트북

`code/original/EGD_CXR_Gaze_Celibration.ipynb`는 로컬에서 확보한 초기 Colab 노트북의 업로드용 사본이다. 모든 셀의 코드와 마크다운은 보존했다. 데이터 영상과 개별 사례 출력의 게시를 피하기 위해 실행 출력 68개와 execution count는 제거했다. 원본은 로컬에 별도로 유지한다. 따라서 이 노트북 사본 전체가 원본과 바이트 단위로 동일한 것은 아니다.

복구 스크립트 내부의 EXACT EXPORT 문구는 그 스크립트 PART A의 출처를 설명하며, 저장소 노트북 사본의 출력 보존 여부를 뜻하지 않는다.

## 확인된 불확실성

- Threshold: 복구 코드는 `np.quantile(confidence, 1-target_coverage)`를 유지한다. 별도 대화에는 `ceil(n*coverage)`에 따라 confidence 내림차순의 순위값을 선택하는 구현도 있다. 당시 최종 실행 방식은 미확정이다. 두 구현 모두 100% coverage에서 `-np.inf`를 사용하도록 수정된 기록이 있다.
- Lesion: mask union 생성 함수는 있으나 실제 metadata 입력과 `save_ext_ils_union_masks(...)` 호출 셀은 복구되지 않았다. 임의 호출을 추가하지 않았다.
- 시점: 초록 제출 후 같은 실험 결과를 사용했다는 설명은 사용자 설명과 대화 기록에 근거한다. 서버 Git history로 검증한 사실은 아니다.
- Colab 경로, 중복된 셀 정의 및 과거 수정 이력을 보관 목적상 그대로 유지했다. 통합 파일의 처음부터 끝까지 실행을 보장하지 않는다.

## 결과 출처와 검증 범위

`results/reported/`는 복구본 마지막 ARCHIVAL REPORTED RESULTS의 고정값을 추출한 것이다. 새로 학습하거나 예측해서 얻은 결과가 아니다. 제공된 결과 캡처와 대조했으며, 개별 예측값·데이터·checkpoint를 이용한 재계산은 하지 않았다.

Python 문법, 표 추출, 노트북 셀 소스 보존 및 사본 해시를 확인했다. 독립적인 학습 반복이나 원래 환경에서의 end-to-end 재현은 검증하지 않았다.

대화 원문, 데이터셋, 모델 가중치 및 추천서는 포함하지 않는다.
