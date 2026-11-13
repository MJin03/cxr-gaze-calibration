# 실행 코드

연구 서버에서 실행하는 코드를 둔다.

| 파일 | 용도 |
|---|---|
| [download_egd_cxr.sh](download_egd_cxr.sh) | EGD-CXR 시선·라벨 다운로드 (음성 제외) |
| [make_reflacx_image_list.py](make_reflacx_image_list.py) | REFLACX 영상의 MIMIC-CXR-JPG 경로 목록 생성 (EGD-CXR와 겹치는 영상 제외) |
| [verify_previous_results.py](verify_previous_results.py) | 이전 연구 보고 수치를 보관된 원본 예측값으로 재계산·대조 |

다음에 추가할 코드: DenseNet-121 Baseline 학습(`train_baseline.py`).
