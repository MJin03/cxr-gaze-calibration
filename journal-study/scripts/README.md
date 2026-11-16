# 실행 코드

연구 서버에서 실행하는 코드를 둔다.

| 파일 | 용도 |
|---|---|
| [build_cohort.py](build_cohort.py) | 1단계: EGD-CXR + REFLACX 통합 목록(판독 단위·영상 단위 CSV) 생성 |
| [build_image_cache.py](build_image_cache.py) | 5-0단계: 영상을 512×512 흑백 PNG로 미리 줄여 저장 (heatmap과 같은 resize) |
| [train.py](train.py) | 5·6단계 공용 학습: --task findings/class3, --gaze none/real/shuffled, 결과는 ~/X-ray/runs/ |
| [evaluate.py](evaluate.py) | 7단계: test 평가 (../ANALYSIS_PLAN.md 기준, referral 25% 주 지표, 환자 단위 bootstrap). 결과는 ~/X-ray/results/ |
| [build_labels.py](build_labels.py) | 2단계: 판독 라벨을 영상 단위 양성/음성으로 변환 (확신도 기준 3·4, 다판독 과반수) |
| [build_split.py](build_split.py) | 3단계: 환자 단위 70/15/15 분할 (5명 판독 영상은 test 고정, 소견·데이터셋·AP 층화) |
| [build_heatmaps.py](build_heatmaps.py) | 4-2단계: train·val 시선 지도(512×512, sigma 7, 합 1)와 일대일 셔플 매핑 생성 |
| [check_gaze_alignment.py](check_gaze_alignment.py) | 4-1단계: 시선 좌표가 JPG 픽셀 좌표와 맞는지 확인 (영상 안쪽 비율, 겹침 그림 6장) |
| [download_egd_cxr.sh](download_egd_cxr.sh) | EGD-CXR 시선·라벨 다운로드 (음성 제외) |
| [make_reflacx_image_list.py](make_reflacx_image_list.py) | REFLACX 영상의 MIMIC-CXR-JPG 경로 목록 생성 (EGD-CXR와 겹치는 영상 제외) |
| [verify_previous_results.py](verify_previous_results.py) | 이전 연구 보고 수치를 보관된 원본 예측값으로 재계산·대조 |

분석 계획: [../ANALYSIS_PLAN.md](../ANALYSIS_PLAN.md) (test 평가 전 고정)
