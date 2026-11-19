# 실행 코드

연구 서버에서 실행하는 코드를 둔다.

| 파일 | 용도 |
|---|---|
| [build_cohort.py](build_cohort.py) | 1단계: EGD-CXR + REFLACX 통합 목록(판독 단위·영상 단위 CSV) 생성 |
| [build_image_cache.py](build_image_cache.py) | 5-0단계: 영상을 512×512 흑백 PNG로 미리 줄여 저장 (heatmap과 같은 resize) |
| [sweep.py](sweep.py) | 개선 단계 A·B·C·C3 실행/요약 (val로 설정 선택, ANALYSIS_PLAN 8절) |
| [train.py](train.py) | 5·6단계 공용 학습: --task, --gaze none/real/shuffled/fs/fs-shuffled/ellipse, --backbone imagenet/chexpert, --schedule, --pool avg/attn/attn3/hybrid3, 결과는 ~/X-ray/runs/ |
| [evaluate.py](evaluate.py) | 7단계: test 평가 (../ANALYSIS_PLAN.md 기준, referral 25% 주 지표, 환자 단위 bootstrap). 결과는 ~/X-ray/results/ |
| [build_labels.py](build_labels.py) | 2단계: 판독 라벨을 영상 단위 양성/음성으로 변환 (확신도 기준 3·4, 다판독 과반수) |
| [build_split.py](build_split.py) | 3단계: 환자 단위 70/15/15 분할 (5명 판독 영상은 test 고정, 소견·데이터셋·AP 층화) |
| [build_heatmaps.py](build_heatmaps.py) | 4-2단계: train·val 시선 지도(512×512, sigma 7, 합 1)와 일대일 셔플 매핑 생성 |
| [build_fs_heatmaps.py](build_fs_heatmaps.py) | C3단계: finding별 시선 지도(말하기 W초 전~말 끝, W = 1.5·3·5)와 병변 타원 지도(3×128×128) 생성, coverage 출력 |
| [compare_edema_words.py](compare_edema_words.py) | C3단계 보완: 폐부종 단어 세트 v1·v2·v3의 train coverage와 타원 겹침 비교 (학습 전 결정용) |
| [check_fs_maps.py](check_fs_maps.py) | C3단계 확인: finding별 시선 지도가 병변 타원 안에 더 많이 떨어지는지(전체 시선 대비), 평균 지도 무게중심 |
| [check_gaze_alignment.py](check_gaze_alignment.py) | 4-1단계: 시선 좌표가 JPG 픽셀 좌표와 맞는지 확인 (영상 안쪽 비율, 겹침 그림 6장) |
| [vindr_common.py](vindr_common.py) | D단계 공용 설정: VinDr 평가 조건·비교·비열등 마진 고정 (ANALYSIS_PLAN 9절) |
| [build_vindr_cache.py](build_vindr_cache.py) | D단계 1: VinDr test DICOM → 512 PNG (plain/eq), label·박스 정리, 밝기 분포로 전처리 선택 |
| [predict_vindr.py](predict_vindr.py) | D단계 2: 학습된 모델로 VinDr 예측 (재학습 없음), hybrid3는 attention 지도 저장 |
| [evaluate_vindr.py](evaluate_vindr.py) | D단계 3: VinDr 평가 (seed 평균·앙상블, 영상+seed bootstrap, 비열등, pointing game) |
| [evaluate_rolling.py](evaluate_rolling.py) | 10절(사후): 순위 기반 자동 문턱(최근 W장 확신도 하위 25%)으로 selective prediction 재평가, 내부·VinDr |
| [compare_stageE.py](compare_stageE.py) | E단계: IWAIT 보고값·3분류 재현·사전 고정·개선 설정을 내부/VinDr, 고정/자동 문턱으로 한 표에 정리 (저장 결과만 사용) |
| [build_gaze_difficulty.py](build_gaze_difficulty.py) | 11절: finding별 시선 창에서 영상별 난이도 d(머문 시간·다시 말한 횟수·퍼짐)와 판독의 확신도 난이도 u 계산 |
| [sweep_smooth.py](sweep_smooth.py) | 11절: fs 위에 난이도 가중 label smoothing (gaze / gaze-shuffled / certainty / uniform) 실행·요약 |
| [cal_common.py](cal_common.py) | 11절 공용 설정: smoothing 조건·비교·외부 데이터 목록 고정 |
| [predict_external.py](predict_external.py) | 11절: 외부 데이터(vindr_test / vindr_train / chexpert_valid) 예측 (재학습 없음) |
| [build_vindr_train_cache.py](build_vindr_train_cache.py) | 11절: VinDr train 무작위 3,000장 → 512 PNG, 판독의 3명 과반수 label |
| [evaluate_calibration.py](evaluate_calibration.py) | 11절: calibration slope·NLL·intercept·ECE·Brier·AUROC, 조건 비교 (환자/영상+seed bootstrap) |
| [download_egd_cxr.sh](download_egd_cxr.sh) | EGD-CXR 시선·라벨 다운로드 (음성 제외) |
| [make_reflacx_image_list.py](make_reflacx_image_list.py) | REFLACX 영상의 MIMIC-CXR-JPG 경로 목록 생성 (EGD-CXR와 겹치는 영상 제외) |
| [verify_previous_results.py](verify_previous_results.py) | 이전 연구 보고 수치를 보관된 원본 예측값으로 재계산·대조 |

분석 계획: [../ANALYSIS_PLAN.md](../ANALYSIS_PLAN.md) (test 평가 전 고정)
