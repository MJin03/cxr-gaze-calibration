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
| [cal_common.py](cal_common.py) | 11·11-2절 공용 설정: smoothing 조건, A 조건(난이도 헤드), 비교, 외부 데이터 목록, B 대상 조건 고정 |
| [predict_external.py](predict_external.py) | 11절: 외부 데이터(chexpert_test / vindr_train / chexpert_valid / vindr_test) 예측, attention 저장. internal_val = B의 기준 분포용 내부 val 예측 |
| [build_vindr_train_cache.py](build_vindr_train_cache.py) | 11절: VinDr train 무작위 3,000장 → 512 PNG, 판독의 3명 과반수 label |
| [evaluate_calibration.py](evaluate_calibration.py) | 11절: calibration slope·NLL·intercept·ECE·Brier·AUROC, 조건 비교 (환자/영상+seed bootstrap). 11-2절 C: label 없는 EM 재보정 |
| [evaluate_uncertainty.py](evaluate_uncertainty.py) | 11-2절 B: attention 엔트로피·확신도·순위 평균 불확실성 점수의 오류 탐지 AUROC, AURC, 자동 문턱 missed |
| [diagnose_underconfidence.py](diagnose_underconfidence.py) | 사후 진단: 외부 확신 부족(slope > 1)의 원인 — 데이터별 slope, VinDr 판독의 비율 label 기준 slope, 일치도별 slope, 앙상블 방식 |
| [reader_recalibration.py](reader_recalibration.py) | 11-6절: 판독 방식 보정. fit = 내부 판독의 5명 영상으로 판독의 간 상관 ρ 추정, eval = 외부 데이터(k명 과반수)에서 보정 전후 비교 |
| [evaluate_discrimination.py](evaluate_discrimination.py) | 12절: 8·9절 조건의 판별력(AUROC, 자동 문턱 missed)을 외부 데이터 여러 곳에서 비교하고 random-effects로 통합 |
| [bayes_recalibration.py](bayes_recalibration.py) | 13절: 정보 수준별(없음/유병률/label n장) 베이지안 재보정, 같은 병원 다른 묶음을 과거 기록으로 쓰는 평가 |
| [download_chexpert_valid.py](download_chexpert_valid.py) | 11절: Redivis(AIMI) CheXpert Plus에서 validation PNG 234장과 label 폴더만 다운로드 (OAuth 로그인) |
| [build_chexpert_cache.py](build_chexpert_cache.py) | 11절: CheXlocalize 묶음의 CheXpert test·validation 정면 영상을 512 PNG(plain·eq)로 변환, 전문의 label 정리(환자 번호 포함), 전처리 선택 |
| [sweep_diff.py](sweep_diff.py) | 11-2절: fs + 난이도 예측 헤드·영상별 logit 축소 (gaze / gaze-shuffled / certainty / free) 실행·요약 |
| [download_egd_cxr.sh](download_egd_cxr.sh) | EGD-CXR 시선·라벨 다운로드 (음성 제외) |
| [make_reflacx_image_list.py](make_reflacx_image_list.py) | REFLACX 영상의 MIMIC-CXR-JPG 경로 목록 생성 (EGD-CXR와 겹치는 영상 제외) |
| [verify_previous_results.py](verify_previous_results.py) | 이전 연구 보고 수치를 보관된 원본 예측값으로 재계산·대조 |
| [build_padchest_cache.py](build_padchest_cache.py) | 14절: PadChest 의사 label 정면 영상 선택(plan) → zip별 다운로드·512 변환·zip 삭제(run, monitor) → labels.csv·전처리 선택·QC(finalize) |
| [sweep_ord.py](sweep_ord.py) | 16절: 순서형 확신도 출력(≥3·≥4·≥5) 학습 (ord_fs, ord_c1 × seed 0~9)과 val 요약 |
| [evaluate_ordinal.py](evaluate_ordinal.py) | 16절 평가: 문턱 선택 t*(k)(내부 5명 판독) → H5 정보 없음, H6 유병률, H7 판별력 유지 |
| [sweep_gazefrac.py](sweep_gazefrac.py) | 17절: 시선 양 학습 곡선 학습 (fs_gf25·50·75 × seed 0~9) |
| [evaluate_gazefrac.py](evaluate_gazefrac.py) | 17절 평가: 시선 비율별 외부 AUROC, 양-반응 기울기(G1), 포화(G2) |
| [count_pleural_words.py](count_pleural_words.py) | 18절: 판독 음성기록의 흉막 세부 소견 단어(흉수·비후·폐첨부·CPA 둔화·plaque) 언급 영상 수 |
| [sweep_nih.py](sweep_nih.py) | 19절: NIH 사전학습 DenseNet 위에서 nih_c1·nih_hyb_none·nih_fs·nih_fs_shuffled 설정 탐색(val) → 선택 → seed 0~9 |
| [evaluate_localization.py](evaluate_localization.py) | 20절: CheXlocalize 전문의 분할·VinDr 박스로 attention 위치 평가 (hit, energy, mIoU, 사람 기준) |
| [evaluate_clinical.py](evaluate_clinical.py) | 21절: 특이도 90%의 민감도·놓친 환자 수/1,000명, 주 비교 Holm 보정 |
| [sweep_gazesource.py](sweep_gazesource.py) | 22절: 시선 출처별 학습 (EGD만 / REFLACX만 시선, seed 0~9) |
| [evaluate_gazesource.py](evaluate_gazesource.py) | 22절 평가: 출처별 시선 효과(R1), 같은 비율 무작위 시선과 비교(R2) |
| [count_more_findings.py](count_more_findings.py) | 24절: 소견 확장 후보의 내부 양성 수·시선 단어 coverage·외부 양성 수 |

분석 계획: [../ANALYSIS_PLAN.md](../ANALYSIS_PLAN.md) (test 평가 전 고정)
