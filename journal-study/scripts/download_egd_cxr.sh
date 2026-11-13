#!/usr/bin/env bash
# EGD-CXR(시선 기록, 라벨, 메타데이터)을 연구 서버에 내려받는다.
# 음성 파일(.wav, .mp3)은 학습에 쓰지 않으므로 제외한다.
# X-ray 영상은 포함되지 않으며 MIMIC-CXR-JPG에서 따로 받는다.
#
# 사용법: bash download_egd_cxr.sh <PhysioNet 아이디> [저장 폴더]
# 비밀번호는 실행 중에 입력한다. 명령어나 파일에 적지 않는다.
set -euo pipefail

USER_ID="${1:?PhysioNet 아이디를 첫 번째 인자로 입력하세요}"
DEST="${2:-$HOME/X-ray/data/raw/egd-cxr}"

mkdir -p "$DEST"
cd "$DEST"

wget -r -N -c -np -nH --cut-dirs=3 \
     -R "*.wav,*.mp3" \
     --user "$USER_ID" --ask-password \
     https://physionet.org/files/egd-cxr/1.0.0/

# 받은 파일만 검사한다. 실패한 파일이 있으면 이름을 출력하고 종료한다.
sha256sum -c --ignore-missing --quiet SHA256SUMS.txt
echo "EGD-CXR 다운로드 및 체크섬 확인 완료: $DEST"
