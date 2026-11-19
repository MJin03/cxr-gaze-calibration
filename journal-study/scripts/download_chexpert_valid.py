"""11절: Stanford AIMI(Redivis)의 CheXpert Plus에서 validation PNG 234장과 label 폴더만 내려받는다.

전체 데이터(6.8TB)는 받지 않는다. 처음 실행하면 Redivis 로그인(OAuth) 안내가 터미널에 나온다:
나온 주소를 브라우저에서 열고 로그인·허용하면 이어서 진행된다. (API 토큰은 쓰지 않는다)

출력  ~/X-ray/data/raw/chexpert/PNG_valid/, ~/X-ray/data/raw/chexpert/CheXpert_Labels/

사용법  pip install --upgrade redivis   (처음 한 번)
        python download_chexpert_valid.py
"""
from pathlib import Path

import redivis

OUT = Path.home() / "X-ray/data/raw/chexpert"
TABLES = {"png_valid": "PNG_valid", "chexpert_labels": "CheXpert_Labels"}


def main():
    ds = redivis.organization("AIMI").dataset("chexpert_plus:v1_0")
    for ref, folder in TABLES.items():
        dst = OUT / folder
        dst.mkdir(parents=True, exist_ok=True)
        print(f"{ref} → {dst}", flush=True)
        table = ds.table(ref)
        if hasattr(table, "to_directory"):
            table.to_directory().download(str(dst), overwrite=True)
        else:  # 이전 버전 redivis
            table.download_files(str(dst), overwrite=True)
    for folder in TABLES.values():
        files = [p for p in (OUT / folder).rglob("*") if p.is_file()]
        print(f"{folder}: 파일 {len(files)}개, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB")
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
