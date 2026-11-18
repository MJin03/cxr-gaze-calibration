"""D단계 2: 학습된 모델로 VinDr-CXR test를 예측한다 (재학습 없음).

조건과 seed는 vindr_common.CONDITIONS (ANALYSIS_PLAN.md 9절)에 고정되어 있다.
영상은 build_vindr_cache.py가 고른 전처리(preprocess_choice.json)를 쓴다.

출력  각 run 폴더에
    preds_vindr.csv   image_id, logit_pleural, logit_cardiac, logit_edema (temperature 적용 전)
    att_vindr.npy     hybrid3 조건만: finding별 attention logit (N×3×16×16, float16, preds_vindr.csv 순서)

사용법  python predict_vindr.py [--batch-size 32] [--overwrite]
"""
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

from vindr_common import CONDITIONS, FINDINGS_MODEL, HYBRID3, VINDR_DIR, run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--variant", default="", help="비우면 preprocess_choice.json의 선택")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow-missing", action="store_true", help="동작 확인용: 없는 run은 건너뜀")
    args = ap.parse_args()

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from train import Net  # noqa: E402

    variant = args.variant or json.load(open(VINDR_DIR / "preprocess_choice.json"))["choice"]
    labels = pd.read_csv(VINDR_DIR / "labels.csv")
    ids = labels.image_id.tolist()
    img_dir = VINDR_DIR / f"images_512_{variant}"
    print(f"VinDr test {len(ids)}장, 전처리 {variant}")

    jobs = [(c, s) for c in CONDITIONS for s in CONDITIONS[c][2]]
    missing = [run_dir(c, s) for c, s in jobs if not (run_dir(c, s) / "best.pt").exists()]
    if missing and not args.allow_missing:
        raise SystemExit(f"학습된 run이 없습니다 ({len(missing)}개). 예: {missing[0]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    cache = {}
    t0 = time.time()
    for k, (c, s) in enumerate(jobs, 1):
        d = run_dir(c, s)
        if not (d / "best.pt").exists():
            print(f"[{k}/{len(jobs)}] 없음, 건너뜀: {d.name}")
            continue
        if (d / "preds_vindr.csv").exists() and not args.overwrite:
            print(f"[{k}/{len(jobs)}] 이미 있음: {d.name}")
            continue
        cfg = json.load(open(d / "config.json"))
        size = int(cfg.get("img_size", 512))
        if size not in cache:  # 영상은 한 번만 읽어 둔다
            src = img_dir if size == 512 else None
            arr = []
            for i in ids:
                im = Image.open(src / f"{i}.png").convert("L") if src else \
                    Image.open(img_dir / f"{i}.png").convert("L").resize((size, size), Image.BILINEAR)
                arr.append(np.asarray(im))
            cache[size] = torch.from_numpy(np.stack(arr))
        x_all = cache[size]
        model = Net(len(FINDINGS_MODEL), cfg.get("backbone", "imagenet"), False, cfg.get("pool", "avg")).to(device)
        model.load_state_dict(torch.load(d / "best.pt", map_location=device)["model"])
        model.eval()
        logits, atts = [], []
        with torch.no_grad():
            for b in range(0, len(ids), args.batch_size):
                x = (x_all[b:b + args.batch_size].float() / 255.0)[:, None].to(device)
                with torch.autocast(device.type, enabled=amp):
                    lg, att = model(x)
                logits.append(lg.float().cpu())
                if c in HYBRID3:
                    atts.append(att.float().cpu())
        lg = torch.cat(logits).numpy()
        out = pd.DataFrame({"image_id": ids, **{f"logit_{f}": lg[:, j] for j, f in enumerate(FINDINGS_MODEL)}})
        out.to_csv(d / "preds_vindr.csv", index=False)
        if c in HYBRID3:
            np.save(d / "att_vindr.npy", torch.cat(atts).numpy().astype(np.float16))
        print(f"[{k}/{len(jobs)}] {c} seed {s} 완료 ({time.time() - t0:.0f}s 경과)", flush=True)
    print("예측 완료")


if __name__ == "__main__":
    main()
