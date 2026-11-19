"""11절: 학습된 모델로 외부 데이터를 예측한다 (재학습 없음).

조건과 seed는 cal_common.CONDITIONS에 고정. 영상은 각 데이터 폴더의 preprocess_choice.json이 고른 512 PNG.
출력  각 run 폴더에 preds_{dataset}.csv (image_id, logit_pleural, logit_cardiac, logit_edema)

사용법  python predict_external.py --dataset vindr_test|vindr_train|chexpert_valid [--overwrite]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

from cal_common import CONDITIONS, DATASETS, FINDINGS_MODEL, S10, run_dir
from train import Net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow-missing", action="store_true", help="동작 확인용")
    args = ap.parse_args()
    ds = DATASETS[args.dataset]
    choice = json.load(open(ds["img"] / "preprocess_choice.json"))["choice"]
    img_dir = ds["img"] / f"images_512_{choice}"
    ids = pd.read_csv(ds["labels"]).image_id.tolist()
    print(f"{args.dataset}: {len(ids)}장, 전처리 {choice}")

    jobs = [(c, s) for c in CONDITIONS for s in S10]
    missing = [run_dir(c, s) for c, s in jobs if not (run_dir(c, s) / "best.pt").exists()]
    if missing and not args.allow_missing:
        raise SystemExit(f"학습된 run이 없습니다 ({len(missing)}개). 예: {missing[0]}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    cache, t0 = {}, time.time()
    for k, (c, s) in enumerate(jobs, 1):
        d = run_dir(c, s)
        out = d / f"preds_{args.dataset}.csv"
        if not (d / "best.pt").exists():
            continue
        if out.exists() and not args.overwrite:
            print(f"[{k}/{len(jobs)}] 이미 있음: {c} seed {s}")
            continue
        cfg = json.load(open(d / "config.json"))
        size = int(cfg.get("img_size", 512))
        if size not in cache:
            cache[size] = torch.from_numpy(np.stack([np.asarray(
                Image.open(img_dir / f"{i}.png").convert("L").resize((size, size), Image.BILINEAR)
                if size != 512 else Image.open(img_dir / f"{i}.png").convert("L")) for i in ids]))
        model = Net(len(FINDINGS_MODEL), cfg.get("backbone", "imagenet"), False, cfg.get("pool", "avg")).to(device)
        res = model.load_state_dict(torch.load(d / "best.pt", map_location=device)["model"], strict=False)
        if res.unexpected_keys or set(res.missing_keys) - {"mean", "std"}:
            raise SystemExit(f"{d.name}: 가중치 키가 맞지 않습니다 {res.missing_keys} {res.unexpected_keys}")
        model.eval()
        lg = []
        with torch.no_grad():
            for b in range(0, len(ids), args.batch_size):
                x = (cache[size][b:b + args.batch_size].float() / 255.0)[:, None].to(device)
                with torch.autocast(device.type, enabled=amp):
                    lg.append(model(x)[0].float().cpu())
        lg = torch.cat(lg).numpy()
        pd.DataFrame({"image_id": ids, **{f"logit_{f}": lg[:, j] for j, f in enumerate(FINDINGS_MODEL)}}).to_csv(out, index=False)
        print(f"[{k}/{len(jobs)}] {c} seed {s} 완료 ({time.time() - t0:.0f}s)", flush=True)
    print("예측 완료")


if __name__ == "__main__":
    main()
