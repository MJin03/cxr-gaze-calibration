"""11절: 학습된 모델로 외부 데이터를 예측한다 (재학습 없음).

조건과 seed는 cal_common.CONDITIONS에 고정. 영상은 각 데이터 폴더의 preprocess_choice.json이 고른 512 PNG.
출력  각 run 폴더에 preds_{dataset}.csv (image_id, logit_pleural, logit_cardiac, logit_edema)
      hybrid3 모델은 att_{dataset}.npy (finding별 attention logit, N×3×16×16, float16) — 11-2절 B 분석용

      internal_val: 내부 val 영상 (11-2절 B의 기준 분포용, attention이 있는 조건만)

사용법  python predict_external.py --dataset chexpert_test|vindr_train|chexpert_valid|vindr_test|internal_val [--overwrite]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

from cal_common import CONDITIONS, DATASETS, FINDINGS_MODEL, GF_CONDITIONS, INTERNAL_VAL, ORD_CONDITIONS, PROC, S10, UNCERTAINTY_CONDS, run_dir
from train import Net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS) + [INTERNAL_VAL], required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow-missing", action="store_true", help="동작 확인용")
    ap.add_argument("--conds", nargs="*", help="예측할 조건 (기본: CONDITIONS 전부). 16절: --conds ord_fs ord_c1")
    args = ap.parse_args()
    if args.dataset == INTERNAL_VAL:
        split = pd.read_csv(PROC / "split_seed42.csv")
        ids, img_dir, choice = sorted(split.loc[split.split == "val", "dicom_id"]), PROC / "images_512", "내부"
        conds = UNCERTAINTY_CONDS
    else:
        ds = DATASETS[args.dataset]
        choice = json.load(open(ds["img"] / "preprocess_choice.json"))["choice"]
        img_dir = ds["img"] / f"images_512_{choice}"
        ids = pd.read_csv(ds["labels"]).image_id.tolist()
        conds = list(CONDITIONS)
    if args.conds:
        bad = [c for c in args.conds if c not in CONDITIONS and c not in ORD_CONDITIONS and c not in GF_CONDITIONS]
        if bad:
            raise SystemExit(f"모르는 조건: {bad}")
        conds = args.conds
    print(f"{args.dataset}: {len(ids)}장, 전처리 {choice}")

    jobs = [(c, s) for c in conds for s in S10]
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
            pool = json.load(open(d / "config.json")).get("pool", "avg")
            need_att = pool in ("attn3", "hybrid3") and not (d / f"att_{args.dataset}.npy").exists()
            if pd.read_csv(out, usecols=["image_id"]).image_id.astype(str).tolist() != [str(i) for i in ids]:
                print(f"[{k}/{len(jobs)}] 기존 파일의 영상 목록이 달라 다시 예측: {c} seed {s}")
            elif need_att:
                print(f"[{k}/{len(jobs)}] attention 파일이 없어 다시 예측: {c} seed {s}")
            else:
                print(f"[{k}/{len(jobs)}] 이미 있음: {c} seed {s}")
                continue
        cfg = json.load(open(d / "config.json"))
        size = int(cfg.get("img_size", 512))
        if size not in cache:
            cache[size] = torch.from_numpy(np.stack([np.asarray(
                Image.open(img_dir / f"{i}.png").convert("L").resize((size, size), Image.BILINEAR)
                if size != 512 else Image.open(img_dir / f"{i}.png").convert("L")) for i in ids]))
        ordinal = str(cfg.get("ordinal", "False")) == "True"
        model = Net(len(FINDINGS_MODEL), cfg.get("backbone", "imagenet"), False, cfg.get("pool", "avg"),
                    diff=cfg.get("diff_target", "none") != "none", ordinal=ordinal).to(device)
        res = model.load_state_dict(torch.load(d / "best.pt", map_location=device)["model"], strict=False)
        if res.unexpected_keys or set(res.missing_keys) - {"mean", "std"}:
            raise SystemExit(f"{d.name}: 가중치 키가 맞지 않습니다 {res.missing_keys} {res.unexpected_keys}")
        model.eval()
        lg, atts, ords = [], [], []
        with torch.no_grad():
            for b in range(0, len(ids), args.batch_size):
                x = (cache[size][b:b + args.batch_size].float() / 255.0)[:, None].to(device)
                with torch.autocast(device.type, enabled=amp):
                    o, a = model(x)
                lg.append(o.float().cpu())
                if ordinal:
                    ords.append(model.last_ord.float().cpu())
                if model.multi:
                    atts.append(a.float().cpu())
        lg = torch.cat(lg).numpy()
        if atts:
            np.save(d / f"att_{args.dataset}.npy", torch.cat(atts).numpy().astype(np.float16))
        cols = {f"logit_{f}": lg[:, j] for j, f in enumerate(FINDINGS_MODEL)}
        if ordinal:  # [t4 finding들, t5 finding들] → logit_{f}_t4, logit_{f}_t5
            od = torch.cat(ords).numpy()
            K = len(FINDINGS_MODEL)
            for ti, t in enumerate([4, 5]):
                for j, f in enumerate(FINDINGS_MODEL):
                    cols[f"logit_{f}_t{t}"] = od[:, ti * K + j]
        pd.DataFrame({"image_id": ids, **cols}).to_csv(out, index=False)
        print(f"[{k}/{len(jobs)}] {c} seed {s} 완료 ({time.time() - t0:.0f}s)", flush=True)
    print("예측 완료")


if __name__ == "__main__":
    main()
