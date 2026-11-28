"""11절: 학습된 모델로 외부 데이터를 예측한다 (재학습 없음).

조건과 seed는 cal_common.CONDITIONS에 고정. 영상은 각 데이터 폴더의 preprocess_choice.json이 고른 512 PNG.
출력  각 run 폴더에 preds_{dataset}.csv (image_id, logit_pleural, logit_cardiac, logit_edema)
      hybrid3 모델은 att_{dataset}.npy (finding별 attention logit, N×3×16×16, float16) — 11-2절 B 분석용

      internal_val: 내부 val 영상 (11-2절 B의 기준 분포용, attention이 있는 조건만)
      25절: *_more 데이터 (--conds more_fs more_c1 more_hyb_none more_fs_shuffled). 출력 이름은 run config의 targets에서
            (logit_atelectasis, logit_consolidation). internal_test_more는 새로 예측하지 않고 학습 때 저장한 preds_test.csv를 옮김

사용법  python predict_external.py --dataset chexpert_test|vindr_train|chexpert_valid|vindr_test|internal_val [--overwrite]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

from cal_common import (CNX_CONDITIONS, CONDITIONS, DATASETS, FINDINGS_MODEL, GF_CONDITIONS, INTERNAL_VAL, MORE_CONDITIONS,
                        NIH_CONDITIONS, ORD_CONDITIONS, PROC, SRC_CONDITIONS, S10, UNCERTAINTY_CONDS, run_dir)
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
        conds, ds = UNCERTAINTY_CONDS, None
    elif DATASETS[args.dataset].get("internal"):  # 25절 내부 test: 학습 때 저장한 예측을 옮김
        ds = DATASETS[args.dataset]
        ids, img_dir, choice = pd.read_csv(ds["labels"]).image_id.tolist(), None, "내부(학습 때 예측)"
        conds = list(MORE_CONDITIONS)
    else:
        ds = DATASETS[args.dataset]
        choice = json.load(open(ds["img"] / "preprocess_choice.json"))["choice"]
        img_dir = ds["img"] / f"images_512_{choice}"
        ids = pd.read_csv(ds["labels"]).image_id.tolist()
        conds = list(CONDITIONS)
    if args.conds:
        bad = [c for c in args.conds if not any(c in d for d in (CONDITIONS, ORD_CONDITIONS, GF_CONDITIONS, NIH_CONDITIONS, SRC_CONDITIONS,
                                                                  CNX_CONDITIONS, MORE_CONDITIONS))]
        if bad:
            raise SystemExit(f"모르는 조건: {bad}")
        conds = args.conds
    more_ds, more_c = args.dataset.endswith("_more"), any(c in MORE_CONDITIONS for c in conds)
    if more_ds != more_c or (more_c and not all(c in MORE_CONDITIONS for c in conds)):
        raise SystemExit("25절: *_more 데이터는 more_* 조건으로만, more_* 조건은 *_more 데이터로만 예측합니다.")
    if more_ds and not ds["findings"]:
        print(f"[건너뜀] {args.dataset}: 평가할 소견 없음 (label 파일 없음 또는 양성 < 30)")
        return
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
            need_att = (pool in ("attn3", "hybrid3") and not (d / f"att_{args.dataset}.npy").exists()
                        and not (ds is not None and ds.get("internal")))
            if pd.read_csv(out, usecols=["image_id"]).image_id.astype(str).tolist() != [str(i) for i in ids]:
                print(f"[{k}/{len(jobs)}] 기존 파일의 영상 목록이 달라 다시 예측: {c} seed {s}")
            elif need_att:
                print(f"[{k}/{len(jobs)}] attention 파일이 없어 다시 예측: {c} seed {s}")
            else:
                print(f"[{k}/{len(jobs)}] 이미 있음: {c} seed {s}")
                continue
        cfg = json.load(open(d / "config.json"))
        targets = cfg.get("targets", [f"{f}_t3" for f in FINDINGS_MODEL])
        if isinstance(targets, str):
            targets = eval(targets)  # noqa: S307 (예전 config 호환)
        names = [t.rsplit("_t", 1)[0] for t in targets]   # 기존 run = FINDINGS_MODEL, 25절 = atelectasis, consolidation
        if ds is not None and ds.get("internal"):
            te = pd.read_csv(d / "preds_test.csv").set_index("dicom_id")
            missing_ids = [i for i in ids if i not in te.index]
            if missing_ids:
                raise SystemExit(f"{d.name}: preds_test.csv에 없는 영상 {len(missing_ids)}장 (예: {missing_ids[0]})")
            te = te.loc[ids]
            pd.DataFrame({"image_id": ids, **{f"logit_{n}": te[f"logit_{n}"].values for n in names}}).to_csv(out, index=False)
            print(f"[{k}/{len(jobs)}] {c} seed {s} 완료 (내부 test 옮김)", flush=True)
            continue
        size = int(cfg.get("img_size", 512))
        if size not in cache:
            cache[size] = torch.from_numpy(np.stack([np.asarray(
                Image.open(img_dir / f"{i}.png").convert("L").resize((size, size), Image.BILINEAR)
                if size != 512 else Image.open(img_dir / f"{i}.png").convert("L")) for i in ids]))
        ordinal = str(cfg.get("ordinal", "False")) == "True"
        model = Net(len(names), cfg.get("backbone", "imagenet"), False, cfg.get("pool", "avg"),
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
        cols = {f"logit_{f}": lg[:, j] for j, f in enumerate(names)}
        if ordinal:  # [t4 finding들, t5 finding들] → logit_{f}_t4, logit_{f}_t5
            od = torch.cat(ords).numpy()
            K = len(names)
            for ti, t in enumerate([4, 5]):
                for j, f in enumerate(names):
                    cols[f"logit_{f}_t{t}"] = od[:, ti * K + j]
        pd.DataFrame({"image_id": ids, **cols}).to_csv(out, index=False)
        print(f"[{k}/{len(jobs)}] {c} seed {s} 완료 ({time.time() - t0:.0f}s)", flush=True)
    print("예측 완료")


if __name__ == "__main__":
    main()
