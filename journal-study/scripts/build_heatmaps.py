"""4-2단계: train·val 영상의 시선 지도(512×512)와 셔플 대조군 매핑을 만든다.

시선 지도 만드는 법
    1. 영상 밖의 고정점은 버린다
    2. 고정점 위치를 512×512 좌표로 바꾼다 (영상을 512×512로 바로 줄이는 것과 같은 변환)
    3. 머문 시간(초)을 가중치로 점을 찍는다
    4. Gaussian blur (sigma 7픽셀)로 부드럽게 퍼뜨린다
    5. 전체 합이 1이 되게 정규화한다 → 데이터셋 간 고정점 수 차이를 없앤다
    6. 시선이 2개인 영상(EGD+REFLACX 중복)은 각각 만든 지도를 평균한다

test 영상은 만들지 않는다 (평가 때는 영상만 입력).

셔플 대조군
    train 영상마다 "다른 환자"의 시선 지도를 하나씩 일대일로 배정한다 (같은 지도를 두 번 쓰지 않음).

출력
    ~/X-ray/data/processed/heatmaps_512_s7/{dicom_id}.npy   (float16, 합 1)
    ~/X-ray/data/processed/heatmap_index.csv                 (영상별 시선 수, 사용한 고정점 수, 안쪽 비율)
    ~/X-ray/data/processed/shuffle_map_seed{SEED}.csv        (dicom_id → donor_dicom_id)
    ~/X-ray/data/processed/qc/heatmap_*.png                  (확인용 그림)

사용법  python build_heatmaps.py [--seed 42] [--sigma 7] [--size 512]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter

PROC = Path.home() / "X-ray/data/processed"


def load_fixations(row, egd_groups) -> pd.DataFrame:
    """판독 하나의 고정점을 (x, y, w) 표로 돌려준다. 좌표는 원본 영상 픽셀 기준."""
    if row.dataset == "egd":
        f = egd_groups.get_group(row.dicom_id)
        return pd.DataFrame({"x": f["X_ORIGINAL"].values, "y": f["Y_ORIGINAL"].values,
                             "w": f["FPOGD"].values})
    f = pd.read_csv(row.gaze_file)
    return pd.DataFrame({"x": f["x_position"].values, "y": f["y_position"].values,
                         "w": (f["timestamp_end_fixation"] - f["timestamp_start_fixation"]).values})


def make_heatmap(fix: pd.DataFrame, w: int, h: int, size: int, sigma: float):
    inside = (fix.x >= 0) & (fix.x < w) & (fix.y >= 0) & (fix.y < h)
    f = fix[inside]
    hm = np.zeros((size, size), dtype=np.float64)
    if len(f) == 0:
        return None, 0, float(inside.mean()) if len(fix) else 0.0
    weight = np.clip(f.w.values.astype(float), 0, None)
    if weight.sum() <= 0:
        weight = np.ones(len(f))
    cols = np.clip((f.x.values * size / w).astype(int), 0, size - 1)
    rows = np.clip((f.y.values * size / h).astype(int), 0, size - 1)
    np.add.at(hm, (rows, cols), weight)
    hm = gaussian_filter(hm, sigma=sigma)
    return hm / hm.sum(), len(f), float(inside.mean())


def shuffle_map(train: pd.DataFrame, seed: int) -> pd.DataFrame:
    """일대일 셔플: 모든 영상이 다른 환자의 지도를 정확히 하나씩 받는다."""
    rng = np.random.default_rng(seed)
    ids = train.dicom_id.values
    subj = train.subject_id.values
    for _ in range(1000):
        perm = rng.permutation(len(ids))
        bad = np.where(subj[perm] == subj)[0]
        for i in bad:  # 같은 환자가 걸리면 다른 위치와 자리를 바꾼다
            for j in rng.permutation(len(ids)):
                if subj[perm[j]] != subj[i] and subj[perm[i]] != subj[j]:
                    perm[i], perm[j] = perm[j], perm[i]
                    break
        if (subj[perm] != subj).all():
            return pd.DataFrame({"dicom_id": ids, "donor_dicom_id": ids[perm]})
    raise RuntimeError("셔플 매핑을 만들지 못했습니다")


def save_qc(hm_dir: Path, index: pd.DataFrame, imgs: pd.DataFrame, size: int) -> None:
    qc = PROC / "qc"
    qc.mkdir(exist_ok=True)
    picks = pd.concat([index[index.datasets == "egd"].head(3), index[index.datasets == "reflacx"].head(3)])
    for k, r in enumerate(picks.itertuples()):
        path = imgs.loc[imgs.dicom_id == r.dicom_id, "image_path"].iloc[0]
        im = Image.open(path).convert("L").resize((size, size))
        hm = np.load(hm_dir / f"{r.dicom_id}.npy").astype(float)
        plt.figure(figsize=(5, 5))
        plt.imshow(im, cmap="gray")
        plt.imshow(hm, cmap="jet", alpha=0.4)
        plt.title(f"{r.datasets} {r.dicom_id[:12]} | fix {r.n_fix}", fontsize=9)
        plt.axis("off")
        plt.savefig(qc / f"heatmap_{k + 1}_{r.datasets}.png", dpi=80, bbox_inches="tight")
        plt.close()
    train_ids = index.loc[index.split == "train", "dicom_id"]
    mean = np.mean([np.load(hm_dir / f"{d}.npy").astype(float) for d in train_ids], axis=0)
    plt.figure(figsize=(5, 5))
    plt.imshow(mean, cmap="jet")
    plt.title("mean gaze heatmap (train)", fontsize=9)
    plt.axis("off")
    plt.savefig(qc / "heatmap_mean_train.png", dpi=80, bbox_inches="tight")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sigma", type=float, default=7)
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    reads = pd.read_csv(PROC / "cohort_readings.csv")
    split = pd.read_csv(PROC / f"split_seed{args.seed}.csv")
    imgs = pd.read_csv(PROC / "cohort_images.csv").merge(split[["dicom_id", "split"]], on="dicom_id")
    target = imgs[imgs.split.isin(["train", "val"])]
    hm_dir = PROC / f"heatmaps_{args.size}_s{args.sigma:g}"
    hm_dir.mkdir(parents=True, exist_ok=True)

    egd_file = reads.loc[reads.dataset == "egd", "gaze_file"].iloc[0]
    egd_groups = pd.read_csv(egd_file, usecols=["DICOM_ID", "X_ORIGINAL", "Y_ORIGINAL", "FPOGD"]).groupby("DICOM_ID")
    usable = reads[reads.gaze_ok & reads.dicom_id.isin(target.dicom_id)]

    rows = []
    for i, img in enumerate(target.itertuples(), 1):
        w, h = Image.open(img.image_path).size
        maps, n_fix, inside = [], 0, []
        for r in usable[usable.dicom_id == img.dicom_id].itertuples():
            hm, n, ins = make_heatmap(load_fixations(r, egd_groups), w, h, args.size, args.sigma)
            inside.append(ins)
            if hm is not None:
                maps.append(hm); n_fix += n
        if maps:
            avg = np.mean(maps, axis=0)
            np.save(hm_dir / f"{img.dicom_id}.npy", (avg / avg.sum()).astype(np.float16))
        rows.append({"dicom_id": img.dicom_id, "subject_id": img.subject_id, "split": img.split,
                     "datasets": img.datasets, "n_gaze": len(maps), "n_fix": n_fix,
                     "inside_ratio": np.mean(inside) if inside else np.nan, "width": w, "height": h})
        if i % 500 == 0:
            print(f"  {i}/{len(target)}장 처리")

    index = pd.DataFrame(rows)
    index.to_csv(PROC / "heatmap_index.csv", index=False)
    missing = index[index.n_gaze == 0]
    if len(missing):
        print(f"[주의] 시선 지도가 없는 train·val 영상 {len(missing)}장:", missing.dicom_id.head().tolist())

    train = index[(index.split == "train") & (index.n_gaze > 0)]
    smap = shuffle_map(train, args.seed)
    smap.to_csv(PROC / f"shuffle_map_seed{args.seed}.csv", index=False)
    same_patient = (smap.merge(train[["dicom_id", "subject_id"]], on="dicom_id")
                    .merge(train[["dicom_id", "subject_id"]].rename(columns={"dicom_id": "donor_dicom_id", "subject_id": "donor_subject"}),
                           on="donor_dicom_id"))
    save_qc(hm_dir, index, imgs, args.size)

    print("\n=== 시선 지도 요약")
    print(index.groupby("split").agg(영상=("dicom_id", "size"), 지도있음=("n_gaze", lambda s: int((s > 0).sum())),
                                     시선2개=("n_gaze", lambda s: int((s == 2).sum())),
                                     고정점_중앙값=("n_fix", "median"), 안쪽비율=("inside_ratio", "mean")).to_string())
    print(f"\n가로가 더 긴 영상: {int((index.width > index.height).sum())}장")
    print(f"셔플 매핑: {len(smap)}장, 같은 환자 배정 {int((same_patient.subject_id == same_patient.donor_subject).sum())}건, "
          f"donor 중복 {int(smap.donor_dicom_id.duplicated().sum())}건")
    print(f"저장: {hm_dir}\n      {PROC / 'heatmap_index.csv'}\n      {PROC / f'shuffle_map_seed{args.seed}.csv'}\n      {PROC / 'qc'}")


if __name__ == "__main__":
    main()
