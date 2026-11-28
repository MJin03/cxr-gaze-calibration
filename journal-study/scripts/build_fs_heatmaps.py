"""C3단계: finding별 시선 지도(finding-specific gaze)와 병변 타원(ellipse) 지도를 만든다 (ANALYSIS_PLAN.md 8절 C3).

finding별 시선 지도
    1. 판독 녹취에서 finding 관련 단어를 찾는다 (소문자, 문장부호 제거 후 어근으로 시작하면 일치. 예: effusions, blunted)
         pleural  effusion, pleura, costophrenic, blunt
         cardiac  cardiomegal, cardiac, heart, cardiomediastin
         edema    --edema-set 에 따라 (2026-11-18 추가, train coverage만 보고 결정, 학습 결과와 무관)
                  v1  edema, oedema, congest, vascul, interstitial, kerley
                  v2  v1 + perihilar, hilar, haz(hazy/haziness), cephaliz, overload, fluid
                  v3  v2 + lung(lung/lungs: "lungs are clear"처럼 폐 전체를 보고 하는 말)
       부정 표현("no effusion")도 포함한다
    2. 단어를 말하기 시작하기 W초 전부터 말을 마칠 때까지를 창으로 잡는다 (여러 번 말하면 창을 합친다)
    3. 창과 겹치는 고정점만 쓰고, 창 안에 머문 시간을 가중치로 한다
    4. 나머지는 4-2단계와 같다: 영상 밖 고정점 제외, 512×512 좌표, Gaussian sigma 7, 합 1
    5. 저장 용량을 줄이려고 512 지도를 4×4 칸씩 더해 128×128로 저장한다 (모델은 16×16으로 줄여 쓰므로 영향 없음)
    6. 해당 단어가 없으면 그 finding 채널은 0 (학습 때 그 finding의 gaze loss는 계산하지 않음)
    7. 판독이 2개인 영상은 지도가 있는 판독끼리 채널별로 평균

병변 타원 지도 (REFLACX만, 참고 비교)
    판독의가 그린 타원 중 certainty ≥ 3이고 해당 finding이 표시된 것을 채워서 합 1로 (512에서 만든 뒤 128로 저장)
    pleural = Pleural abnormality (phase 1은 Pleural effusion, Pleural thickening), cardiac = Enlarged cardiac silhouette,
    edema = Pulmonary edema

시간 기준
    REFLACX  fixations.csv의 timestamp_start/end_fixation과 timestamps_transcription.csv가 같은 시계
    EGD      fixations.csv의 FPOGS(시작) + FPOGD(머문 시간)와 transcript.json의 begin/end_time
    → 확인용으로 "단어 창 안에 고정점이 하나라도 있는 비율"과 시선·녹취 끝 시각을 출력한다

출력 (채널 순서: pleural, cardiac, edema / float16 / 3×128×128)
    ~/X-ray/data/processed/heatmaps_fs_w{W}_128/{dicom_id}.npy   W = 1.5, 3, 5
    ~/X-ray/data/processed/heatmaps_ellipse_128/{dicom_id}.npy   (타원이 하나라도 있는 영상만)
    ~/X-ray/data/processed/fs_heatmap_index.csv                  영상별 finding별 지도 유무
    ~/X-ray/data/processed/fs_mentions.csv                       판독별 단어 수·창 안 고정점 수
    ~/X-ray/data/processed/fs_build_config.json                  사용한 단어 세트·창 길이
    ~/X-ray/data/processed/qc/fs_*.png                           확인용 그림

25절 소견 확장 (2026-11-28 추가): --finding-set more
    소견 atelectasis(어근 atelecta), consolidation(어근 consolid) 2채널, 창 3초만, REFLACX 영상만 (이 소견 label이 REFLACX에만 있음)
    기존 파일을 덮어쓰지 않도록 출력 이름이 다르다
        heatmaps_fs2_w3_128/{dicom_id}.npy (float16, 2×128×128), fs2_heatmap_index.csv, fs2_mentions.csv,
        fs2_build_config.json, qc/fs2_*.png. 병변 타원 지도는 만들지 않음
    label은 build_more_labels.py가 만든 more_labels.csv (atelectasis_t3, consolidation_t3)

사용법  python build_fs_heatmaps.py [--windows 1.5 3 5] [--edema-set v1|v2|v3] [--max-images N]
        python build_fs_heatmaps.py --finding-set more --windows 3
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter

PROC = Path.home() / "X-ray/data/processed"
RAW = Path.home() / "X-ray/data/raw"
FINDINGS = ["pleural", "cardiac", "edema"]
ROOTS = {
    "pleural": ["effusion", "pleura", "costophrenic", "blunt"],
    "cardiac": ["cardiomegal", "cardiac", "heart", "cardiomediastin"],
    "edema": ["edema", "oedema", "congest", "vascul", "interstitial", "kerley"],
}
EDEMA_SETS = {
    "v1": ["edema", "oedema", "congest", "vascul", "interstitial", "kerley"],
    "v2": ["edema", "oedema", "congest", "vascul", "interstitial", "kerley",
           "perihilar", "hilar", "haz", "cephaliz", "overload", "fluid"],
}
EDEMA_SETS["v3"] = EDEMA_SETS["v2"] + ["lung"]
ELLIPSE_COLS = {
    "pleural": ["Pleural abnormality", "Pleural effusion", "Pleural thickening"],
    "cardiac": ["Enlarged cardiac silhouette"],
    "edema": ["Pulmonary edema"],
}
SIZE, OUT_SIZE, SIGMA = 512, 128, 7.0
# 25절 소견 확장 (REFLACX label: Atelectasis, Consolidation)
MORE_FINDINGS = ["atelectasis", "consolidation"]
MORE_ROOTS = {"atelectasis": ["atelecta"], "consolidation": ["consolid"]}
PREFIX = "fs"   # 출력 이름 앞부분 (more = fs2)


# ------------------------------------------------------------ 읽기
def norm_word(w: str) -> str:
    return re.sub(r"[^a-z]", "", str(w).lower())


def load_words(row) -> pd.DataFrame:
    """판독 하나의 단어 표 (word, start, end). 없으면 빈 표."""
    if row.dataset == "egd":
        p = RAW / "egd-cxr/audio_segmentation_transcripts" / row.dicom_id / "transcript.json"
        if not p.exists():
            return pd.DataFrame(columns=["word", "start", "end"])
        items = json.load(open(p)).get("time_stamped_text", [])
        rows = [(tok, it["begin_time"], it["end_time"]) for it in items for tok in str(it["phrase"]).split()]
    else:
        p = Path(row.gaze_file).parent / "timestamps_transcription.csv"
        if not p.exists():
            return pd.DataFrame(columns=["word", "start", "end"])
        t = pd.read_csv(p)
        rows = list(zip(t["word"], t["timestamp_start_word"], t["timestamp_end_word"]))
    df = pd.DataFrame(rows, columns=["word", "start", "end"])
    df["word"] = df["word"].map(norm_word)
    return df


def load_fixations(row, egd_groups) -> pd.DataFrame:
    """(x, y, start, end) 표. 좌표는 원본 영상 픽셀, 시간은 초."""
    if row.dataset == "egd":
        f = egd_groups.get_group(row.dicom_id)
        s = f["FPOGS"].values.astype(float)
        return pd.DataFrame({"x": f["X_ORIGINAL"].values, "y": f["Y_ORIGINAL"].values,
                             "start": s, "end": s + f["FPOGD"].values.astype(float)})
    f = pd.read_csv(row.gaze_file)
    return pd.DataFrame({"x": f["x_position"].values, "y": f["y_position"].values,
                         "start": f["timestamp_start_fixation"].values.astype(float),
                         "end": f["timestamp_end_fixation"].values.astype(float)})


def mention_windows(words: pd.DataFrame, finding: str, w: float, roots=None) -> list:
    roots = roots or ROOTS[finding]
    hit = words["word"].map(lambda x: any(x.startswith(r) for r in roots))
    wins = sorted((max(0.0, s - w), e) for s, e in zip(words.loc[hit, "start"], words.loc[hit, "end"]))
    merged = []
    for s, e in wins:  # 겹치는 창은 합친다
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def dwell_in_windows(fix: pd.DataFrame, wins: list) -> np.ndarray:
    """고정점마다 창 안에 머문 시간(초)."""
    d = np.zeros(len(fix))
    for s, e in wins:
        d += np.clip(np.minimum(fix["end"].values, e) - np.maximum(fix["start"].values, s), 0, None)
    return d


# ------------------------------------------------------------ 지도
def pool_to_out(hm: np.ndarray) -> np.ndarray:
    k = SIZE // OUT_SIZE
    return hm.reshape(OUT_SIZE, k, OUT_SIZE, k).sum(axis=(1, 3))


def gaze_map(fix: pd.DataFrame, weight: np.ndarray, w: int, h: int):
    inside = (fix.x >= 0) & (fix.x < w) & (fix.y >= 0) & (fix.y < h) & (weight > 0)
    if not inside.any():
        return None
    f, wt = fix[inside], weight[inside.values]
    hm = np.zeros((SIZE, SIZE))
    cols = np.clip((f.x.values * SIZE / w).astype(int), 0, SIZE - 1)
    rows = np.clip((f.y.values * SIZE / h).astype(int), 0, SIZE - 1)
    np.add.at(hm, (rows, cols), wt)
    hm = gaussian_filter(hm, sigma=SIGMA)
    hm = pool_to_out(hm)
    return hm / hm.sum() if hm.sum() > 0 else None


def ellipse_map(ell: pd.DataFrame, finding: str, w: int, h: int):
    cols = [c for c in ELLIPSE_COLS[finding] if c in ell.columns]
    if not cols or ell.empty:
        return None
    flag = ell[cols].astype(str).apply(lambda s: s.str.lower().isin(["true", "1", "1.0"])).any(axis=1)
    sel = ell[flag & (pd.to_numeric(ell["certainty"], errors="coerce") >= 3)]
    if sel.empty:
        return None
    yy, xx = np.mgrid[0:SIZE, 0:SIZE] + 0.5
    m = np.zeros((SIZE, SIZE))
    for r in sel.itertuples():
        x0, x1 = r.xmin * SIZE / w, r.xmax * SIZE / w
        y0, y1 = r.ymin * SIZE / h, r.ymax * SIZE / h
        cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, max((x1 - x0) / 2, 0.5), max((y1 - y0) / 2, 0.5)
        m = np.maximum(m, (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1).astype(float))
    m = pool_to_out(m)
    return m / m.sum() if m.sum() > 0 else None


def stack_mean(maps: list):
    """판독별 K채널 지도(None 가능)를 채널별로 평균. 없는 채널은 0. K = len(FINDINGS)"""
    K = len(FINDINGS)
    out = np.zeros((K, OUT_SIZE, OUT_SIZE))
    has = np.zeros(K, bool)
    for k in range(K):
        ch = [m[k] for m in maps if m[k] is not None]
        if ch:
            a = np.mean(ch, axis=0)
            out[k], has[k] = a / a.sum(), True
    return out, has


def save_qc(index: pd.DataFrame, imgs: pd.DataFrame, win: float) -> None:
    qc = PROC / "qc"
    qc.mkdir(exist_ok=True)
    full = index.assign(n_ch=index[[f"fs{win:g}_{f}" for f in FINDINGS]].sum(axis=1)).sort_values("n_ch", ascending=False)
    full = full[full.n_ch > 0]  # 지도가 하나라도 있는 영상만
    picks = (pd.concat([full[full.datasets.str.contains("egd")].head(2),
                        full[(full.datasets == "reflacx") & full.ellipse_any].head(2)]) if PREFIX == "fs"
             else full.head(4))
    for k, r in enumerate(picks.itertuples(), 1):
        path = imgs.loc[imgs.dicom_id == r.dicom_id, "image_path"].iloc[0]
        im = Image.open(path).convert("L").resize((OUT_SIZE, OUT_SIZE))
        fs = np.load(PROC / f"heatmaps_{PREFIX}_w{win:g}_{OUT_SIZE}" / f"{r.dicom_id}.npy").astype(float)
        gl = PROC / "heatmaps_512_s7" / f"{r.dicom_id}.npy"
        panels = [("global gaze", pool_to_out(np.load(gl).astype(float)) if gl.exists() else None)]
        panels += [(f"{f} gaze ({win:g}s)", fs[i] if fs[i].sum() > 0 else None) for i, f in enumerate(FINDINGS)]
        ep = PROC / f"heatmaps_ellipse_{OUT_SIZE}" / f"{r.dicom_id}.npy"
        if ep.exists() and PREFIX == "fs":
            el = np.load(ep).astype(float)
            panels += [(f"{f} ellipse", el[i] if el[i].sum() > 0 else None) for i, f in enumerate(FINDINGS)]
        fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.3))
        for ax, (title, hm) in zip(axes, panels):
            ax.imshow(im, cmap="gray")
            if hm is not None:
                ax.imshow(hm, cmap="jet", alpha=0.45)
            ax.set_title(title if hm is not None else title + " (none)", fontsize=8)
            ax.axis("off")
        fig.suptitle(f"{r.datasets} {r.dicom_id[:12]}", fontsize=9)
        fig.savefig(qc / f"{PREFIX}_{k}_{r.datasets}.png", dpi=80, bbox_inches="tight")
        plt.close(fig)


# ------------------------------------------------------------ 메인
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=float, nargs="*", default=[1.5, 3.0, 5.0])
    ap.add_argument("--main-window", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-images", type=int, default=0, help="동작 확인용")
    ap.add_argument("--edema-set", choices=list(EDEMA_SETS), default="v1")
    ap.add_argument("--finding-set", choices=["base", "more"], default="base", help="25절: more = 무기폐·경화")
    args = ap.parse_args()
    global FINDINGS, ROOTS, PREFIX
    more = args.finding_set == "more"
    if more:
        FINDINGS, ROOTS, PREFIX = list(MORE_FINDINGS), dict(MORE_ROOTS), "fs2"
        if args.main_window not in args.windows:
            raise SystemExit(f"--finding-set more: --windows에 주 창 {args.main_window:g}초가 있어야 합니다 (예: --windows 3)")
        print(f"25절 소견 확장: {FINDINGS}, 단어 어근 {ROOTS}, 창 {args.windows}")
    else:
        ROOTS["edema"] = EDEMA_SETS[args.edema_set]
        print(f"폐부종 단어 세트 {args.edema_set}: {ROOTS['edema']}")

    reads = pd.read_csv(PROC / "cohort_readings.csv")
    split = pd.read_csv(PROC / f"split_seed{args.seed}.csv")[["dicom_id", "split"]]
    imgs = pd.read_csv(PROC / "cohort_images.csv").merge(split, on="dicom_id")
    lab_file = PROC / ("more_labels.csv" if more else "cohort_images_labeled.csv")
    if not lab_file.exists():
        raise SystemExit(f"label 파일이 없습니다: {lab_file} (먼저 python build_more_labels.py)")
    lab = pd.read_csv(lab_file)[["dicom_id"] + [f"{f}_t3" for f in FINDINGS]]
    target = imgs[imgs.split.isin(["train", "val"])]
    if more:  # 이 소견 label은 REFLACX에만 있음
        target = target[target.dicom_id.isin(lab.dicom_id)]
    target = target.reset_index(drop=True)
    if args.max_images:
        target = target.head(args.max_images)
    usable = reads[reads.gaze_ok & reads.dicom_id.isin(target.dicom_id)]

    egd_file = reads.loc[reads.dataset == "egd", "gaze_file"].iloc[0]
    egd_groups = pd.read_csv(egd_file, usecols=["DICOM_ID", "X_ORIGINAL", "Y_ORIGINAL", "FPOGS", "FPOGD"]).groupby("DICOM_ID")

    dirs = {w: PROC / f"heatmaps_{PREFIX}_w{w:g}_{OUT_SIZE}" for w in args.windows}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    ell_dir = PROC / f"heatmaps_ellipse_{OUT_SIZE}"
    ell_dir.mkdir(parents=True, exist_ok=True)

    img_rows, men_rows = [], []
    for i, img in enumerate(target.itertuples(), 1):
        w, h = Image.open(img.image_path).size
        per_win = {win: [] for win in args.windows}
        ell_maps = []
        for r in usable[usable.dicom_id == img.dicom_id].itertuples():
            words = load_words(r)
            fix = load_fixations(r, egd_groups)
            mrow = {"reading_id": r.reading_id, "dataset": r.dataset, "dicom_id": r.dicom_id, "split": img.split,
                    "n_words": len(words), "fix_end_max": fix["end"].max() if len(fix) else np.nan,
                    "word_end_max": words["end"].max() if len(words) else np.nan}
            for win in args.windows:
                maps = []
                for f in FINDINGS:
                    wins = mention_windows(words, f, win)
                    dwell = dwell_in_windows(fix, wins)
                    maps.append(gaze_map(fix, dwell, w, h) if wins else None)
                    if win == args.main_window:
                        mrow[f"{f}_mentions"] = len(wins)
                        mrow[f"{f}_fix_in_window"] = int((dwell > 0).sum())
                per_win[win].append(maps)
            if r.dataset == "reflacx" and not more:
                p = Path(r.gaze_file).parent / "anomaly_location_ellipses.csv"
                ell = pd.read_csv(p) if p.exists() else pd.DataFrame()
                ell_maps.append([ellipse_map(ell, f, w, h) for f in FINDINGS])
            men_rows.append(mrow)

        row = {"dicom_id": img.dicom_id, "subject_id": img.subject_id, "split": img.split, "datasets": img.datasets}
        for win in args.windows:
            m, has = stack_mean(per_win[win]) if per_win[win] else (None, np.zeros(len(FINDINGS), bool))
            if has.any():
                np.save(dirs[win] / f"{img.dicom_id}.npy", m.astype(np.float16))
            for k, f in enumerate(FINDINGS):
                row[f"fs{win:g}_{f}"] = bool(has[k])
        m, has = stack_mean(ell_maps) if ell_maps else (None, np.zeros(len(FINDINGS), bool))
        if has.any() and not more:
            np.save(ell_dir / f"{img.dicom_id}.npy", m.astype(np.float16))
        for k, f in enumerate(FINDINGS):
            row[f"ellipse_{f}"] = bool(has[k])
        row["ellipse_any"] = bool(has.any())
        img_rows.append(row)
        if i % 500 == 0:
            print(f"  {i}/{len(target)}장 처리", flush=True)

    index = pd.DataFrame(img_rows).merge(lab, on="dicom_id", how="left")
    index.to_csv(PROC / f"{PREFIX}_heatmap_index.csv", index=False)
    json.dump({"finding_set": args.finding_set, "findings": FINDINGS, "edema_set": None if more else args.edema_set,
               "roots": ROOTS, "windows": args.windows},
              open(PROC / f"{PREFIX}_build_config.json", "w"), indent=2)
    men = pd.DataFrame(men_rows)
    men.to_csv(PROC / f"{PREFIX}_mentions.csv", index=False)
    save_qc(index, imgs, args.main_window)

    # ------------------------------------------------------ 요약
    pd.set_option("display.width", 200)
    print("\n=== 시간 맞춤 확인 (판독 단위, 중앙값 초): 시선 마지막 시각 vs 녹취 마지막 단어 시각")
    print(men.groupby("dataset")[["fix_end_max", "word_end_max", "n_words"]].median().round(1).to_string())
    print(f"\n=== 단어가 나온 판독 중 창({args.main_window:g}초) 안에 고정점이 하나라도 있는 비율")
    for f in FINDINGS:
        m = men[men[f"{f}_mentions"] > 0]
        print(f"  {f:8s}", {ds: f"{(g[f'{f}_fix_in_window'] > 0).mean():.3f} (판독 {len(g)})" for ds, g in m.groupby("dataset")})
    print("\n=== 영상 단위 coverage: finding별 지도가 있는 비율 (train)")
    tr = index[index.split == "train"]
    cov = []
    for f in FINDINGS:
        y = tr[f"{f}_t3"] == 1
        r = {"finding": f}
        for win in args.windows:
            c = tr[f"fs{win:g}_{f}"]
            r[f"{win:g}초 전체"], r[f"{win:g}초 양성"], r[f"{win:g}초 음성"] = c.mean(), c[y].mean(), c[~y].mean()
        if not more:
            e = tr[f"ellipse_{f}"]
            r["ellipse 전체"], r["ellipse 양성"] = e.mean(), e[y].mean()
        cov.append(r)
    print(pd.DataFrame(cov).round(3).to_string(index=False))
    print("\n데이터셋별 (train, 주 창):")
    print(tr.groupby("datasets")[[f"fs{args.main_window:g}_{f}" for f in FINDINGS]].mean().round(3).to_string())
    print(f"\n저장: {', '.join(str(d) for d in dirs.values())}" + ("" if more else f"\n      {ell_dir}")
          + f"\n      {PROC / f'{PREFIX}_heatmap_index.csv'}\n      {PROC / f'{PREFIX}_mentions.csv'}\n      {PROC / 'qc'}/{PREFIX}_*.png")


if __name__ == "__main__":
    main()
