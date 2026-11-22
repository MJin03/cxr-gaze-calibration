"""14절: PadChest(세 번째 병원, 스페인 San Juan 병원) 의사 label 정면 영상을 받아 학습 영상과 같은 형식(512×512 흑백 PNG)으로 만든다.

규칙 (ANALYSIS_PLAN.md 14절, 다운로드·예측 전에 고정)
    대상  MethodLabel == Physician, Projection ∈ {PA, AP, AP_horizontal}, Pediatric != PED,
          ImageDir ∈ {0..11, 54} (12번 zip은 의사 정면 306장뿐이라 제외)
    제외  label 해석 불가·빈 label, "exclude" 포함, label이 "unchanged"뿐(판독문이 소견을 나열하지 않음)
    label (판독문 1명 → reader 수 k = 1)
          pleural_effusion = pleural effusion, loculated pleural effusion
          pleural          = pleural_effusion + pleural thickening 계열(apical/calcified pleural thickening, pleural plaques,
                             calcified pleural plaques) + hydropneumothorax, empyema, pleural mass   (VinDr: effusion 또는 thickening)
          cardiac          = cardiomegaly
          edema            = pulmonary edema, kerley lines (양성 약 100장 → 보조 결과)
          민감도: pleural_cpa = pleural + costophrenic angle blunting, edema_hi = edema + heart insufficiency,
                  suboptimal·unchanged_any 표시 (제외 민감도 분석용)
    묶음  batch a = zip 0~5, batch b = zip 6~11, 54 (13절의 같은 병원 다른 묶음 평가용)
    영상  16-bit PNG → plain(최솟값~최댓값), win(CSV의 WindowCenter/Width), eq(plain에 histogram equalization)
          → 512×512 직접 축소(BILINEAR) → 내부 train과 히스토그램 L1 거리 최소인 방식 선택 (label·예측 사용 안 함)
          흑백 반전 여부는 finalize의 qc_examples.png를 보고 예측 전에 정함 (--invert)

순서  zip 하나 받기 → 필요한 영상만 zip 안에서 바로 읽어 변환 → zip 삭제. 다음 zip은 변환하는 동안 미리 받음.
      서버 디스크는 한 번에 zip 2개(약 45GB)까지만 씀. 끊겨도 다시 실행하면 끝난 zip은 건너뛰고 받던 zip은 이어 받음.

링크  환경변수 PADCHEST_TOKEN (링크의 /s/ 뒤 부분). 화면·명령줄에 나오지 않도록 0600 임시 설정 파일로 wget에 전달.

사용법
    python build_padchest_cache.py plan                 # 선택 규칙 적용 결과만 출력 (다운로드 없음)
    python build_padchest_cache.py run [--zips 0-11,54] # 다운로드 + 변환 (dl 세션)
    python build_padchest_cache.py monitor              # 1분마다 진행 상황 (mon 세션)
    python build_padchest_cache.py finalize [--invert none|mono1|all]   # labels.csv, 전처리 선택, QC 그림
"""
import argparse
import ast
import io
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timedelta
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

HOME = Path.home() / "X-ray"
PROC = HOME / "data/processed"
RAW = HOME / "data/raw/padchest"
OUT = PROC / "padchest"
ZIPDIR = RAW / "zips"
STATUS = OUT / "status.json"
LOG = HOME / "results/logs/padchest_dl.txt"
WGETRC = Path.home() / ".cache/padchest_wgetrc"
WEBDAV = "https://b2drop.bsc.es/public.php/webdav"
SIZE = 512
ZIPS_DEFAULT = list(range(12)) + [54]
BATCH_A = set(range(6))
FRONTAL = ["PA", "AP", "AP_horizontal"]

EFF = {"pleural effusion", "loculated pleural effusion"}
PLEURAL = EFF | {"pleural thickening", "apical pleural thickening", "calcified pleural thickening", "pleural plaques",
                 "calcified pleural plaques", "hydropneumothorax", "empyema", "pleural mass"}
CARDIAC = {"cardiomegaly"}
EDEMA = {"pulmonary edema", "kerley lines"}


def parse_zips(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def parse_labels(s):
    try:
        v = ast.literal_eval(s) if isinstance(s, str) else None
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(v, (list, tuple)):
        return None
    return {str(x).strip().lower() for x in v if str(x).strip()}


def select(zips):
    """선택 규칙 적용. (선택된 표, 제외 단계별 개수)"""
    t = pd.read_csv(next(RAW.glob("PADCHEST_*.gz")), low_memory=False)
    steps = [("전체", len(t))]
    t = t[t.MethodLabel == "Physician"]; steps.append(("의사 label", len(t)))
    t = t[t.Projection.isin(FRONTAL)]; steps.append(("정면 PA/AP/AP_horizontal", len(t)))
    t = t[t.Pediatric != "PED"]; steps.append(("소아 제외", len(t)))
    t = t[t.ImageDir.isin(zips)].copy(); steps.append((f"zip {zips[0]}~{zips[-1]} 중 선택 zip", len(t)))
    t["labset"] = t.Labels.map(parse_labels)
    t = t[t.labset.map(lambda s: bool(s))]; steps.append(("label 해석 가능·비어 있지 않음", len(t)))
    t = t[~t.labset.map(lambda s: "exclude" in s)]; steps.append(("'exclude' 제외", len(t)))
    t = t[t.labset.map(lambda s: len(s - {"unchanged"}) > 0)]; steps.append(("'unchanged'만 있는 영상 제외", len(t)))
    t = t.drop_duplicates("ImageID"); steps.append(("ImageID 중복 제거", len(t)))
    has = lambda S: t.labset.map(lambda s: int(len(s & S) > 0))  # noqa: E731
    t["pleural_effusion"], t["pleural"] = has(EFF), has(PLEURAL)
    t["cardiac"], t["edema"] = has(CARDIAC), has(EDEMA)
    t["pleural_cpa"] = has(PLEURAL | {"costophrenic angle blunting"})
    t["edema_hi"] = has(EDEMA | {"heart insufficiency"})
    t["suboptimal"] = has({"suboptimal study"})
    t["unchanged_any"] = has({"unchanged"})
    t["batch"] = np.where(t.ImageDir.isin(BATCH_A), "a", "b")
    return t, steps


def first_num(v):
    m = re.search(r"-?\d+(\.\d+)?", str(v))
    return float(m.group(0)) if m else np.nan


# ---------- 다운로드 ----------
def write_wgetrc():
    tok = os.environ.get("PADCHEST_TOKEN", "").strip()
    if not tok:
        sys.exit("PADCHEST_TOKEN이 비어 있음. 안내된 read -s 줄을 먼저 실행")
    WGETRC.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(WGETRC, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(f"http_user = {tok}\nhttp_password = \nauth_no_challenge = on\n")
    os.chmod(WGETRC, 0o600)


def wget_env():
    return {**os.environ, "WGETRC": str(WGETRC)}


def remote_sizes():
    r = subprocess.run(["wget", "-q", "-O", "-", "--method=PROPFIND", "--header=Depth: 1", f"{WEBDAV}/"],
                       env=wget_env(), capture_output=True, text=True, timeout=300)
    sizes = {}
    for blk in re.split(r"<d:response>", r.stdout):
        m = re.search(r"<d:href>[^<]*/(\d+)\.zip</d:href>", blk)
        n = re.search(r"<d:getcontentlength>(\d+)</d:getcontentlength>", blk)
        if m and n:
            sizes[int(m.group(1))] = int(n.group(1))
    return sizes


def start_download(i):
    ZIPDIR.mkdir(parents=True, exist_ok=True)
    part = ZIPDIR / f"{i}.zip.part"
    return subprocess.Popen(["wget", "-c", "-q", "--tries=50", "--waitretry=30", "--read-timeout=120",
                             "-O", str(part), f"{WEBDAV}/{i}.zip"], env=wget_env())


def finish_download(i, proc, expected, tries=3):
    part, done = ZIPDIR / f"{i}.zip.part", ZIPDIR / f"{i}.zip"
    for k in range(tries):
        rc = proc.wait()
        size = part.stat().st_size if part.exists() else -1
        if rc == 0 and (expected is None or size == expected):
            part.rename(done)
            return done
        log(f"zip {i} 다운로드 불완전 (코드 {rc}, {size}/{expected} bytes) → 이어 받기 재시도 {k + 1}/{tries}")
        time.sleep(30)
        proc = start_download(i)
    raise RuntimeError(f"zip {i} 다운로드 실패")


# ---------- 변환 ----------
_Z = None


def _init(zpath):
    global _Z
    _Z = zipfile.ZipFile(zpath)


def to_variants(arr, wc, ww):
    a = arr.astype(np.float64)
    lo, hi = a.min(), a.max()
    plain = ((a - lo) / (hi - lo) * 255.0) if hi > lo else np.zeros_like(a)
    if np.isfinite(wc) and np.isfinite(ww) and ww > 1:
        w = np.clip((a - (wc - ww / 2)) / ww, 0, 1) * 255.0
        win_ok = True
    else:
        w, win_ok = plain, False
    p8, w8 = Image.fromarray(plain.astype(np.uint8)), Image.fromarray(w.astype(np.uint8))
    return p8, w8, ImageOps.equalize(p8), win_ok


def work(job):
    member, iid, wc, ww = job
    try:
        data = _Z.read(member)  # CRC 검사 포함
        arr = np.asarray(Image.open(io.BytesIO(data)))
        if arr.ndim == 3:
            arr = arr[..., 0]
        h, w = arr.shape
        p8, w8, e8, win_ok = to_variants(arr, wc, ww)
        for v, im in [("plain", p8), ("win", w8), ("eq", e8)]:
            im.resize((SIZE, SIZE), Image.BILINEAR).save(OUT / f"images_512_{v}" / f"{iid}.png")
        return {"image_id": iid, "width": w, "height": h, "raw_min": int(arr.min()), "raw_max": int(arr.max()),
                "win_ok": win_ok, "error": ""}
    except Exception as e:  # noqa: BLE001
        return {"image_id": iid, "width": np.nan, "height": np.nan, "raw_min": np.nan, "raw_max": np.nan,
                "win_ok": False, "error": str(e)[:200]}


def process_zip(i, zpath, rows, workers):
    with zipfile.ZipFile(zpath) as z:
        names = {Path(n).name: n for n in z.namelist() if n.lower().endswith(".png")}
    jobs, miss = [], []
    for r in rows.itertuples():
        n = names.get(r.ImageID)
        if n is None:
            miss.append(r.ImageID)
        else:
            jobs.append((n, Path(r.ImageID).stem, first_num(r.WindowCenter_DICOM), first_num(r.WindowWidth_DICOM)))
    with Pool(workers, initializer=_init, initargs=(str(zpath),)) as pool:
        meta = pd.DataFrame(pool.map(work, jobs, chunksize=8))
    if miss:
        meta = pd.concat([meta, pd.DataFrame({"image_id": [Path(m).stem for m in miss], "error": "zip에 없음"})])
    meta["ImageDir"] = i
    return meta


# ---------- 상태·로그 ----------
def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def save_status(st):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATUS.with_suffix(".tmp")
    json.dump(st, open(tmp, "w"))
    tmp.replace(STATUS)


def cmd_plan(args):
    t, steps = select(parse_zips(args.zips))
    print("=== 선택 단계")
    for k, n in steps:
        print(f"  {k:32s} {n:7d}")
    print(f"\n환자 {t.PatientID.nunique()}명")
    cols = ["pleural", "pleural_effusion", "cardiac", "edema", "pleural_cpa", "edema_hi", "suboptimal", "unchanged_any"]
    print("\n=== label 양성 수 (batch별)")
    print(t.groupby("batch")[cols].sum().T.assign(합계=lambda d: d.sum(axis=1)).to_string())
    print("\n=== batch별 영상 수:", t.batch.value_counts().sort_index().to_dict())
    print("Projection:", t.Projection.value_counts().to_dict())
    print("Photometric:", t.PhotometricInterpretation_DICOM.value_counts().to_dict())
    print("Window 값 있음:", int(t.WindowCenter_DICOM.map(first_num).notna().sum()))


def cmd_run(args):
    zips = parse_zips(args.zips)
    t, steps = select(zips)
    for v in ["plain", "win", "eq"]:
        (OUT / f"images_512_{v}").mkdir(parents=True, exist_ok=True)
    (OUT / "meta").mkdir(parents=True, exist_ok=True)
    write_wgetrc()
    try:
        sizes = remote_sizes()
        if not sizes:
            sys.exit("PadChest 파일 목록을 못 읽음 → 링크(PADCHEST_TOKEN)를 다시 입력")
        if not all(i in sizes for i in zips):
            log(f"[주의] 원격 크기를 못 읽은 zip: {[i for i in zips if i not in sizes]} (크기 확인 없이 진행)")
        todo = [i for i in zips if not (OUT / "meta" / f"{i}.csv").exists()]
        st = {"zips": zips, "expected": {str(i): sizes.get(i) for i in zips}, "n_total": int(len(t)),
              "done": [i for i in zips if i not in todo], "downloaded": [], "dl": None, "proc": None,
              "started": time.time(), "bytes_at_start": 0}
        st["bytes_at_start"] = sum(sizes.get(i, 0) for i in st["done"]) + sum(
            (ZIPDIR / f"{i}.zip.part").stat().st_size for i in todo if (ZIPDIR / f"{i}.zip.part").exists())
        save_status(st)
        log(f"대상 {len(t)}장, zip {len(zips)}개 중 남은 {len(todo)}개: {todo}")
        if not todo:
            log("모두 완료. finalize 실행")
            return
        pending = {}

        def launch(i):
            if (ZIPDIR / f"{i}.zip").exists():
                pending[i] = None
            else:
                pending[i] = start_download(i)
                st["dl"] = i; save_status(st)
                log(f"zip {i} 다운로드 시작 ({(sizes.get(i) or 0) / 1e9:.1f} GB)")

        launch(todo[0])
        for k, i in enumerate(todo):
            zpath = ZIPDIR / f"{i}.zip"
            if pending[i] is not None:
                zpath = finish_download(i, pending[i], sizes.get(i))
            st["downloaded"].append(i); st["dl"] = None; save_status(st)
            log(f"zip {i} 다운로드 완료")
            if k + 1 < len(todo):
                launch(todo[k + 1])
            st["proc"] = i; save_status(st)
            rows = t[t.ImageDir == i]
            log(f"zip {i} 변환 시작 ({len(rows)}장)")
            try:
                meta = process_zip(i, zpath, rows, args.workers)
            except zipfile.BadZipFile as e:
                log(f"zip {i} 손상 ({e}) → 삭제 후 다시 받기 필요. 다시 실행하면 처음부터 받음")
                zpath.unlink(missing_ok=True)
                raise
            meta.to_csv(OUT / "meta" / f"{i}.csv", index=False)
            nerr = int((meta.error.fillna("") != "").sum())
            if not args.keep_zip:
                zpath.unlink()
            st["done"].append(i); st["proc"] = None; save_status(st)
            log(f"zip {i} 변환 완료: 성공 {len(meta) - nerr}장, 실패 {nerr}장 → zip 삭제"
                f"  [전체 {len(st['done'])}/{len(zips)} zip]")
        log("모든 zip 완료. 다음: finalize")
    finally:
        WGETRC.unlink(missing_ok=True)


def cmd_monitor(args):
    prev = None
    while True:
        if not STATUS.exists():
            print(f"[{datetime.now():%H:%M}] 아직 시작 전 (status 없음)", flush=True)
            time.sleep(60); continue
        st = json.load(open(STATUS))
        exp = {int(k): (v or 0) for k, v in st["expected"].items()}
        total = sum(exp.values())
        got = sum(exp[i] for i in set(st["done"]) | set(st["downloaded"]))
        cur = ""
        if st.get("dl") is not None:
            p = ZIPDIR / f"{st['dl']}.zip.part"
            s = p.stat().st_size if p.exists() else 0
            got += s
            cur = f"받는 중 zip {st['dl']} {s / 1e9:.1f}/{exp[st['dl']] / 1e9:.1f}GB"
        if st.get("proc") is not None:
            cur += f"  변환 중 zip {st['proc']}"
        now = time.time()
        rate = (got - prev[1]) / (now - prev[0]) if prev and now > prev[0] else \
            (got - st["bytes_at_start"]) / max(now - st["started"], 1)
        prev = (now, got)
        remain = max(total - got, 0)
        if rate > 0:
            eta = timedelta(seconds=int(remain / rate))
            fin = (datetime.now() + eta).strftime("%m-%d %H:%M")
        else:
            eta, fin = "?", "?"
        el = timedelta(seconds=int(now - st["started"]))
        nimg = sum(1 for _ in (OUT / "images_512_plain").glob("*.png")) if (OUT / "images_512_plain").exists() else 0
        last = ""
        if LOG.exists():
            lines = [l for l in LOG.read_text(errors="ignore").splitlines() if l.strip()]
            last = lines[-1][-90:] if lines else ""
        print(f"[{datetime.now():%H:%M}] zip 완료 {len(st['done'])}/{len(st['zips'])} | {cur.strip() or '대기'} | "
              f"받음 {got / 1e9:.0f}/{total / 1e9:.0f}GB ({100 * got / max(total, 1):.0f}%) | {rate / 1e6:.1f}MB/s | "
              f"영상 {nimg}/{st['n_total']} | 경과 {el} | 남은 {eta} → 완료 {fin} | 최근: {last}", flush=True)
        if len(st["done"]) == len(st["zips"]):
            print("전부 완료", flush=True)
            break
        time.sleep(60)


def cmd_finalize(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(Path(__file__).parent))
    from build_vindr_cache import mean_hist

    zips = parse_zips(args.zips)
    t, _ = select(zips)
    metas = [pd.read_csv(OUT / "meta" / f"{i}.csv") for i in zips if (OUT / "meta" / f"{i}.csv").exists()]
    if len(metas) < len(zips) and not args.allow_partial:
        sys.exit(f"아직 끝나지 않은 zip 있음 ({len(metas)}/{len(zips)})")
    meta = pd.concat(metas)
    meta["error"] = meta.error.fillna("")
    bad = meta[meta.error != ""]
    if len(bad):
        print(f"[주의] 변환 실패 {len(bad)}장 (제외). 예: {bad.error.iloc[0]}")
    ok = meta[meta.error == ""]
    t["image_id"] = t.ImageID.map(lambda s: Path(s).stem)
    lab = t.merge(ok[["image_id", "width", "height", "raw_min", "raw_max", "win_ok"]], on="image_id")

    if args.invert != "none" and (OUT / "invert_done.json").exists():
        print("흑백 반전은 이미 적용됨 → 건너뜀 (두 번 반전 방지)")
    elif args.invert != "none":
        sel = lab if args.invert == "all" else lab[lab.PhotometricInterpretation_DICOM == "MONOCHROME1"]
        for v in ["plain", "win", "eq"]:
            for iid in sel.image_id:
                p = OUT / f"images_512_{v}" / f"{iid}.png"
                Image.fromarray(255 - np.asarray(Image.open(p))).save(p)
        json.dump({"invert": args.invert, "n": int(len(sel)), "when": datetime.now().isoformat()},
                  open(OUT / "invert_done.json", "w"))
        print(f"흑백 반전 적용: {len(sel)}장 ({args.invert})")

    out = pd.DataFrame({"image_id": lab.image_id, "group": lab.PatientID, "batch": lab.batch,
                        "pleural": lab.pleural, "pleural_effusion": lab.pleural_effusion, "cardiac": lab.cardiac,
                        "edema": lab.edema, "pleural_cpa": lab.pleural_cpa, "edema_hi": lab.edema_hi,
                        "suboptimal": lab.suboptimal, "unchanged_any": lab.unchanged_any, "view": lab.Projection,
                        "photometric": lab.PhotometricInterpretation_DICOM, "zip": lab.ImageDir,
                        "width": lab.width, "height": lab.height, "win_ok": lab.win_ok})
    out.to_csv(OUT / "labels.csv", index=False)
    for b in ["a", "b"]:
        out[out.batch == b].to_csv(OUT / f"labels_{b}.csv", index=False)

    split = pd.read_csv(PROC / "split_seed42.csv")
    ids = split.loc[split.split == "train", "dicom_id"].sample(frac=1, random_state=0).head(1000)
    h_int = mean_hist([PROC / "images_512" / f"{d}.png" for d in ids])
    samp = out.image_id.sample(min(3000, len(out)), random_state=0)
    rows = []
    for v in ["plain", "win", "eq"]:
        h = mean_hist([OUT / f"images_512_{v}" / f"{i}.png" for i in samp])
        rows.append({"variant": v, "L1_to_internal": float(np.abs(h - h_int).sum()), "mean": float((h * np.arange(256)).sum())})
    choice = min(rows, key=lambda r: r["L1_to_internal"])["variant"]
    json.dump({"choice": choice, "variants": rows, "internal_mean": float((h_int * np.arange(256)).sum()),
               "rule": "내부 train 영상과 히스토그램 L1 거리 최소 (무작위 3,000장)", "invert": args.invert},
              open(OUT / "preprocess_choice.json", "w"), indent=2, ensure_ascii=False)

    # QC: photometric별 12장씩 (label 안 봄). 사람이 보고 반전 여부를 예측 전에 결정
    fig, ax = plt.subplots(4, 6, figsize=(15, 10.5))
    picks = []
    for ph in ["MONOCHROME1", "MONOCHROME2"]:
        s = out[out.photometric == ph]
        picks += [(ph, i) for i in s.image_id.sample(min(12, len(s)), random_state=0)]
    for a in ax.ravel():
        a.axis("off")
    for a, (ph, iid) in zip(ax.ravel(), picks):
        a.imshow(np.asarray(Image.open(OUT / f"images_512_{choice}" / f"{iid}.png")), cmap="gray", vmin=0, vmax=255)
        a.set_title(ph, fontsize=8)
    fig.suptitle(f"PadChest QC ({choice}) — 뼈·심장이 밝고 폐가 어두우면 정상")
    fig.tight_layout()
    fig.savefig(OUT / "qc_examples.png", dpi=80)

    print("\n=== label (정면 영상, batch별)")
    cols = ["pleural", "pleural_effusion", "cardiac", "edema", "pleural_cpa", "edema_hi"]
    print(out.groupby("batch")[cols].sum().T.assign(합계=lambda d: d.sum(axis=1)).to_string())
    print(f"영상 {len(out)}장 (a {int((out.batch == 'a').sum())}, b {int((out.batch == 'b').sum())}), "
          f"환자 {out.group.nunique()}명")
    print("Photometric:", out.photometric.value_counts().to_dict(), "| window 값 사용:", int(out.win_ok.sum()))
    print("\n=== 전처리 선택 (label·예측 사용 안 함)")
    print(pd.DataFrame(rows).round(3).to_string(index=False), "→", choice)
    print(f"QC 그림: {OUT / 'qc_examples.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run", "monitor", "finalize"])
    ap.add_argument("--zips", default="0-11,54")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--keep-zip", action="store_true")
    ap.add_argument("--invert", choices=["none", "mono1", "all"], default="none")
    ap.add_argument("--allow-partial", action="store_true", help="동작 확인용")
    args = ap.parse_args()
    {"plan": cmd_plan, "run": cmd_run, "monitor": cmd_monitor, "finalize": cmd_finalize}[args.cmd](args)


if __name__ == "__main__":
    main()
