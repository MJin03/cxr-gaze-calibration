"""18절 준비: 판독 음성기록(transcript)에서 흉막 세부 소견 단어가 몇 장에 나오는지 센다 (학습·평가 결과는 보지 않음).

세부 소견 (단어 어근, 소문자·알파벳만 남긴 단어의 앞부분 일치)
    effusion   effusion
    thickening thicken
    apical     apical, apex, apices
    blunting   blunt, costophrenic
    plaque     plaque
부정: 앞 4단어 안에 no, without, negative, free, resolved, resolution, clear 가 있으면 "부정 언급"으로 따로 셈
출력: split(train/val/test) × 흉막 label(t3 양성/음성)별, 세부 소견이 부정 없이 언급된 영상 수

판단 규칙 (ANALYSIS_PLAN 18절): train 흉막 양성 영상에서 effusion·thickening·apical 각각 100장 이상이면 세부 출력 실험을 설계, 아니면 하지 않음

사용법  python count_pleural_words.py
"""
import pandas as pd

from build_fs_heatmaps import PROC, load_words

SUB = {"effusion": ["effusion"], "thickening": ["thicken"], "apical": ["apical", "apex", "apices"],
       "blunting": ["blunt", "costophrenic"], "plaque": ["plaque"]}
NEG = {"no", "without", "negative", "free", "resolved", "resolution", "clear"}


def flags(words):
    w = words["word"].tolist()
    pos, neg = {k: False for k in SUB}, {k: False for k in SUB}
    for i, tok in enumerate(w):
        for k, roots in SUB.items():
            if any(tok.startswith(r) for r in roots):
                if NEG & set(w[max(0, i - 4):i]):
                    neg[k] = True
                else:
                    pos[k] = True
    return pos, neg


def main():
    reads = pd.read_csv(PROC / "cohort_readings.csv")
    lab = pd.read_csv(PROC / "cohort_images_labeled.csv")[["dicom_id", "pleural_t3"]]
    sp = pd.read_csv(PROC / "split_seed42.csv")
    reads = reads.merge(sp, on="dicom_id").merge(lab, on="dicom_id")
    rows = []
    for k, r in enumerate(reads.itertuples(), 1):
        words = load_words(r)
        pos, neg = flags(words) if len(words) else ({s: False for s in SUB}, {s: False for s in SUB})
        rows.append({"dicom_id": r.dicom_id, "dataset": r.dataset, "split": r.split, "pleural_t3": r.pleural_t3,
                     "has_words": len(words) > 0, **{f"pos_{s}": pos[s] for s in SUB}, **{f"neg_{s}": neg[s] for s in SUB}})
        if k % 500 == 0:
            print(f"  {k}/{len(reads)} 판독 읽음", flush=True)
    t = pd.DataFrame(rows)
    # 영상 단위: 판독 여러 개면 하나라도 언급하면 언급
    agg = {c: "max" for c in t.columns if c.startswith(("pos_", "neg_", "has_"))}
    img = t.groupby(["dicom_id", "split", "pleural_t3"]).agg(agg).reset_index()
    out = PROC / "pleural_subword_counts.csv"
    img.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print(f"\n=== 판독 기록이 있는 영상 {int(img.has_words.sum())}/{len(img)}장")
    for sp_ in ["train", "val", "test"]:
        g = img[img.split == sp_]
        print(f"\n[{sp_}] 영상 {len(g)}장 (흉막 양성 {int(g.pleural_t3.sum())})")
        tab = pd.DataFrame({
            "흉막 양성: 언급": [int(g.loc[g.pleural_t3 == 1, f"pos_{s}"].sum()) for s in SUB],
            "흉막 양성: 부정만": [int((g.loc[g.pleural_t3 == 1, f"neg_{s}"] & ~g.loc[g.pleural_t3 == 1, f"pos_{s}"]).sum()) for s in SUB],
            "흉막 음성: 언급": [int(g.loc[g.pleural_t3 == 0, f"pos_{s}"].sum()) for s in SUB],
            "흉막 음성: 부정만": [int((g.loc[g.pleural_t3 == 0, f"neg_{s}"] & ~g.loc[g.pleural_t3 == 0, f"pos_{s}"]).sum()) for s in SUB],
        }, index=list(SUB))
        print(tab.to_string())
    tr = img[(img.split == "train") & (img.pleural_t3 == 1)]
    print("\n=== 판단 (train 흉막 양성 영상, 부정 없는 언급 ≥ 100장)")
    for s in ["effusion", "thickening", "apical"]:
        n = int(tr[f"pos_{s}"].sum())
        print(f"  {s:11s} {n:5d}장 → {'충분' if n >= 100 else '부족'}")
    ok = all(int(tr[f"pos_{s}"].sum()) >= 100 for s in ["effusion", "thickening", "apical"])
    print(f"  → 세부 출력 실험: {'설계 진행' if ok else '하지 않음 (세부 소견 학습 자료 부족)'}")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
