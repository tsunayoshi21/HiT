#!/usr/bin/env python
"""WER/CER medio de un dir de predicciones vs GT (matching por doc id)."""
import argparse, glob, os, statistics as st, jiwer

ap = argparse.ArgumentParser()
ap.add_argument("--preds", required=True)
ap.add_argument("--gt", required=True)
ap.add_argument("--tag", default="")
args = ap.parse_args()

gts = {os.path.basename(g).replace("_GT.txt", ""): open(g, encoding="utf-8").read()
       for g in glob.glob(f"{args.gt}/*_GT.txt")}
rows = []
for f in sorted(glob.glob(f"{args.preds}/*.txt")):
    base = os.path.basename(f)[:-4]
    doc = base.split("_")[0]
    if doc not in gts:
        continue
    pred = open(f, encoding="utf-8").read()
    ref = gts[doc]
    w = jiwer.process_words(ref, pred)
    c = jiwer.process_characters(ref, pred)
    rows.append((doc, w.wer, c.cer))

if not rows:
    print(f"{args.tag}: SIN MATCHES"); raise SystemExit
wer = st.mean(r[1] for r in rows); cer = st.mean(r[2] for r in rows)
mwer = st.median(r[1] for r in rows); mcer = st.median(r[2] for r in rows)
print(f"{args.tag}: n={len(rows)} WER={wer:.3f} CER={cer:.3f} (mediana WER={mwer:.3f} CER={mcer:.3f})")
for d, w, c in rows:
    print(f"    {d}: WER={w:.3f} CER={c:.3f}")
