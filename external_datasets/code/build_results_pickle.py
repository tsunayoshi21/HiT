#!/usr/bin/env python
"""
Construye un pickle con métricas POR-DOCUMENTO para varias condiciones de un dataset,
para análisis estadístico posterior (tests pareados baseline vs HiT, etc.).

Uso:
  python build_results_pickle.py --gt smoke_finebooks_full/gt \
      --out resultados_pickles/finebooks_full.pkl --dataset finebooks_full \
      baseline=smoke_finebooks_full/results/fb_baseline \
      hit_onnxtr=smoke_finebooks_full/results/fb_onnxtr \
      hit_surya=smoke_finebooks_full/results/fb_surya

Salida (pickle): dict con
  - 'dataset': nombre
  - 'per_doc': dict {condición: {doc: {WER,CER,MER,WIL,WIP, n_ref_words,n_ref_chars,...}}}
  - 'long': pandas.DataFrame formato largo [dataset,condition,doc,WER,CER,MER,WIL,WIP,...]
  - 'docs': lista ordenada de doc ids comunes a todas las condiciones
Además imprime medias/medianas por condición.
"""
import argparse, glob, os, pickle
import statistics as st
import jiwer
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--gt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("conditions", nargs="+", help="label=preds_dir ...")
args = ap.parse_args()

gts = {os.path.basename(g).replace("_GT.txt", ""): open(g, encoding="utf-8").read()
       for g in glob.glob(f"{args.gt}/*_GT.txt")}

conds = {}
for c in args.conditions:
    label, d = c.split("=", 1)
    conds[label] = d


def metrics(ref, hyp):
    w = jiwer.process_words(ref, hyp)
    c = jiwer.process_characters(ref, hyp)
    return dict(WER=w.wer, MER=w.mer, WIL=w.wil, WIP=w.wip, CER=c.cer,
                n_ref_words=len(ref.split()), n_ref_chars=len(ref),
                n_hyp_words=len(hyp.split()), n_hyp_chars=len(hyp))


per_doc = {}
rows = []
for label, d in conds.items():
    per_doc[label] = {}
    for f in sorted(glob.glob(f"{d}/*.txt")):
        doc = os.path.basename(f)[:-4].split("_")[0]
        if doc not in gts:
            continue
        m = metrics(gts[doc], open(f, encoding="utf-8").read())
        per_doc[label][doc] = m
        rows.append(dict(dataset=args.dataset, condition=label, doc=doc, **m))

# docs comunes a TODAS las condiciones (para tests pareados)
common = set.intersection(*[set(per_doc[l]) for l in conds]) if conds else set()
docs = sorted(common)

df = pd.DataFrame(rows)
out = dict(dataset=args.dataset, per_doc=per_doc, long=df, docs=docs,
           conditions=list(conds.keys()))
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "wb") as fh:
    pickle.dump(out, fh)

print(f"=== {args.dataset}: pickle -> {args.out} ===")
print(f"condiciones: {list(conds.keys())} | docs comunes: {len(docs)}")
for label in conds:
    ws = [per_doc[label][d]["WER"] for d in docs]
    cs = [per_doc[label][d]["CER"] for d in docs]
    print(f"  {label:12s}: n={len(ws)} WER media={st.mean(ws):.4f} med={st.median(ws):.4f} | "
          f"CER media={st.mean(cs):.4f} med={st.median(cs):.4f}")
