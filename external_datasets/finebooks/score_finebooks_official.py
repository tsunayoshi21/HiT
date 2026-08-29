#!/usr/bin/env python3
"""
Scoring de finebooks/BHL con las métricas OFICIALES del leaderboard:
  - Reading-CER  (micro): moderniza ſ->s (equivalencias tratadas como correctas)
  - Diplomatic-CER (micro): NO moderniza ſ (long-s cuenta como error)
  - Recall (micro, palabras): de las palabras que debían estar, cuántas produjo
  - Over-extraction: texto generado en páginas-lámina (GT vacío) -> anti-alucinación
  - Loop rate: % de páginas con salida desbocada (proxy: > LOOP_CHARS chars)
Micro = sum(edits)/sum(chars_ref) sobre páginas de TEXTO (manifest sparse=False).
Uso: python score_finebooks_official.py <cond1> <cond2> ...
"""
import json, os, re, sys, unicodedata
try:
    from Levenshtein import distance as lev
except Exception:
    from rapidfuzz.distance.Levenshtein import distance as lev

BASE = "."                      # carpeta finebooks/ de la submission
RESROOT = "results/predictions"  # baseline/ hit_onnxtr/ hit_dhiss_plus/
LOOP_CHARS = 8000
man = json.load(open(f"{BASE}/manifest.json"))


def strip_wrapper(s):
    """Quita el eco del wrapper HiT (post-proceso trivial): RAW_TEXT_START/END,
    'Page dimensions:' y líneas de ancla [XxY]palabra que el modelo a veces repite."""
    s = re.sub(r"RAW_TEXT_START|RAW_TEXT_END", " ", s)
    s = re.sub(r"(?m)^\s*Page dimensions:.*$", " ", s)
    s = re.sub(r"(?m)^\s*\[\d+x\d+\].*$", " ", s)
    return s


def _common(s):
    s = strip_wrapper(s)
    s = re.sub(r"<[^>]+>", " ", s)          # tags XML/HTML
    s = re.sub(r"\$[^$]*\$", " ", s)        # markup $...$
    s = s.replace("­", "")             # soft hyphen
    s = re.sub(r"-\s*\n\s*", "", s)         # de-hifenación fin de línea
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def reading_norm(s):
    return _common(s).replace("ſ", "s")   # ſ -> s (modernización)


def diplomatic_norm(s):
    return _common(s)                          # conserva ſ


def read(p):
    return open(p, encoding="utf-8").read()


def score(cond):
    # micro accumulators
    r_ed = r_ch = d_ed = d_ch = 0
    rec_hit = rec_tot = 0
    text_n = 0
    sparse_n = 0; sparse_halluc = 0; sparse_out_chars = 0
    loop_n = 0; all_n = 0
    for doc, info in man.items():
        fp = f"{RESROOT}/{cond}/{doc}_{cond}.txt"; gp = f"{BASE}/gt/{doc}_GT.txt"
        if not (os.path.isfile(fp) and os.path.isfile(gp)):
            continue
        raw = read(fp)
        all_n += 1
        if len(raw) > LOOP_CHARS:
            loop_n += 1
        if info["sparse"]:
            sparse_n += 1
            out = reading_norm(raw)
            sparse_out_chars += len(out)
            if len(out) > 20:
                sparse_halluc += 1
            continue
        # página de texto
        gt_r = reading_norm(read(gp)); oc_r = reading_norm(raw)
        gt_d = diplomatic_norm(read(gp)); oc_d = diplomatic_norm(raw)
        r_ed += lev(oc_r, gt_r); r_ch += max(len(gt_r), 1)
        d_ed += lev(oc_d, gt_d); d_ch += max(len(gt_d), 1)
        gw = set(gt_r.split()); pw = set(oc_r.split())
        rec_hit += sum(1 for w in gw if w in pw); rec_tot += len(gw)
        text_n += 1
    return dict(
        text_n=text_n, sparse_n=sparse_n,
        reading_cer=r_ed / max(r_ch, 1),
        diplomatic_cer=d_ed / max(d_ch, 1),
        recall=rec_hit / max(rec_tot, 1),
        over_extraction_rate=sparse_halluc / max(sparse_n, 1),
        sparse_mean_chars=sparse_out_chars / max(sparse_n, 1),
        loop_rate=loop_n / max(all_n, 1),
    )


rows = [(c, score(c)) for c in sys.argv[1:]]
print(f"{'condición':16s} {'ReadCER':>8s} {'ReadAcc%':>8s} {'DiplCER':>8s} "
      f"{'Recall':>7s} {'OverExt%':>8s} {'SparseCh':>8s} {'Loop%':>6s}  (nText/nSparse)")
for c, s in rows:
    print(f"{c:16s} {s['reading_cer']:8.4f} {100*(1-s['reading_cer']):8.2f} "
          f"{s['diplomatic_cer']:8.4f} {s['recall']:7.4f} {100*s['over_extraction_rate']:8.1f} "
          f"{s['sparse_mean_chars']:8.0f} {100*s['loop_rate']:6.1f}  "
          f"({s['text_n']}/{s['sparse_n']})")
