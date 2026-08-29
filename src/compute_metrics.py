#!/usr/bin/env python
"""Compute OCR quality metrics for one or more experiment output folders.

For every experiment it reads the model transcription and the ground truth and
reports 13 metrics, averaged over the documents of the test set:

    BLEU, SIMILARITY, ROUGE1, ROUGE2, ROUGE-L, ROUGE-LSUM, BERTSCORE,
    METEOR, WER, CER, MER, WIL, WIP

CPU metrics (everything except BERTScore) run in parallel across cores; BERTScore
runs on the GPU in a single batched call. ROUGE-L uses an exact bit-parallel LCS
(identical values to the reference O(n*m) table, just faster on long documents).

Expected layout (run from the repository root):
    <gt-dir>/DOC{01..14}_GT.txt
    <results-dir>/<exp>/DOC{01..14}_<exp>.txt

Usage:
    # all experiment folders found under results/
    python src/compute_metrics.py

    # only some experiments, custom output
    python src/compute_metrics.py --exps olmocr2_baseline olmocr2_append --out metrics_olmocr2.csv
"""
import os
# BERTScore runs on the first visible GPU. Override CUDA_VISIBLE_DEVICES outside
# if you need a specific device.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import glob
import time
import difflib
import argparse
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_DOCS = 14                       # documents with ground truth (DOC01..DOC14)
BERT_MODEL = "albert-base-v1"
BERT_LANG = "es"

COLS = ["BLEU", "SIMILARITY", "ROUGE1", "ROUGE2", "ROUGE-L", "ROUGE-LSUM",
        "BERT", "METEOR", "WER", "CER", "MER", "WIL", "WIP"]


# ---------------------------------------------------------------------------
# Exact bit-parallel LCS length (Crochemore et al. 2001) for a fast ROUGE-L.
# Returns the SAME LCS length as the standard O(n*m) DP table, so ROUGE-L values
# are identical; this only avoids the ~minutes-long pure-Python table on the
# largest documents.
# ---------------------------------------------------------------------------
def _lcs_len_bitparallel(ref, can):
    if not ref or not can:
        return 0
    pm = {}
    for j, tok in enumerate(can):
        pm[tok] = pm.get(tok, 0) | (1 << j)
    n = len(can)
    full = (1 << n) - 1
    v = full
    get = pm.get
    for tok in ref:
        u = v & get(tok, 0)
        v = ((v + u) | (v - u)) & full
    return n - bin(v).count("1")


def _patch_rouge_lcs():
    from rouge_score import rouge_scorer, scoring

    def _fast_score_lcs(target_tokens, prediction_tokens):
        if not target_tokens or not prediction_tokens:
            return scoring.Score(precision=0, recall=0, fmeasure=0)
        lcs_length = _lcs_len_bitparallel(target_tokens, prediction_tokens)
        precision = lcs_length / len(prediction_tokens)
        recall = lcs_length / len(target_tokens)
        fmeasure = scoring.fmeasure(precision, recall)
        return scoring.Score(precision=precision, recall=recall, fmeasure=fmeasure)

    rouge_scorer._score_lcs = _fast_score_lcs


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def docstr(doc_id):
    return f"DOC{doc_id:02d}"


# ---------------------------------------------------------------------------
# CPU-metric worker (one task = one (experiment, document) pair)
# ---------------------------------------------------------------------------
_ROUGE = _BLEU = _METEOR = None


def _init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""   # workers never touch the GPU
    global _ROUGE, _BLEU, _METEOR
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    _patch_rouge_lcs()
    from evaluate import load
    _ROUGE = load("rouge")
    _BLEU = load("bleu")
    _METEOR = load("meteor")


def _cpu_metrics(task):
    name, doc_id, gt, ocr = task
    from nltk.tokenize import word_tokenize
    import jiwer

    d = {}
    d["BLEU"] = _BLEU.compute(predictions=[ocr], references=[gt], tokenizer=word_tokenize)["bleu"]
    d["SIMILARITY"] = difflib.SequenceMatcher(None, gt, ocr).ratio()
    r = _ROUGE.compute(predictions=[ocr], references=[gt], tokenizer=word_tokenize)
    d["ROUGE1"] = r["rouge1"]
    d["ROUGE2"] = r["rouge2"]
    d["ROUGE-L"] = r["rougeL"]
    d["ROUGE-LSUM"] = r["rougeLsum"]
    d["METEOR"] = _METEOR.compute(predictions=[ocr], references=[gt])["meteor"]
    w = jiwer.process_words(gt, ocr)
    d["WER"] = w.wer
    d["MER"] = w.mer
    d["WIL"] = w.wil
    d["WIP"] = w.wip
    d["CER"] = jiwer.process_characters(gt, ocr).cer
    return name, doc_id, d


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
def build_tasks(exps, gt_dir, results_dir):
    gts = {i: read_file(os.path.join(gt_dir, f"{docstr(i)}_GT.txt")) for i in range(1, N_DOCS + 1)}
    tasks, missing = [], []
    for name in exps:
        for i in range(1, N_DOCS + 1):
            f = os.path.join(results_dir, name, f"{docstr(i)}_{name}.txt")
            if not os.path.isfile(f):
                missing.append(f)
                continue
            tasks.append((name, i, gts[i], read_file(f)))
    if missing:
        print("ERROR - missing input files:")
        for f in missing:
            print("  ", f)
        raise SystemExit(1)
    return tasks


def compute(exps, gt_dir, results_dir, workers):
    tasks = build_tasks(exps, gt_dir, results_dir)
    print(f"Computing {len(exps)} experiment(s) x {N_DOCS} docs = {len(tasks)} pairs.")
    results = {name: {docstr(i): {} for i in range(1, N_DOCS + 1)} for name in exps}

    # 1) CPU metrics in parallel
    t0 = time.time()
    workers = min(workers, len(tasks))
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        done = 0
        for name, doc_id, d in ex.map(_cpu_metrics, tasks, chunksize=1):
            results[name][docstr(doc_id)].update(d)
            done += 1
            if done % 20 == 0 or done == len(tasks):
                print(f"  CPU metrics {done}/{len(tasks)}  ({time.time()-t0:.1f}s)", flush=True)
    print(f"  -> CPU metrics done in {time.time()-t0:.1f}s (workers={workers})")

    # 2) BERTScore, batched on GPU
    t1 = time.time()
    from evaluate import load
    bert = load("bertscore")
    preds = [t[3] for t in tasks]
    refs = [t[2] for t in tasks]
    f1 = bert.compute(predictions=preds, references=refs, lang=BERT_LANG,
                      model_type=BERT_MODEL, batch_size=64)["f1"]
    for (name, doc_id, gt, ocr), score in zip(tasks, f1):
        results[name][docstr(doc_id)]["BERT"] = score
    print(f"  -> BERTScore (batched) in {time.time()-t1:.1f}s")
    return results


def write_csv(results, csv_path):
    import pandas as pd
    averages = {}
    for name, docs in results.items():
        acc = {c: 0.0 for c in COLS}
        for doc, metrics in docs.items():
            for m, v in metrics.items():
                acc[m] += v
        n = len(docs)
        averages[name] = {c: acc[c] / n for c in COLS}
    df = pd.DataFrame.from_dict(averages, orient="index").rename(columns={"BERT": "BERTSCORE"})
    df.to_csv(csv_path)
    print(f"\nCSV written: {csv_path} ({len(df)} experiments)")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df[["BLEU", "ROUGE-L", "BERTSCORE", "WER", "CER"]].round(4).to_string())
    return df


def main():
    ap = argparse.ArgumentParser(description="Compute OCR metrics for experiment folders.")
    ap.add_argument("--gt-dir", default="data/gt", help="Folder with DOCxx_GT.txt files")
    ap.add_argument("--results-dir", default="results", help="Parent folder of experiment subfolders")
    ap.add_argument("--exps", nargs="+", default=None,
                    help="Experiment folder names (default: every subfolder of --results-dir)")
    ap.add_argument("--out", default="metrics.csv")
    ap.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 8))
    args = ap.parse_args()

    exps = args.exps
    if not exps:
        exps = sorted(p.name for p in Path(args.results_dir).iterdir() if p.is_dir())
        if not exps:
            raise SystemExit(f"No experiment subfolders found under {args.results_dir}/")
        print(f"Auto-detected experiments: {exps}")

    results = compute(exps, args.gt_dir, args.results_dir, args.workers)
    write_csv(results, args.out)


if __name__ == "__main__":
    main()
