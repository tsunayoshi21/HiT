# Rodrigo — Spanish historical manuscript (in-domain for DHiSS)

## Dataset
The **Rodrigo corpus**: a single 1545 manuscript (the *Historia de España* of Archbishop
Rodrigo Jiménez de Rada), written by one hand in **Old Spanish**. A classic HTR benchmark.

- **Source:** Zenodo — *Rodrigo corpus 1.0.0*, `doi:10.5281/zenodo.1490009`.
  Citation: Serrano, N., Castro-Bleda, M. J., & Juan, A. *The RODRIGO Database.* LREC 2010.
- **Split used:** the official `test` partition → **205 pages**.

## Ground truth — how it was built
The corpus is distributed at **line level** (`images/Rodrigo_<page>_<line>.png` +
`text/transcriptions.txt`). We reconstruct one image + one GT per page with
`prep_rodrigo_pages.py`, which **stacks the line crops of each test page vertically**
(scaled to ≤2048 px on the long side) and **joins their transcriptions** in order
(`_` → space). Output: 205 page images + `DOCxx_GT.txt` (here shipped in `gt/`).

## Results (olmOCR-2, vLLM, greedy)
See `results/RESULTS.md`; per-doc metrics in `results/rodrigo_full.pkl`.
Predictions (one `.txt` per page) in `results/predictions/{baseline,hit_onnxtr,hit_dhiss_plus}/`.

**Summary:** the in-domain anchor **DHiSS+ ties the baseline** (WER 0.4953 vs 0.4954,
paired Wilcoxon p=0.84); the generic anchor is significantly worse. HiT does not beat
the baseline here.

## Reproduce
```bash
# 0. download + reconstruct pages (needs the Zenodo tar with images/ text/ partitions/)
python prep_rodrigo_pages.py --dl <rodrigo_dl_dir> --n 205 \
       --out-img page_images/2048 --out-gt gt

# 1. anchors (env: doctr for DHiSS+, onnxtr for the generic one) — see ../code/
python ../code/gen_dhiss_anchors_from_images.py --image-dir page_images/2048 \
       --reco-model DHiSS_v2_vitstr_base --ocr-threshold 0.90 --out-dir anchors/dhiss_v2_vitstr_th90
python ../code/gen_onnxtr_anchors.py --image-dir page_images/2048 \
       --out-dir anchors/onnxtr_parseqML_th50 --threshold 0.5

# 2. olmOCR-2, 3 conditions in one batch (env: vLLM)
python ../code/run_vlm_vllm.py --image-dir page_images/2048 --output-parent results/predictions \
       --shard 0/1 --gpu 0 \
       --cond baseline:: \
       --cond hit_onnxtr:append:anchors/onnxtr_parseqML_th50 \
       --cond hit_dhiss_plus:append:anchors/dhiss_v2_vitstr_th90

# 3. per-doc pickle
python ../code/build_results_pickle.py --gt gt --dataset rodrigo_full --out results/rodrigo_full.pkl \
       baseline=results/predictions/baseline \
       hit_onnxtr=results/predictions/hit_onnxtr \
       hit_dhiss_plus=results/predictions/hit_dhiss_plus
```
