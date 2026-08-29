# External-dataset validation of HiT

Beyond the paper's internal 14-document test set (Table 3), we validate HiT on **two
established, public historical-document datasets** that are out-of-distribution for the
DHiSS recognizer. This isolates when anchor injection helps, and whether the anchor
needs to be *in-domain*.

Everything here uses the **same HiT method**: word-level coordinate anchors appended to
a VLM prompt. The backbone is **olmOCR-2** (`allenai/olmOCR-2-7B-1025`), served with
**vLLM** (greedy). For every dataset we compare **three conditions**:

| Condition | Anchor recognizer | Character |
|---|---|---|
| `baseline` | — (no HiT) | the VLM alone |
| `hit_onnxtr` | ParSeq-Multilingual, pretrained (`onnxtr`) | generic, printed-multilingual |
| `hit_dhiss_plus` | **DHiSS+** (v2 ViTSTR, `../../ocr_weights/`) | **in-domain** (Spanish historical HTR) |

The two anchors span the domain axis: DHiSS+ is specialized for Spanish historical
handwriting; onnxtr's ParSeq-Multilingual is a generic printed-text recognizer.

```
external_datasets/
├── code/                              # shared runners (dataset-agnostic)
│   ├── run_vlm_vllm.py                # olmOCR-2 via vLLM, N conditions in one batch, doc-sharded
│   ├── gen_dhiss_anchors_from_images.py  # DHiSS/DHiSS+ anchors from page PNGs (doctr)
│   ├── gen_onnxtr_anchors.py          # generic ParSeq-Multilingual anchors (onnxtr)
│   ├── build_results_pickle.py        # per-doc metrics -> .pkl (for paired tests)
│   └── metrics.py                     # quick WER/CER of a predictions dir vs GT
├── rodrigo/                           # Spanish manuscript (in-domain for DHiSS)
└── finebooks/                         # printed multilingual books (out-of-domain; official BHL leaderboard)
```

## Headline result

On both datasets, at scale and against a strong olmOCR-2 baseline, **HiT does not
improve transcription**; the in-domain anchor (**DHiSS+**) is the only one that does not
hurt, while the generic anchor degrades quality. HiT's measurable benefit is
**robustness** — it suppresses hallucination on blank/plate pages (see finebooks'
official *over-extraction*, 81% → 24%).

Per-document metrics for paired statistics are in each dataset's
`results/<dataset>_full.pkl` (`d['long']` is a tidy DataFrame). See each subfolder's
`README.md` and `results/RESULTS.md`.

> Surya (a neural OCR we also tried as an anchor source) is **not included**: it did not
> change the conclusions and is omitted.
