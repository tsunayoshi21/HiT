# finebooks — results (olmOCR-2, vLLM, greedy · 1 649 pages)

## Official BHL leaderboard metrics
`score_finebooks_official.py` · 1 227 text pages / 422 plates · HiT wrapper-echo stripped.

| Condition | Reading-CER ↓ | Read-Acc ↑ | Recall ↑ | Over-extraction ↓ | Loop ↓ |
|---|---|---|---|---|---|
| **baseline** | **0.0388** | **96.1%** | 0.8905 | 81.3% | 0.4% |
| hit_onnxtr | 0.0414 | 95.9% | 0.8942 | **24.2%** | 0.1% |
| hit_dhiss_plus | 0.0402 | 96.0% | 0.8877 | **24.2%** | 0.1% |

Baseline 96.1% reading-accuracy ≈ leaderboard's 95.7% for olmOCR-2 (same scale).
Transcription: baseline best; in-domain DHiSS+ (0.0402) < generic onnxtr (0.0414).
Robustness: HiT cuts over-extraction on plates 81% → 24% and loops 0.4% → 0.1%.

## Plain per-doc WER/CER
Per-doc values in `finebooks_full.pkl`. Means over all pages are inflated by the 422
blank plates (baseline hallucination); read the median.

| Condition | WER mean | WER median | CER mean | CER median |
|---|---|---|---|---|
| baseline | 11.11 | 0.328 | 59.48 | 0.0589 |
| hit_onnxtr | 1.10 | 0.328 | 21.10 | 0.0615 |
| hit_dhiss_plus | 1.10 | 0.333 | 21.09 | 0.0601 |

Filtered to the 1 336 pages with real text (GT ≥ 20 chars): baseline WER 0.4045 /
CER 0.1671; hit_dhiss_plus 0.4068 / 0.1506; hit_onnxtr 0.4133 / 0.2090.
