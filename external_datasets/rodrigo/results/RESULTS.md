# Rodrigo — results (olmOCR-2, vLLM, greedy · 205 test pages)

WER/CER (lower is better), per-document then averaged. Per-doc values in `rodrigo_full.pkl`.

| Condition | Anchor | WER mean | WER median | CER mean | CER median | vs baseline (WER, paired Wilcoxon) |
|---|---|---|---|---|---|---|
| baseline | — | 0.4954 | 0.5000 | 0.1098 | 0.1084 | — |
| hit_onnxtr | ParSeq-Multilingual (generic) | 0.5100 | 0.5145 | 0.1319 | 0.1142 | worse, p < 1e-4 |
| **hit_dhiss_plus** | **DHiSS+ (in-domain)** | **0.4953** | **0.4956** | 0.1112 | **0.1074** | tie, p = 0.84 |

**Reading:** only the in-domain anchor (DHiSS+) is statistically indistinguishable from
the baseline; the generic anchor degrades WER and CER. No HiT variant beats the baseline.
