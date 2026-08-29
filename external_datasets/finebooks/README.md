# finebooks / BHL — printed multilingual books (out-of-domain; official leaderboard)

## Dataset
Pages from **five 18th–19th-century natural-history books** (shells, British birds,
fishes, chitin skeleton, Cuvier's *Histoire Naturelle*) digitised by the Biodiversity
Heritage Library. Printed, engraved plates, period typography.

- **Source:** Hugging Face dataset `finebooks/bhl-impact-gt`.
- **License / citation:** CC-BY 3.0 — *IMPACT Centre of Competence & Biodiversity
  Heritage Library (2012). IMPACT-BHL ground truth (PRImA PAGE XML, CC-BY 3.0).*
- **GT:** from the EU IMPACT project (2011–2012): ABBYY FineReader + manual re-keying/QA,
  ~99.95% character accuracy. Shipped in `gt/` (`text` field, reading order).
- **Size:** **1 649 latin-script pages** (a 6th, Cyrillic, volume is excluded).
  Languages: EN · FR · DE · LA.
- **`manifest.json`:** per-page `barcode`, `gt_chars`, `sparse` — **422 pages are plates**
  (illustrations, empty GT). `prep_finebooks_full.py` downloads the pages from the HF
  dataset and builds this manifest.

## Official BHL leaderboard metrics
finebooks has an official leaderboard (HF blog, "Historical Books OCR Leaderboard").
`score_finebooks_official.py` reproduces its metric definitions:

- **Reading-CER** (micro, reading-lane: de-hyphenation, NFKC, ſ→s treated as correct) — main metric.
- **Diplomatic-CER** (ſ not modernised; identical here — the IMPACT GT is already modernised).
- **Recall** (word-level), **Over-extraction** (spurious text on plates), **Loop-rate**.
- HiT predictions are scored after stripping the trivial anchor-wrapper echo (`RAW_TEXT_*`).

```
$ python score_finebooks_official.py baseline hit_onnxtr hit_dhiss_plus
```
(see `results/RESULTS.md` for the numbers). Our baseline reads **96.1%** reading-accuracy,
matching the leaderboard's **95.7%** for olmOCR-2 → we measure on the same scale.

**Summary:** on the official transcription metric the **baseline is best**; HiT does not
help (the in-domain DHiSS+ anchor 0.0402 < generic onnxtr 0.0414 — in-domain hurts less).
HiT's real effect is robustness: **over-extraction on plates drops 81% → 24%** and the
loop-rate falls (the baseline invents text on blank pages; HiT does not).

## Results
Per-doc metrics in `results/finebooks_full.pkl`; plain WER/CER in `results/RESULTS.md`;
predictions in `results/predictions/{baseline,hit_onnxtr,hit_dhiss_plus}/`.

> Note: plain **mean** WER/CER over all 1 649 pages is dominated by the 422 blank plates
> (the baseline hallucinates there → astronomical means). Use the median, filter by
> `gt_chars`, or the official over-extraction metric.

## Reproduce
```bash
python prep_finebooks_full.py                 # downloads pages + GT from HF, writes manifest.json
# anchors + run + pickle: same 3-step recipe as rodrigo/README.md (../code/)
python score_finebooks_official.py baseline hit_onnxtr hit_dhiss_plus
```
