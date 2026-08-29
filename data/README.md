# Test set

The paper's internal test set: **14 Spanish-language documents** (244 pages total),
one PDF per document in `pdfs/` (`DOC01.pdf` … `DOC14.pdf`) and one ground-truth
transcription per document in `gt/` (`DOCxx_GT.txt`). They are scanned/typewritten
20th-century Chilean judicial and archival documents — a demanding, low-quality-scan
setting for OCR.

- **Rendering.** Pages are rasterised at 2048 px on the longest side (`../src/render_pages.py`).
- **Ground truth.** Manual transcription per document, plain text, in reading order.
- **Metrics.** Computed per document and averaged (`../src/compute_metrics.py`).

## Sources

| Doc | Pages | Original document | Source |
|---|---:|---|---|
| DOC01 | 4  | (Ampliación) Declaración M. Townley — 21.04.78 | CIPER — Letelier collection |
| DOC02 | 10 | Declaración M. Callejas — 16.09.91 | CIPER — Letelier collection |
| DOC03 | 5  | Comparecencia M. Townley — 01.04.78 | CIPER — Letelier collection |
| DOC04 | 27 | Declaración M. Townley — sin fechar | CIPER — Letelier collection |
| DOC05 | 2  | Quiñones | Vicaría — obtained via collaboration |
| DOC06 | 11 | Quiñones | Vicaría — obtained via collaboration |
| DOC07 | 51 | Declaración M. Townley — sin fechar | CIPER — Letelier collection |
| DOC08 | 68 | Confirmación sentencia caso Letelier — 30.05.1995 | CIPER — Letelier collection |
| DOC09 | 1  | Carta de M. Townley a Gustavo Etchepare — 02.06.78 | CIPER — Letelier collection |
| DOC10 | 2  | Minuta policial caso Letelier — sin fechar | CIPER — Letelier collection |
| DOC11 | 33 | Informe Manuel Contreras | El Mundo (elmundo.es) |
| DOC12 | 2  | Confesión Michael Townley — 13.03.78 | CIPER — Letelier collection |
| DOC13 | 6  | Declaración Eric Marcy — septiembre 1991 | CIPER — Letelier collection |
| DOC14 | 22 | Querella caso Letelier — 20.07.1991 | CIPER — Letelier collection |
| **Total** | **244** | | |

**Collections / links**

- **CIPER — Letelier collection** (DOC01–04, DOC07–10, DOC12–14) — public Google Journalist
  Studio *Pinpoint* collection:
  `https://journaliststudio.google.com/pinpoint/search?collection=dde1fa4b375f1cdb`
- **El Mundo** (DOC11) — *Informe Manuel Contreras*, published by elmundo.es:
  article `https://www.elmundo.es/elmundo/2005/05/13/internacional/1116005207.html`,
  PDF `https://e01-elmundo.uecdn.es/documentos/2005/05/13/informe_manuel_contreras.pdf`
- **Vicaría** (DOC05–06) — obtained via collaboration (no direct public link).

> The source documents are reproduced here for research reproducibility. Rights remain
> with the original holders; confirm terms before any further redistribution.
