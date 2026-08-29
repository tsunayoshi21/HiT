# Test set

The paper's internal test set: **14 Spanish historical documents** (244 pages total),
one PDF per document in `pdfs/` (`DOC01.pdf` … `DOC14.pdf`) and one ground-truth
transcription per document in `gt/` (`DOCxx_GT.txt`).

- **Rendering.** Pages are rasterised at 2048 px on the longest side (`src/render_pages.py`).
- **Ground truth.** Manual transcription per document, plain text, in reading order.
- **Metrics.** Computed per document and averaged (`src/compute_metrics.py`).

> Provenance / redistribution of the source documents follows the terms described in the
> paper. If you plan to redistribute, confirm the rights for each document with its
> holding institution.
