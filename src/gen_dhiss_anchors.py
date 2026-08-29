#!/usr/bin/env python3
"""
Genera y cachea los anchors del recognizer DHiSS (doctr) para un conjunto de PDFs.

Reusa la logica de `ocr_inference.py` (initialize_ocr_model + get_anchor_with_ocr_model)
para producir, por cada pagina, el texto de anclaje con coordenadas
(formato `[XxY]palabra`) que luego se inyecta en el prompt de cualquier VLM
(DeepSeek-OCR, Mistral/pixtral, etc.). Esto desacopla la generacion de anchors
(que necesita doctr + los pesos DHiSS) de la inferencia de los baselines nuevos.

Los anchors se guardan en:
    <out_dir>/<DOCID>_p<NN>.txt        (uno por pagina, 1-indexado)

Uso:
    conda activate olmo_doc
    python gen_dhiss_anchors.py \
        --pdf-folder data/pdfs \
        --reco-model DHiSS_v2_vitstr_base \
        --ocr-threshold 0.90 \
        --gpu 0 \
        --out-dir anchors/DHiSS_v2_vitstr_base_th90

"""
import argparse
import logging
import sys
from pathlib import Path

import ocr_inference as oi


def build_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("gen_dhiss_anchors")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    lg.addHandler(fh)
    return lg


def main():
    ap = argparse.ArgumentParser(description="Genera y cachea anchors DHiSS por pagina.")
    ap.add_argument("--pdf-folder", default="data/pdfs")
    ap.add_argument("--reco-model", default="DHiSS_v2_vitstr_base",
                    choices=list(oi.SUPPORTED_MODELS.keys()))
    ap.add_argument("--detector-model", default="db_resnet50")
    ap.add_argument("--ocr-threshold", type=float, default=0.90)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out-dir", default=None,
                    help="Carpeta de salida (default: dhiss_anchors/<reco>_th<NN>)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Regenerar aunque el anchor ya exista en cache")
    args = ap.parse_args()

    thr_tag = f"th{int(round(args.ocr_threshold * 100)):02d}"
    out_dir = Path(args.out_dir) if args.out_dir else Path("anchors") / f"{args.reco_model}_{thr_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = build_logger(Path("logs") / f"gen_anchors_{args.reco_model}_{thr_tag}.log")
    # ocr_inference usa un logger a nivel de modulo (definido en su main()); lo inyectamos.
    oi.logger = logger
    oi.CONFIG["gpu_id"] = args.gpu
    oi.CONFIG["reco_model"] = args.reco_model
    oi.CONFIG["detector_model"] = args.detector_model

    print(f"Inicializando OCR model (reco={args.reco_model}, det={args.detector_model}, gpu={args.gpu})...")
    ocr_model = oi.initialize_ocr_model(reco=args.reco_model, detector=args.detector_model)

    pdfs = sorted(Path(args.pdf_folder).glob("*.pdf"))
    if not pdfs:
        print(f"No se encontraron PDFs en {args.pdf_folder}", file=sys.stderr)
        sys.exit(1)

    print(f"PDFs: {len(pdfs)} | umbral={args.ocr_threshold} | salida={out_dir}")
    total_pages = 0
    for pdf in pdfs:
        doc_id = pdf.stem  # DOCxx
        n = oi.get_pdf_page_count(str(pdf))
        print(f"[{doc_id}] {n} paginas")
        for page in range(1, n + 1):
            out_f = out_dir / f"{doc_id}_p{page:02d}.txt"
            if out_f.exists() and not args.overwrite:
                total_pages += 1
                continue
            try:
                anchor = oi.get_anchor_with_ocr_model(
                    str(pdf), page, ocr_model, threshold=args.ocr_threshold
                )
            except Exception as e:
                logger.error(f"Error anchor {doc_id} p{page}: {e}")
                anchor = ""
            out_f.write_text(anchor, encoding="utf-8")
            total_pages += 1
            print(f"  {doc_id} p{page:02d} -> {len(anchor)} chars", flush=True)

    print(f"LISTO: {total_pages} paginas cacheadas en {out_dir}")


if __name__ == "__main__":
    main()
