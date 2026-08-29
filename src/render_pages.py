#!/usr/bin/env python3
"""
Renderiza cada pagina de cada PDF a un PNG (lado mayor = target_dim px) y las cachea.
Usa el mismo renderer (olmocr.render_pdf_to_base64png) y resolucion (2048px) que los
experimentos principales, para que los baselines locales (DeepSeek-OCR, Kraken, Calamari)
operen sobre imagenes identicas a las del resto de la Tabla 3.

Salida: <out_dir>/<DOCID>_p<NN>.png   (1-indexado)

Uso:
    conda activate olmo_doc
    python render_pages.py --pdf-folder PDFs --target-dim 2048 --out-dir page_images/2048
"""
import argparse
import base64
from io import BytesIO
from pathlib import Path

import PyPDF2
from PIL import Image
from olmocr.data.renderpdf import render_pdf_to_base64png


def page_count(pdf):
    with open(pdf, "rb") as f:
        return len(PyPDF2.PdfReader(f).pages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-folder", default="PDFs")
    ap.add_argument("--target-dim", type=int, default=2048)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path("page_images") / str(args.target_dim)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(Path(args.pdf_folder).glob("*.pdf"))
    total = 0
    for pdf in pdfs:
        doc = pdf.stem
        n = page_count(pdf)
        for p in range(1, n + 1):
            out = out_dir / f"{doc}_p{p:02d}.png"
            if out.exists() and not args.overwrite:
                total += 1
                continue
            b64 = render_pdf_to_base64png(str(pdf), p, target_longest_image_dim=args.target_dim)
            img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
            img.save(out)
            total += 1
            print(f"{doc} p{p:02d} -> {img.size}", flush=True)
    print(f"LISTO: {total} paginas en {out_dir}")


if __name__ == "__main__":
    main()
