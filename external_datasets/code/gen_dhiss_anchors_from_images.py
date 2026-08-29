#!/usr/bin/env python3
"""Genera anclas DHiSS ([XxY]palabra) directamente desde IMAGENES de pagina (PNG),
para datasets HF que ya vienen como imagen (no PDF). Reusa el reconocedor y el
formato de ocr_inference.get_anchor_with_ocr_model (misma logica, pero from_images).

Uso (env olmo_doc_cu128):
    python gen_anchors_from_images.py --image-dir smoke_hf/page_images/2048 \
        --reco-model DHiSS_v2_vitstr_base --ocr-threshold 0.90 --gpu 0 \
        --out-dir smoke_hf/anchors/DHiSS_v2_vitstr_base_th90
"""
import argparse, logging
from pathlib import Path
import torch
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import ocr_inference as oi


def build_pretrained_ocr(reco_arch, detector, gpu):
    """ocr_predictor con reconocedor PRETRAINED de doctr (para texto impreso moderno)."""
    m = ocr_predictor(det_arch=detector, reco_arch=reco_arch, det_bs=1, reco_bs=1,
                      assume_straight_pages=False, straighten_pages=False,
                      export_as_straight_boxes=True, preserve_aspect_ratio=True,
                      symmetric_pad=True, detect_orientation=False, detect_language=False,
                      disable_crop_orientation=False, disable_page_orientation=False,
                      resolve_lines=True, resolve_blocks=False, paragraph_break=0.035,
                      pretrained=True).eval()
    m.to(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    return m

logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")
oi.logger = logging.getLogger("gen_anchors_img")


def anchor_from_image(img_path, ocr_model, threshold=0.90):
    """Replica get_anchor_with_ocr_model pero para una imagen (no PDF)."""
    with torch.no_grad():
        doc = DocumentFile.from_images(str(img_path))
        result = ocr_model(doc)
        page = result.pages[0]
        page_result = f"Page dimensions: {page.dimensions[1]/2:.1f}x{page.dimensions[0]/2:.1f}\n"
        y, x = page.dimensions
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    if word.confidence > threshold:
                        ((xmin, ymin), (xmax, ymax)) = word.geometry
                        wg = ((int(xmin*x), int(ymin*y)), (int(xmax*x), int(ymax*y)))
                        (xmin_px, ymin_px), (xmax_px, ymax_px) = wg
                        mc = ((xmin_px+xmax_px)//2, (ymin_px+ymax_px)//2)
                        mc = (mc[0], page.dimensions[0]-mc[1])
                        mc = (int(mc[0]/2), int(mc[1]/2))
                        page_result += f"[{mc[0]}x{mc[1]}]{word.value}\n"
        del doc, result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return page_result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--reco-model", default="DHiSS_v2_vitstr_base", choices=list(oi.SUPPORTED_MODELS.keys()))
    ap.add_argument("--pretrained-arch", default=None, choices=["vitstr_base", "parseq"],
                    help="Si se da, usa el reconocedor PRETRAINED de doctr (ignora --reco-model DHiSS)")
    ap.add_argument("--detector-model", default="db_resnet50")
    ap.add_argument("--ocr-threshold", type=float, default=0.90)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--shard", default="0/1", help="i/n para paralelizar en CPU")
    a = ap.parse_args()

    oi.CONFIG["gpu_id"] = a.gpu
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    if a.pretrained_arch:
        print(f"Init OCR PRETRAINED (reco={a.pretrained_arch}, det={a.detector_model}, gpu={a.gpu})...", flush=True)
        ocr_model = build_pretrained_ocr(a.pretrained_arch, a.detector_model, a.gpu)
    else:
        print(f"Init OCR (reco={a.reco_model}, det={a.detector_model}, gpu={a.gpu})...", flush=True)
        ocr_model = oi.initialize_ocr_model(reco=a.reco_model, detector=a.detector_model)

    pngs = sorted(Path(a.image_dir).glob("*.png"))
    si, sn = (int(x) for x in a.shard.split("/"))
    pngs = [pngs[j] for j in range(len(pngs)) if j % sn == si]
    print(f"shard {si}/{sn}: {len(pngs)} imagenes -> {out}", flush=True)
    for png in pngs:
        of = out / f"{png.stem}.txt"
        if of.exists() and not a.overwrite:
            continue
        try:
            anc = anchor_from_image(png, ocr_model, a.ocr_threshold)
        except Exception as e:
            print(f"  ERROR {png.name}: {e}", flush=True); anc = ""
        of.write_text(anc, encoding="utf-8")
        print(f"  {png.stem} -> {len(anc)} chars", flush=True)
    print("LISTO", flush=True)


if __name__ == "__main__":
    main()
