#!/usr/bin/env python
"""
Genera anchors HiT usando onnxtr (detector + recognizer pretrained, p.ej. parseq
multilingual) en vez de los fine-tunings DHiSS. Formato de salida IDENTICO al de
ocr_inference.get_anchor_with_ocr_model (Page dimensions + [XxY]word), para que
run_vlm.py lo consuma con --dhiss-anchor-dir.

Uso:
  python gen_onnxtr_anchors.py --image-dir smoke_finebooks/page_images/2048 \
      --out-dir smoke_finebooks/anchors/onnxtr_parseqML_th50 \
      --reco-hub Felix92/onnxtr-parseq-multilingual-v1 --det db_resnet50 --threshold 0.5
"""
import argparse
from pathlib import Path
from onnxtr.io import DocumentFile
from onnxtr.models import ocr_predictor, from_hub
from onnxtr.models.engine import EngineConfig


def anchor_for_page(page, threshold):
    y, x = page.dimensions  # (H, W)
    out = f"Page dimensions: {page.dimensions[1]/2:.1f}x{page.dimensions[0]/2:.1f}\n"
    n = 0
    for block in page.blocks:
        for line in block.lines:
            for word in line.words:
                if word.confidence > threshold:
                    ((xmin, ymin), (xmax, ymax)) = word.geometry
                    xmin_px, ymin_px = int(xmin * x), int(ymin * y)
                    xmax_px, ymax_px = int(xmax * x), int(ymax * y)
                    cx = (xmin_px + xmax_px) // 2
                    cy = (ymin_px + ymax_px) // 2
                    cy = page.dimensions[0] - cy       # flip Y
                    cx, cy = int(cx / 2), int(cy / 2)
                    out += f"[{cx}x{cy}]{word.value}\n"
                    n += 1
    return out, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--reco-hub", default="Felix92/onnxtr-parseq-multilingual-v1")
    ap.add_argument("--reco-arch", default=None,
                    help="si se da, usa arch pretrained en vez de --reco-hub (p.ej. parseq)")
    ap.add_argument("--det", default="db_resnet50")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--glob", default="*_p*.png")
    ap.add_argument("--shard", default="0/1", help="i/n para paralelizar en CPU")
    ap.add_argument("--threads", type=int, default=0, help="intra_op_num_threads (0=default)")
    args = ap.parse_args()

    import onnxruntime as ort
    so = None
    if args.threads:
        so = ort.SessionOptions()
        so.intra_op_num_threads = args.threads
        so.inter_op_num_threads = 1
    gpu_cfg = EngineConfig(providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                           session_options=so)
    if args.reco_arch:
        reco = args.reco_arch
        print(f"reco arch pretrained: {reco}")
    else:
        reco = from_hub(args.reco_hub, engine_cfg=gpu_cfg)
        print(f"reco from_hub: {args.reco_hub} (vocab={len(reco.cfg.get('vocab',''))})")

    predictor = ocr_predictor(det_arch=args.det, reco_arch=reco,
                              assume_straight_pages=True,
                              det_engine_cfg=gpu_cfg, reco_engine_cfg=gpu_cfg,
                              clf_engine_cfg=gpu_cfg)
    # verificar provider real de una sesión
    try:
        prov = predictor.det_predictor.model.runtime.get_providers()
        print(f"providers activos: {prov}")
    except Exception:
        pass

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted(Path(args.image_dir).glob(args.glob))
    i, n = (int(x) for x in args.shard.split("/"))
    imgs = [imgs[j] for j in range(len(imgs)) if j % n == i]
    print(f"shard {i}/{n}: {len(imgs)} imágenes -> {out_dir}")
    for p in imgs:
        doc_img = DocumentFile.from_images([str(p)])
        res = predictor(doc_img)
        page = res.pages[0]
        anchor, n = anchor_for_page(page, args.threshold)
        # nombre de archivo: mismo stem que la imagen (DOC_pNN.txt)
        (out_dir / f"{p.stem}.txt").write_text(anchor, encoding="utf-8")
        print(f"  {p.name} -> {n} palabras", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
