#!/usr/bin/env python3
"""
HiT — anchor engine (importable library).

This module is *not* a runnable script. It provides the functions the rest of the
pipeline imports to build the DHiSS coordinate anchors:

    SUPPORTED_MODELS          # DHiSS / DHiSS+ recognizer registry (ViTSTR / ParSeq)
    load_reco_model(reco)     # load a fine-tuned doctr recognizer checkpoint
    initialize_ocr_model(...) # doctr detector + fine-tuned recognizer -> OCR predictor
    get_anchor_with_ocr_model(pdf, page, ocr_model, threshold)  # -> "[XxY]word" anchors
    get_pdf_page_count(pdf)
    build_custom_ocr_prompt(anchor_text)

Used by: gen_dhiss_anchors.py, run_gpt4o.py, and the external-datasets anchor tools.
The VLM transcription itself is run by the standalone scripts (run_vlm.py,
run_mistral4_vllm.py, run_gpt4o.py) — not here.

Author: Anonymous (double-blind submission)
"""

import logging

import torch
import PyPDF2

from doctr.io import DocumentFile
from doctr.models import ocr_predictor, parseq, sar_resnet31, vitstr_base
from doctr.datasets import VOCABS

logger = logging.getLogger(__name__)

# Minimal defaults for the anchor engine (override per call or via CONFIG).
CONFIG = {
    "gpu_id": 0,
    "detector_model": "db_resnet50",
    "ocr_threshold": 0.90,
}

# ============================================================================
# Fine-tuned recognizer registry.
# DHiSS  = v1-corrected  ·  DHiSS+ = v2  ·  each as ViTSTR and ParSeq.
# Checkpoints live under ocr_weights/ (see ocr_weights/download.py).
# ============================================================================
SUPPORTED_MODELS = {
    "DHiSS_v1_parseq": {
        "model_class": parseq,
        "checkpoint": "ocr_weights/DHiSS_finetuning_parseq_10.pt",
        "max_length": 50,
    },
    "DHiSS_v1_sar_resnet31": {
        "model_class": sar_resnet31,
        "checkpoint": "ocr_weights/DHiSS_finetuning_sar_resnet31_10.pt",
        "max_length": 50,
    },
    "DHiSS_v1_vitstr_base": {
        "model_class": vitstr_base,
        "checkpoint": "ocr_weights/DHiSS_finetuning_vitstr_base_10.pt",
        "max_length": 50,
    },
    "DHiSS_v1_corrected_vitstr_base": {
        "model_class": vitstr_base,
        "checkpoint": "ocr_weights/DHiSS_finetuning_v1_corrected_full_vitstr_base_10.pt",
        "max_length": 25,
    },
    "DHiSS_v1_corrected_parseq": {
        "model_class": parseq,
        "checkpoint": "ocr_weights/DHiSS_finetuning_v1_corrected_full_parseq_10.pt",
        "max_length": None,
    },
    "DHiSS_v2_vitstr_base": {
        "model_class": vitstr_base,
        "checkpoint": "ocr_weights/DHiSS_finetuning_v2_vitstr_base_10.pt",
        "max_length": 50,
    },
    "DHiSS_v2_parseq": {
        "model_class": parseq,
        "checkpoint": "ocr_weights/DHiSS_finetuning_v2_parseq_10.pt",
        "max_length": None,  # uses default
    },
}


def load_reco_model(reco: str):
    """Load a fine-tuned doctr recognition model from SUPPORTED_MODELS."""
    if reco not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {reco}. Use one of {list(SUPPORTED_MODELS.keys())}")

    model_config = SUPPORTED_MODELS[reco]
    model_class = model_config["model_class"]
    checkpoint_path = model_config["checkpoint"]
    max_length = model_config["max_length"]

    if max_length:
        reco_model = model_class(
            pretrained=False, pretrained_backbone=False,
            vocab=VOCABS["spanish"], max_length=max_length,
        )
    else:
        reco_model = model_class(
            pretrained=False, pretrained_backbone=False, vocab=VOCABS["spanish"],
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    missing_keys, unexpected_keys = reco_model.load_state_dict(checkpoint, strict=False)
    if unexpected_keys:
        logger.warning(f"Unexpected keys in checkpoint (ignored): {unexpected_keys}")
    if missing_keys:
        logger.warning(f"Missing keys in model (random init): {missing_keys}")
    logger.info(f"Recognizer {reco} loaded from {checkpoint_path}")
    return reco_model


def initialize_ocr_model(reco: str = "DHiSS_v2_vitstr_base", detector: str = "db_resnet50"):
    """doctr detector + fine-tuned recognizer -> OCR predictor (on CONFIG['gpu_id'])."""
    logger.info(f"Initializing OCR model (reco={reco}, detector={detector})...")
    reco_model = load_reco_model(reco)
    ocr_model = ocr_predictor(
        det_arch=detector,
        reco_arch=reco_model,
        det_bs=1, reco_bs=1,
        assume_straight_pages=False, straighten_pages=False,
        export_as_straight_boxes=True, preserve_aspect_ratio=True, symmetric_pad=True,
        detect_orientation=False, detect_language=False,
        disable_crop_orientation=False, disable_page_orientation=False,
        resolve_lines=True, resolve_blocks=False, paragraph_break=0.035,
        pretrained=True,
    ).eval()
    gpu_id = CONFIG["gpu_id"]
    ocr_model.to(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    logger.info("OCR model ready.")
    return ocr_model


def get_pdf_page_count(pdf_file_path: str) -> int:
    """Number of pages in a PDF (0 on error)."""
    try:
        with open(pdf_file_path, "rb") as file:
            return len(PyPDF2.PdfReader(file).pages)
    except Exception as e:
        logger.error(f"Error reading PDF page count: {e}")
        return 0


def get_anchor_with_ocr_model(pdf_path: str, page_number: int, ocr_model, threshold: float = 0.95) -> str:
    """
    Run the fine-tuned OCR model on one PDF page and format the coordinate anchors.

    Returns a string: a `Page dimensions:` header followed by one `[XxY]word` line per
    word whose confidence exceeds `threshold` (coordinates are the word's mass-center,
    Y-flipped and halved, matching the format the VLM prompt expects).
    """
    with torch.no_grad():
        doc = DocumentFile.from_pdf(pdf_path)
        result = ocr_model([doc[page_number - 1]])
        page = result.pages[0]

        page_result = f"Page dimensions: {page.dimensions[1]/2:.1f}x{page.dimensions[0]/2:.1f}\n"
        y, x = page.dimensions

        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    if word.confidence > threshold:
                        ((xmin, ymin), (xmax, ymax)) = word.geometry
                        (xmin_px, ymin_px) = (int(xmin * x), int(ymin * y))
                        (xmax_px, ymax_px) = (int(xmax * x), int(ymax * y))
                        mass_center = ((xmin_px + xmax_px) // 2, (ymin_px + ymax_px) // 2)
                        mass_center = (mass_center[0], page.dimensions[0] - mass_center[1])
                        mass_center = (int(mass_center[0] / 2), int(mass_center[1] / 2))
                        page_result += f"[{mass_center[0]}x{mass_center[1]}]{word.value}\n"

        del doc, result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return page_result


def build_custom_ocr_prompt(anchor_text: str) -> str:
    """A plain-text OCR prompt that embeds the anchor as a reference (Spanish)."""
    return (
        "Eres un experto modelo de OCR que extrae transcripciones de imágenes de documentos. "
        "Tu tarea es generar una transcripción precisa del texto visible en la imagen proporcionada. "
        "No incluyas información adicional, solo el texto visible en la imagen.\n\n"
        "Se te proporcionará un texto de anclaje extraído de la página. Úsalo como referencia.\n\n"
        f"Texto de anclaje:\n{anchor_text}\n\n"
        "Devuelve el texto como un objeto JSON con la estructura:\n"
        "{\n  'natural_text': 'texto extraído aquí'\n}\n"
    )
