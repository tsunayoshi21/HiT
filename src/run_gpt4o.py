#!/usr/bin/env python3
"""
HiT on GPT-4o (OpenAI API).

Same method as the local backbones, driven over the OpenAI API. For every PDF page we
build a prompt and send it with the page image:

    --mode baseline  (without HiT) : standard anchor (olmOCR's get_anchor_text)
    --mode append    (with HiT)    : DHiSS coordinate anchors (fine-tuned recognizer)

The anchor engine is imported from ocr_inference.py. Provide the key via
`export OPENAI_API_KEY=...` or `--openai-api-key`. No key is stored in this repository.

Example:
    python src/run_gpt4o.py --pdf-folder data/pdfs --mode baseline --output-folder gpt4o_baseline
    python src/run_gpt4o.py --pdf-folder data/pdfs --mode append \
        --reco-model DHiSS_v2_vitstr_base --ocr-threshold 0.90 --output-folder gpt4o_append

Author: Anonymous (double-blind submission)
"""
import argparse
import json
import os
from pathlib import Path

from openai import OpenAI
from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text

import ocr_inference as oi

JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "ocr_response", "strict": True,
        "schema": {
            "type": "object",
            "properties": {"natural_text": {"type": "string",
                           "description": "The extracted text from the document image."}},
            "required": ["natural_text"], "additionalProperties": False,
        },
    },
}


def transcribe_page(client, model, image_b64, prompt, max_tokens, temperatures):
    """One page via OpenAI, with a temperature fallback if the JSON does not parse."""
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
    ]}]
    for t in temperatures:
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens,
                temperature=t, response_format=JSON_SCHEMA,
            )
            nat = json.loads(resp.choices[0].message.content).get("natural_text")
            return nat if isinstance(nat, str) else ("" if nat is None else str(nat))
        except json.JSONDecodeError:
            continue
        except Exception as e:  # API / rate errors -> try next temperature
            print(f"    OpenAI error (temp={t}): {str(e)[:120]}", flush=True)
            continue
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-folder", default="data/pdfs")
    ap.add_argument("--mode", choices=["baseline", "append"], required=True)
    ap.add_argument("--output-parent", default="results")
    ap.add_argument("--output-folder", required=True)
    ap.add_argument("--reco-model", default="DHiSS_v2_vitstr_base",
                    choices=list(oi.SUPPORTED_MODELS.keys()))
    ap.add_argument("--detector-model", default="db_resnet50")
    ap.add_argument("--ocr-threshold", type=float, default=0.90)
    ap.add_argument("--openai-model", default="gpt-4o")
    ap.add_argument("--openai-api-key", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--target-dim", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperatures", type=float, nargs="+", default=[0.1, 0.4, 0.8])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("No OpenAI API key. Set OPENAI_API_KEY or use --openai-api-key.")
    client = OpenAI(api_key=key)

    # The DHiSS recognizer is only needed for the with-HiT (append) condition.
    ocr_model = None
    if args.mode == "append":
        oi.CONFIG["gpu_id"] = args.gpu
        ocr_model = oi.initialize_ocr_model(reco=args.reco_model, detector=args.detector_model)

    out_dir = Path(args.output_parent) / args.output_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(Path(args.pdf_folder).glob("*.pdf"))
    print(f"{len(pdfs)} PDFs | mode={args.mode} | model={args.openai_model}", flush=True)
    for pdf in pdfs:
        doc = pdf.stem
        out_f = out_dir / f"{doc}_{args.output_folder}.txt"
        if out_f.exists() and not args.overwrite:
            print(f"[{doc}] skip"); continue
        n_pages = oi.get_pdf_page_count(str(pdf))
        texts = []
        for pg in range(1, n_pages + 1):
            image_b64 = render_pdf_to_base64png(str(pdf), pg, target_longest_image_dim=args.target_dim)
            if args.mode == "append":
                anchor = oi.get_anchor_with_ocr_model(str(pdf), pg, ocr_model, threshold=args.ocr_threshold)
            else:
                anchor = get_anchor_text(str(pdf), pg, pdf_engine="pdfreport", target_length=4000)
            prompt = build_finetuning_prompt(anchor)
            txt = transcribe_page(client, args.openai_model, image_b64, prompt,
                                  args.max_tokens, args.temperatures)
            texts.append(txt)
            print(f"  {doc} p{pg:02d} -> {len(txt)} chars", flush=True)
        out_f.write_text("\n".join(texts), encoding="utf-8")
        print(f"[{doc}] saved", flush=True)


if __name__ == "__main__":
    main()
