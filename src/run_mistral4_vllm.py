#!/usr/bin/env python3
"""
Cliente para Mistral-Small-4-119B-NVFP4 servido localmente por vLLM (OpenAI-compatible).
Mistral-4 es un VLM instruible (multimodal). Se trata como gpt-4o/Qwen:
  * baseline: instruccion de transcripcion + imagen.
  * append  : + anchors DHiSS al final del prompt.
Se usa reasoning_effort="none" para evitar el modo de razonamiento (OCR limpio).

Uso: (con el server vLLM ya arriba en --base-url)
    python run_mistral4_vllm.py --mode baseline --output-folder mistral4_baseline
    python run_mistral4_vllm.py --mode append   --output-folder mistral4_append \
        --dhiss-anchor-dir anchors/DHiSS_v2_vitstr_base_th90
"""
import argparse, base64, os, sys, time
from pathlib import Path

INSTR = ("Transcribe fielmente TODO el texto de esta imagen de un documento historico, "
         "en texto plano y respetando el orden de lectura natural. Devuelve unicamente la "
         "transcripcion, sin comentarios, sin markdown y sin repetir texto.")

def list_pages(image_dir):
    docs = {}
    for png in sorted(Path(image_dir).glob("*_p*.png")):
        doc, pg = png.stem.rsplit("_p", 1)
        docs.setdefault(doc, []).append((int(pg), png))
    for d in docs: docs[d].sort()
    return docs

def build_prompt(mode, anchor):
    if mode == "append" and anchor:
        return (INSTR + "\n\nComo referencia, un OCR previo de la pagina (puede tener errores):\n"
                f"RAW_TEXT_START\n{anchor}\nRAW_TEXT_END")
    return INSTR

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "append"], required=True)
    ap.add_argument("--model", default="mistralai/Mistral-Small-4-119B-2603-NVFP4")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--image-dir", default="page_images/2048")
    ap.add_argument("--dhiss-anchor-dir", default="anchors/DHiSS_v2_vitstr_base_th90")
    ap.add_argument("--output-parent", default="results")
    ap.add_argument("--output-folder", required=True)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--limit-docs", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")

    docs = list_pages(args.image_dir)
    if args.limit_docs:
        docs = dict(list(docs.items())[:args.limit_docs])
    out_dir = Path(args.output_parent) / args.output_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    anchor_dir = Path(args.dhiss_anchor_dir)

    def b64(png): return base64.b64encode(Path(png).read_bytes()).decode()

    def infer(png, anchor):
        prompt = build_prompt(args.mode, anchor)
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(png)}"}}]}]
        for attempt in range(4):
            try:
                r = client.chat.completions.create(
                    model=args.model, messages=msgs, max_tokens=args.max_tokens,
                    temperature=0.0, extra_body={"reasoning_effort": "none"})
                return r.choices[0].message.content or ""
            except Exception as e:
                if attempt == 3:
                    print(f"    error final: {e}", flush=True); return ""
                time.sleep(2 ** attempt)

    t0 = time.time()
    for doc, pages in docs.items():
        out_f = out_dir / f"{doc}_{args.output_folder}.txt"
        if out_f.exists() and not args.overwrite:
            print(f"[{doc}] skip"); continue
        texts = []
        for pg, png in pages:
            anchor = ""
            if args.mode == "append":
                af = anchor_dir / f"{doc}_p{pg:02d}.txt"
                anchor = af.read_text(encoding="utf-8") if af.exists() else ""
            t = infer(png, anchor)
            texts.append(t)
            print(f"  {doc} p{pg:02d} -> {len(t)} chars", flush=True)
        out_f.write_text("\n".join(texts), encoding="utf-8")
        print(f"[{doc}] guardado", flush=True)
    print(f"LISTO en {time.time()-t0:.0f}s -> {out_dir}", flush=True)

if __name__ == "__main__":
    main()
