#!/usr/bin/env python3
"""
Runner generico para VLMs instruibles (chat) que hacen OCR: Qwen3-VL, Gemma-4,
MiMo-VL, Kimi-VL, Janus-Pro, etc. A diferencia de DeepSeek-OCR (no instruible),
estos SIGUEN instrucciones, asi que:

  * baseline (--mode baseline): se les pide transcribir la imagen.
  * append   (--mode append)  : misma instruccion + los anchors DHiSS appendeados
                                al final del prompt (para ver si ayudan).

Consume imagenes ya renderizadas (page_images/2048) y anchors cacheados.
Salida: results_baselines/<output-folder>/DOCxx_<output-folder>.txt

Uso:
    conda activate vlm_latest
    python run_vlm.py --model Qwen/Qwen3-VL-8B-Instruct --mode baseline \
        --output-folder qwen3vl8b_baseline --gpu 1
    python run_vlm.py --model Qwen/Qwen3-VL-8B-Instruct --mode append \
        --dhiss-anchor-dir anchors/DHiSS_v2_vitstr_base_th90 \
        --output-folder qwen3vl8b_append --gpu 1
"""
import argparse, os, sys, time
from pathlib import Path

INSTR = ("Transcribe fielmente TODO el texto de esta imagen de un documento historico, "
         "en texto plano y respetando el orden de lectura natural. Devuelve unicamente la "
         "transcripcion, sin comentarios, sin formato markdown y sin repetir texto.")

def list_pages(image_dir):
    docs = {}
    for png in sorted(Path(image_dir).glob("*_p*.png")):
        doc, pg = png.stem.rsplit("_p", 1)
        docs.setdefault(doc, []).append((int(pg), png))
    for d in docs: docs[d].sort()
    return docs

def extract_json_text(s):
    """Extrae y concatena los campos 'text' de un JSON de layout (Infinity-Parser2),
    en el orden en que aparecen. Robusto a fences ```json y a preambulo de razonamiento."""
    import json, re
    # tomar el bloque JSON (primer '[' hasta el ultimo ']')
    i, j = s.find("["), s.rfind("]")
    if i == -1 or j == -1 or j <= i:
        return s  # no hay JSON; devolver tal cual
    block = s[i:j + 1]
    try:
        data = json.loads(block)
    except Exception:
        # fallback: extraer los "text": "..." por regex
        return "\n".join(re.findall(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"', block))
    texts = []
    for el in data:
        if isinstance(el, dict):
            t = el.get("text", "")
            if t:
                texts.append(t)
    return "\n".join(texts)


def strip_html_tags(s):
    import re, html
    s = re.sub(r"</(p|div|h[1-6]|li|tr|br)\s*>", "\n", s, flags=re.I)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    return html.unescape(s)


def build_prompt(mode, anchor):
    if mode == "append" and anchor:
        return (INSTR + "\n\nComo referencia, aqui tienes un OCR previo de la pagina "
                "(puede tener errores; usalo solo como apoyo):\n"
                f"RAW_TEXT_START\n{anchor}\nRAW_TEXT_END")
    return INSTR

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["baseline", "append"], required=True)
    ap.add_argument("--image-dir", default="page_images/2048")
    ap.add_argument("--dhiss-anchor-dir", default="anchors/DHiSS_v2_vitstr_base_th90")
    ap.add_argument("--output-parent", default="results")
    ap.add_argument("--output-folder", required=True)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--limit-docs", type=int, default=None)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--no-think", action="store_true",
                    help="Pasa enable_thinking=False al chat template (si el template lo soporta)")
    ap.add_argument("--task-prompt-file", default=None,
                    help="Archivo con el prompt base a usar en vez del generico (p.ej. prompt canonico JSON de Infinity)")
    ap.add_argument("--postproc", choices=["none", "json_text", "html"], default="none",
                    help="Post-proceso de la salida: json_text extrae campos 'text' del JSON; html quita tags")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    # Prompt base override (para modelos con prompt canonico propio)
    global INSTR
    if args.task_prompt_file:
        INSTR = Path(args.task_prompt_file).read_text(encoding="utf-8").strip()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from PIL import Image
    from transformers import (AutoProcessor, AutoModelForImageTextToText,
                              AutoModelForCausalLM, AutoModel)

    trc = args.trust_remote_code
    print(f"Cargando {args.model} (gpu {args.gpu}, trust_remote_code={trc})...", flush=True)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=trc)
    classes = ([AutoModelForCausalLM, AutoModel, AutoModelForImageTextToText] if trc
               else [AutoModelForImageTextToText, AutoModelForCausalLM, AutoModel])
    model = None
    for C in classes:
        try:
            model = C.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                      device_map="cuda:0", trust_remote_code=trc).eval()
            print(f"cargado con {C.__name__}", flush=True); break
        except Exception as e:
            print(f"  {C.__name__} fail: {str(e)[:90]}", flush=True)
    if model is None:
        raise SystemExit("no se pudo cargar el modelo")
    print(f"cargado en {time.time()-t0:.0f}s | MEM {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    docs = list_pages(args.image_dir)
    if args.limit_docs:
        docs = dict(list(docs.items())[:args.limit_docs])
    out_dir = Path(args.output_parent) / args.output_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    anchor_dir = Path(args.dhiss_anchor_dir)

    def infer(png, anchor):
        prompt = build_prompt(args.mode, anchor)
        img = Image.open(png).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        ct_kwargs = {"enable_thinking": False} if args.no_think else {}
        try:
            # camino generico (Qwen3-VL, Gemma-4, ...)
            inputs = processor.apply_chat_template(messages, add_generation_prompt=True,
                        tokenize=True, return_dict=True, return_tensors="pt", **ct_kwargs)
        except Exception:
            # camino en 2 pasos (Kimi-VL y procesadores antiguos): texto + processor(images=)
            text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **ct_kwargs)
            inputs = processor(images=[img], text=text, return_tensors="pt")
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        in_len = inputs["input_ids"].shape[1]
        txt = processor.batch_decode(gen[:, in_len:], skip_special_tokens=True)[0]
        # Modelos "thinking" (p.ej. MiMo-VL-RL) emiten <think>...</think>; nos quedamos con la respuesta.
        if "</think>" in txt:
            txt = txt.split("</think>")[-1]
        txt = txt.strip()
        if args.postproc == "json_text":
            txt = extract_json_text(txt)
        elif args.postproc == "html":
            txt = strip_html_tags(txt)
        return txt.strip()

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
            try:
                t = infer(png, anchor)
            except Exception as e:
                print(f"  {doc} p{pg:02d} ERROR: {e}", flush=True); t = ""
            texts.append(t)
            print(f"  {doc} p{pg:02d} -> {len(t)} chars", flush=True)
        out_f.write_text("\n".join(texts), encoding="utf-8")
        print(f"[{doc}] guardado", flush=True)
    print(f"LISTO en {time.time()-t0:.0f}s -> {out_dir}", flush=True)

if __name__ == "__main__":
    main()
