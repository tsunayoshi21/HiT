#!/usr/bin/env python
"""
Runner vLLM offline para olmOCR-2 (Qwen2.5-VL). Ejecuta VARIAS condiciones
(baseline / append con distintos anchor-dirs) en un solo batch por GPU, con
sharding por doc. Mucho más rápido que run_vlm.py (transformers batch-1).

Greedy determinista (temperature=0). Prompt IDÉNTICO a run_vlm.py.

Uso:
  python run_vlm_vllm.py --model allenai/olmOCR-2-7B-1025 \
      --image-dir smoke_finebooks_full/page_images/2048 \
      --output-parent smoke_finebooks_full/results --shard 0/3 --gpu 0 \
      --cond baseline::  \
      --cond hit_onnxtr:append:smoke_finebooks_full/anchors/onnxtr_parseqML_th50 \
      --cond hit_surya:append:smoke_finebooks_full/anchors/surya_th40
"""
import argparse, time, os, sys
# CUDA_VISIBLE_DEVICES debe fijarse ANTES de importar torch/vllm.
if "--gpu" in sys.argv:
    _g = sys.argv[sys.argv.index("--gpu") + 1]
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = _g
# Blackwell (sm_120): el sampler FlashInfer aborta (JIT no reconoce sm_120).
# Desactivarlo hace que vLLM use el sampler nativo de PyTorch -> corre en GPU1.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

INSTR = ("Transcribe fielmente TODO el texto de esta imagen de un documento historico, "
         "en texto plano y respetando el orden de lectura natural. Devuelve unicamente la "
         "transcripcion, sin comentarios, sin formato markdown y sin repetir texto.")


def build_prompt(mode, anchor):
    if mode == "append" and anchor:
        return (INSTR + "\n\nComo referencia, aqui tienes un OCR previo de la pagina "
                "(puede tener errores; usalo solo como apoyo):\n"
                f"RAW_TEXT_START\n{anchor}\nRAW_TEXT_END")
    return INSTR


def list_pages(image_dir):
    docs = {}
    for png in sorted(Path(image_dir).glob("*_p*.png")):
        doc, pg = png.stem.rsplit("_p", 1)
        docs.setdefault(doc, []).append((int(pg), png))
    for d in docs:
        docs[d].sort()
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/olmOCR-2-7B-1025")
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--output-parent", required=True)
    ap.add_argument("--cond", action="append", required=True,
                    help="label:mode:anchor_dir  (mode = '' o 'append')")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    conds = []
    for c in args.cond:
        label, mode, adir = (c.split(":", 2) + ["", ""])[:3]
        conds.append((label, mode, adir))

    docs = list_pages(args.image_dir)
    i, n = (int(x) for x in args.shard.split("/"))
    out_root = Path(args.output_parent)
    # Sharding DESPUES del skip: repartir solo los docs que tienen trabajo pendiente
    # (alguna condicion sin archivo de salida) entre las n GPUs -> uso parejo.
    def doc_needs_work(doc):
        if args.overwrite:
            return True
        return any(not (out_root / label / f"{doc}_{label}.txt").exists()
                   for label, _, _ in conds)
    todo = [(d, p) for d, p in sorted(docs.items()) if doc_needs_work(d)]
    todo = [todo[j] for j in range(len(todo)) if j % n == i]
    docs = dict(todo)
    print(f"shard {i}/{n}: {len(docs)} docs con trabajo | condiciones: {[c[0] for c in conds]}", flush=True)

    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    def make_prompt_text(user_text):
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": user_text}]}]
        return proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Construir todos los requests (todas las condiciones del shard)
    requests = []          # dicts para vLLM
    meta = []              # (label, doc, out_path) por request; multipágina -> 1 request/página
    out_dirs = {}
    for label, mode, adir in conds:
        od = Path(args.output_parent) / f"{label}"
        od.mkdir(parents=True, exist_ok=True)
        out_dirs[label] = od

    # cache de imágenes por doc/página
    for doc, pages in docs.items():
        for label, mode, adir in conds:
            out_f = out_dirs[label] / f"{doc}_{label}.txt"
            if out_f.exists() and not args.overwrite:
                continue
            for pg, png in pages:
                anchor = ""
                if mode == "append" and adir:
                    af = Path(adir) / f"{doc}_p{pg:02d}.txt"
                    anchor = af.read_text(encoding="utf-8") if af.exists() else ""
                user_text = build_prompt(mode, anchor)
                img = Image.open(png).convert("RGB")
                requests.append({"prompt": make_prompt_text(user_text),
                                 "multi_modal_data": {"image": img}})
                meta.append((label, doc, pg, out_f))

    print(f"total requests: {len(requests)}", flush=True)
    if not requests:
        print("nada que hacer (todo existe)", flush=True)
        return

    llm = LLM(model=args.model, trust_remote_code=True, dtype="bfloat16",
              limit_mm_per_prompt={"image": 1},
              gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=args.max_model_len, enforce_eager=True)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    t0 = time.time()
    outs = llm.generate(requests, sp)
    dt = time.time() - t0
    print(f"generate: {len(outs)} en {dt:.0f}s ({len(outs)/dt:.2f} req/s)", flush=True)

    # agrupar salidas por doc/condición (multipágina -> unir por \n)
    from collections import defaultdict
    bucket = defaultdict(list)   # (label,doc,out_f) -> [(pg, text)]
    for (label, doc, pg, out_f), o in zip(meta, outs):
        bucket[(label, doc, out_f)].append((pg, o.outputs[0].text.strip()))
    for (label, doc, out_f), lst in bucket.items():
        lst.sort()
        out_f.write_text("\n".join(t for _, t in lst), encoding="utf-8")
    print(f"escritos {len(bucket)} archivos", flush=True)
    print("VLLM_RUN_DONE", flush=True)


if __name__ == "__main__":
    main()
