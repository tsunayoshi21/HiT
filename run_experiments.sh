#!/bin/bash
# =====================================================================
#  Reproduce the HiT experiments (Table 3 subset).
#
#  HiT is the method: inject word-level DHiSS coordinate anchors into a VLM prompt.
#  Everything below is the SAME method applied to different VLM backbones; we ship
#  the backbones for which HiT IMPROVED the transcription. For every backbone we run
#  two conditions:
#    * without HiT (baseline) : the image + a plain transcription instruction.
#    * with HiT    (append)   : the same prompt, plus the DHiSS anchors appended.
#
#  Pipeline stages (run from the repository ROOT):
#    0. render PDFs to page images         (env: env_render_and_anchors)
#    1. generate + cache the DHiSS anchors  (env: env_render_and_anchors)
#    2. run each backbone, w/o and w/ HiT   (env: env_vlm_runners / env_mistral4_vllm)
#    3. compute metrics                     (env: env_metrics)
#
#  Activate the matching conda env before each block (see README.md).
#  Adjust --gpu to your hardware.
# =====================================================================
set -e

# ---- 0. Render every PDF page to a 2048px PNG ------------------------
#   conda activate olmo_doc   (env_render_and_anchors.yml)
python src/render_pages.py --pdf-folder data/pdfs --target-dim 2048 --out-dir page_images/2048

# ---- 1. Generate + cache the DHiSS anchors (doctr DBNet + ViTSTR) ----
#   conda activate olmo_doc   (env_render_and_anchors.yml)
python src/gen_dhiss_anchors.py \
    --pdf-folder data/pdfs \
    --reco-model DHiSS_v2_vitstr_base \
    --ocr-threshold 0.90 \
    --gpu 0 \
    --out-dir anchors/DHiSS_v2_vitstr_base_th90

# ---- 2. HiT on instructable VLM backbones (without HiT + with HiT) ---
#   conda activate vlm_latest   (env_vlm_runners.yml)
#   mode baseline = without HiT ; mode append = with HiT (anchors appended)
run_backbone () {   # $1 = HF model id, $2 = short tag, $3 = extra flags
    python src/run_vlm.py --model "$1" --mode baseline --output-folder "${2}_baseline" --gpu 0 $3
    python src/run_vlm.py --model "$1" --mode append   --output-folder "${2}_append"   --gpu 0 $3
}

run_backbone allenai/olmOCR-7B-0225-preview    olmocr
run_backbone allenai/olmOCR-2-7B-1025          olmocr2
run_backbone google/gemma-4-12b-it             gemma4_12b
run_backbone google/gemma-4-26b-a4b-it         gemma4_26b
run_backbone Qwen/Qwen3-VL-30B-A3B-Instruct    qwen3vl30b
run_backbone XiaomiMiMo/MiMo-VL-7B-RL          mimovl7b     "--max-new-tokens 6000"
# Infinity-Parser2-Flash: layout-aware parser -> canonical JSON prompt + JSON->text postproc
run_backbone infly/Infinity-Parser2-Flash      infinity2flash "--task-prompt-file infinity_prompt.txt --postproc json_text"

# ---- 2b. HiT on Mistral-Small-4 (backbone served locally with vLLM) --
#   conda activate vllm_env   (env_mistral4_vllm.yml)
#   Start the server first (Blackwell / sm_120 flags shown; drop them on other GPUs):
#     CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
#     VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=TRITON_MLA \
#     vllm serve mistralai/Mistral-Small-4-119B-2603-NVFP4 \
#       --tensor-parallel-size 1 --max-model-len 32768 --gpu-memory-utilization 0.92 \
#       --attention-backend TRITON_MLA --enforce-eager \
#       --limit-mm-per-prompt '{"image": 1}' --host 127.0.0.1 --port 8100
python src/run_mistral4_vllm.py --mode baseline --output-folder mistral4_baseline --base-url http://127.0.0.1:8100/v1
python src/run_mistral4_vllm.py --mode append   --output-folder mistral4_append   --base-url http://127.0.0.1:8100/v1

# ---- 2c. HiT on GPT-4o (OpenAI API) ---------------------------------
#   conda activate olmo_doc   (env_render_and_anchors.yml)
#   src/run_gpt4o.py imports the anchor engine from src/ocr_inference.py and is
#   self-contained (renders pages + computes anchors internally; does NOT need stages 0/1).
#     --mode baseline  -> without HiT (standard anchor)
#     --mode append    -> with HiT    (DHiSS coordinate anchors)
#   GPT-4o needs an OpenAI API key — provide it via the environment (preferred):
#       export OPENAI_API_KEY="..."
#   or pass --openai-api-key on the command line. No key is stored in this repository.
python src/run_gpt4o.py --pdf-folder data/pdfs --mode baseline --output-parent results --output-folder gpt4o_baseline
python src/run_gpt4o.py --pdf-folder data/pdfs --mode append   --output-parent results \
    --reco-model DHiSS_v2_vitstr_base --ocr-threshold 0.90 --output-folder gpt4o_append

# ---- 3. Metrics ------------------------------------------------------
#   conda activate metrics   (env_metrics.yml)
python src/compute_metrics.py --results-dir results --gt-dir data/gt --out metrics.csv

# ---- 0'. (optional) get the DHiSS/DHiSS+ recognizer weights ----------
#   python ocr_weights/download.py                       # all four
#   python ocr_weights/download.py --only DHiSS_v2_vitstr_base

# ---- 4. (optional) beyond Table 3 -----------------------------------
#   * external_datasets/  — HiT validated on Rodrigo & finebooks/BHL
#                           (see external_datasets/README.md; env_external_datasets.yml)
#   * finetuning/         — how the DHiSS anchor recognizers were trained (doctr)
