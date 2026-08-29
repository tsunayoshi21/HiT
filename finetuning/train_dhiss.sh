#!/bin/bash
# Fine-tune doctr recognizers on the DHiSS dataset -> anchor recognizers.
#
# Trainer: doctr references/recognition/train_pytorch.py  (doctr fork v0.11.1a0).
#   Place that script next to this file (it is not shipped here). See README.md.
# Dataset: doctr recognition format  ->  $DATA/{train,val}/ = word-crop images + labels.json
#
# Produces ../ocr_weights/DHiSS_finetuning_v2_<arch>_10.pt
set -e

EPOCHS=10
LR=0.001
DATA=DHiss_Dataset_v2               # DHiSS+ (v2).  DHiSS -> DHiss_Dataset_v1_corrected_full
TAG=v2                              # matches the DATA version (v2 / v1_corrected_full)
OUT=../ocr_weights

declare -A BATCH=( ["vitstr_base"]=256 ["parseq"]=384 )

for model in vitstr_base parseq; do
  python train_pytorch.py "$model" \
    --train_path "$DATA/train" \
    --val_path   "$DATA/val" \
    --epochs "$EPOCHS" \
    --batch_size "${BATCH[$model]}" \
    --lr "$LR" \
    --output_dir "$OUT" \
    --pretrained \
    --vocab spanish \
    --max-chars 50 \
    --name "DHiSS_finetuning_${TAG}_${model}_${EPOCHS}"
done
