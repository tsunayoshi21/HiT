# Fine-tuning the DHiSS anchor recognizers

This folder documents how the **anchor recognizers** in `../ocr_weights/` were produced:
a doctr text-recognition model fine-tuned on the **DHiSS** dataset. The recognizer's
word transcriptions become the `[XxY]word` anchors that HiT injects into the VLM prompt.

We ship four recognizers (two datasets × two architectures):

| Weight file (`../ocr_weights/`) | Dataset | Arch | Name in the paper |
|---|---|---|---|
| `DHiSS_finetuning_v1_corrected_full_vitstr_base_10.pt` | DHiSS (v1-corrected) | ViTSTR | **DHiSS** · ViTSTR |
| `DHiSS_finetuning_v1_corrected_full_parseq_10.pt`      | DHiSS (v1-corrected) | ParSeq | **DHiSS** · ParSeq |
| `DHiSS_finetuning_v2_vitstr_base_10.pt`                | DHiSS+ (v2)          | ViTSTR | **DHiSS+** · ViTSTR |
| `DHiSS_finetuning_v2_parseq_10.pt`                     | DHiSS+ (v2)          | ParSeq | **DHiSS+** · ParSeq |

`DHiSS+ · ViTSTR` is the recognizer used for all HiT anchors in the paper.

## Dataset format (doctr recognition)
Each dataset is a folder with a `train/` and a `val/` split. Each split is a directory
of **word-crop images** plus a `labels.json` mapping `{crop_filename: transcription}`
— the standard doctr recognition format. Sizes of DHiSS+ (v2), as loaded during training:

| Split | Word-crop samples |
|---|---|
| train | **148 402** |
| val   | **37 101** |
| **total** | **185 503** |

## Trainer
The trainer is doctr's standard recognition reference script, **included here**:

- `train_pytorch.py` — doctr `references/recognition/train_pytorch.py`
- `utils.py` — its helper module (imported by the trainer)
- `evaluate_pytorch.py` — recognition evaluation script (optional)

Recovered from the doctr fork **`v0.11.0-17-g903e9114d` (0.11.1a0)**, © Mindee, Apache-2.0.
They need `doctr` (python-doctr) installed — use the `env_render_and_anchors` environment
from the repository root, which already provides it.

**Hyperparameters (as used):** `--epochs 10 --lr 0.001` (adam, cosine schedule),
`--vocab spanish --max-chars 50 --pretrained`, input size 32; batch size **256**
(ViTSTR) / **384** (ParSeq).

## Run
```bash
# from this folder, with the DHiSS dataset in DHiss_Dataset_v2/{train,val}/
bash train_dhiss.sh            # trains DHiSS+ (v2), both archs -> ../ocr_weights/
# for the DHiSS (v1-corrected) recognizers, set DATA=DHiss_Dataset_v1_corrected_full in train_dhiss.sh
```
