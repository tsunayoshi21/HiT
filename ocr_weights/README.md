# Anchor recognizers (DHiSS / DHiSS+)

The four fine-tuned doctr recognizers that produce the HiT anchors are hosted on the
Hugging Face Hub and are **not** committed to this repository (they are large `.pt` files).

**Download them here:**

```bash
python ocr_weights/download.py          # all four (~840 MB)
# or just the one used for the paper's anchors:
python ocr_weights/download.py --only DHiSS_v2_vitstr_base
```

| Checkpoint file | Registry key (`--reco-model`) | Dataset | Arch |
|---|---|---|---|
| `DHiSS_finetuning_v2_vitstr_base_10.pt` | `DHiSS_v2_vitstr_base` | DHiSS+ (v2) | ViTSTR |
| `DHiSS_finetuning_v2_parseq_10.pt` | `DHiSS_v2_parseq` | DHiSS+ (v2) | ParSeq |
| `DHiSS_finetuning_v1_corrected_full_vitstr_base_10.pt` | `DHiSS_v1_corrected_vitstr_base` | DHiSS (v1-corrected) | ViTSTR |
| `DHiSS_finetuning_v1_corrected_full_parseq_10.pt` | `DHiSS_v1_corrected_parseq` | DHiSS (v1-corrected) | ParSeq |

`DHiSS_v2_vitstr_base` (**DHiSS+ · ViTSTR**) produced every anchor reported in the paper.
The registry is defined in `../src/ocr_inference.py` (`SUPPORTED_MODELS`); see `../finetuning/`
for how these were trained.

Source: `https://huggingface.co/tsunayoshi21/HiT-DHiSS-recognizers`
