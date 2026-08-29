#!/usr/bin/env python3
"""
Download the DHiSS / DHiSS+ recognizer checkpoints into this folder.

    python ocr_weights/download.py            # all four
    python ocr_weights/download.py --only DHiSS_v2_vitstr_base   # just the one used in the paper

The checkpoints are hosted on the Hugging Face Hub (public). No token needed.
"""
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "tsunayoshi21/HiT-DHiSS-recognizers"

FILES = {
    "DHiSS_v2_vitstr_base":            "DHiSS_finetuning_v2_vitstr_base_10.pt",             # DHiSS+ · ViTSTR (paper)
    "DHiSS_v2_parseq":                 "DHiSS_finetuning_v2_parseq_10.pt",                  # DHiSS+ · ParSeq
    "DHiSS_v1_corrected_vitstr_base":  "DHiSS_finetuning_v1_corrected_full_vitstr_base_10.pt",  # DHiSS · ViTSTR
    "DHiSS_v1_corrected_parseq":       "DHiSS_finetuning_v1_corrected_full_parseq_10.pt",       # DHiSS · ParSeq
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(FILES), help="download a single recognizer")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    names = [args.only] if args.only else list(FILES)
    for name in names:
        fname = FILES[name]
        print(f"downloading {name} -> {fname} ...", flush=True)
        hf_hub_download(repo_id=REPO_ID, filename=fname, local_dir=str(here))
    print(f"done -> {here}")


if __name__ == "__main__":
    main()
