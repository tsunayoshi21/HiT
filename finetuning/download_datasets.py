#!/usr/bin/env python3
"""
Download the DHiSS / DHiSS+ recognition datasets from the Hugging Face Hub.

    python finetuning/download_datasets.py --unzip                 # both
    python finetuning/download_datasets.py --which DHiSS+ --unzip   # just DHiSS+ (v2)

With --unzip the archives are extracted next to this script, giving the folders the
trainer expects:  DHiss_Dataset_v1_corrected_full/  (DHiSS)  and  DHiss_Dataset_v2/  (DHiSS+).
Datasets: https://huggingface.co/datasets/tsunayoshi21/DHiSS-datasets
"""
import argparse
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "tsunayoshi21/DHiSS-datasets"
FILES = {"DHiSS": "DHiSS.zip", "DHiSS+": "DHiSS_plus.zip"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["DHiSS", "DHiSS+", "both"], default="both")
    ap.add_argument("--unzip", action="store_true", help="extract the archive(s) after download")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    names = ["DHiSS", "DHiSS+"] if args.which == "both" else [args.which]
    for n in names:
        print(f"downloading {n} -> {FILES[n]} ...", flush=True)
        z = hf_hub_download(REPO_ID, FILES[n], repo_type="dataset", local_dir=str(here))
        if args.unzip:
            with zipfile.ZipFile(z) as f:
                f.extractall(here)
            print(f"  unzipped -> {here}")
    print("done.")


if __name__ == "__main__":
    main()
