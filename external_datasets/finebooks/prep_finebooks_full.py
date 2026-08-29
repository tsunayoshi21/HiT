#!/usr/bin/env python3
"""Materializa TODAS las páginas latin-script de finebooks (texto + sparse) para el
test COMPLETO. GT del campo text o del PAGE-XML. Marca sparse (GT corto). Resumible."""
import re, json
from pathlib import Path
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image
Image.MAX_IMAGE_PIXELS=None

OUTI=Path("smoke_finebooks_full/page_images/2048"); OUTG=Path("smoke_finebooks_full/gt")
OUTI.mkdir(parents=True, exist_ok=True); OUTG.mkdir(parents=True, exist_ok=True)
def gt_from_xml(xml):
    return "\n".join(m.strip() for m in re.findall(r"<Unicode>(.*?)</Unicode>", xml or "", flags=re.S) if m.strip())

p=hf_hub_download("finebooks/bhl-impact-gt","metadata.parquet",repo_type="dataset")
tab=pq.read_table(p, columns=["file_name","BarCode","text","xml"]).to_pydict()
N=len(tab["file_name"]); manifest={}
i=0; done=0
for r in range(N):
    bc=tab["BarCode"][r]
    if "russ" in bc: continue          # excluir cirílico
    g=(tab["text"][r] or "").strip()
    if len(g)<700:
        gx=gt_from_xml(tab["xml"][r]);
        if len(gx)>len(g): g=gx
    doc=f"P{i+1:04d}"; i+=1
    imgp=OUTI/f"{doc}_p01.png"; gtp=OUTG/f"{doc}_GT.txt"
    manifest[doc]={"barcode":bc,"gt_chars":len(g),"sparse":len(g)<100}
    if imgp.exists() and gtp.exists():   # resume
        done+=1; continue
    try:
        ip=hf_hub_download("finebooks/bhl-impact-gt", tab["file_name"][r], repo_type="dataset")
        img=Image.open(ip).convert("RGB")
    except Exception as e:
        print("img fail", str(e)[:40]); continue
    sc=2048/max(img.size)
    if sc<1: img=img.resize((int(img.width*sc),int(img.height*sc)))
    img.save(imgp); gtp.write_text(g, encoding="utf-8"); done+=1
    if done%100==0: print(f"  {done} páginas...", flush=True)
json.dump(manifest, open("smoke_finebooks_full/manifest.json","w"), ensure_ascii=False)
ntext=sum(1 for v in manifest.values() if v["gt_chars"]>=700)
nsparse=sum(1 for v in manifest.values() if v["sparse"])
print(f"LISTO: {i} páginas latin-script | texto(>=700)={ntext} | sparse(<100)={nsparse}")
