#!/usr/bin/env python3
"""Reconstruye N paginas de Rodrigo apilando verticalmente los crops de linea de
cada pagina (Rodrigo_<pag>_<linea>.png), del split test. GT = transcripciones unidas.
NOTA: reconstruccion aproximada (la version pagina-completa esta gated en PRHLT)."""
import argparse
from collections import defaultdict
from pathlib import Path
from PIL import Image

ap=argparse.ArgumentParser()
ap.add_argument("--dl", required=True, help="carpeta con images/ text/ partitions/")
ap.add_argument("--n", type=int, default=5)
ap.add_argument("--out-img", default="smoke_rodrigo/page_images/2048")
ap.add_argument("--out-gt", default="smoke_rodrigo/gt")
ap.add_argument("--max-longest", type=int, default=2048)
a=ap.parse_args()
Path(a.out_img).mkdir(parents=True, exist_ok=True); Path(a.out_gt).mkdir(parents=True, exist_ok=True)

trans={}
for ln in open(f"{a.dl}/text/transcriptions.txt", encoding="utf-8"):
    parts=ln.rstrip("\n").split(" ",1)
    if len(parts)==2: trans[parts[0]]=parts[1]
test=[l.strip() for l in open(f"{a.dl}/partitions/test.txt") if l.strip()]
# agrupar por pagina
pages=defaultdict(list)
for lid in test:
    pg="_".join(lid.split("_")[:2])  # Rodrigo_00416
    pages[pg].append(lid)
for pg in pages: pages[pg].sort()

def clean(t):  # '_' se usa como separador en Rodrigo -> espacio
    return t.replace("_"," ").strip()

i=0
for pg, lids in sorted(pages.items()):
    if i>=a.n: break
    imgs=[]; gts=[]
    ok=True
    for lid in lids:
        ip=Path(a.dl)/"images"/f"{lid}.png"
        if not ip.exists() or lid not in trans: ok=False; break
        imgs.append(Image.open(ip).convert("RGB")); gts.append(clean(trans[lid]))
    if not ok or not imgs: continue
    W=max(im.width for im in imgs); H=sum(im.height for im in imgs)+ (len(imgs)-1)*8
    canvas=Image.new("RGB",(W,H),(255,255,255)); y=0
    for im in imgs:
        canvas.paste(im,(0,y)); y+=im.height+8
    # escalar lado mayor a max_longest
    scale=a.max_longest/max(canvas.size)
    if scale<1: canvas=canvas.resize((int(W*scale),int(H*scale)))
    doc=f"RDG{i+1:02d}"
    canvas.save(f"{a.out_img}/{doc}_p01.png")
    Path(f"{a.out_gt}/{doc}_GT.txt").write_text("\n".join(gts), encoding="utf-8")
    print(f"{doc}: {pg} lineas={len(imgs)} img={canvas.size} gt_chars={sum(len(g) for g in gts)}", flush=True)
    i+=1
print(f"LISTO: {i} paginas reconstruidas", flush=True)
