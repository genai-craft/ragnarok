"""索引の埋め込みを現在の既定モデル (Engine の embed_model) で作り直す: デモ索引、データレイクの store、FinanceBench の emb。元 PDF があれば本文の無いページは画像で埋め込む。"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("HF_HUB_CACHE", "/data/openvons/choice_spec/hf_cache")
import numpy as np
from ragnarok.engine import Engine, Index
eng = Engine(); t0 = time.time(); n = 0
def redo(d):
    global n
    ix = Index.load(d)
    if ix.meta.get("embed") == eng.embed_model: return
    src = ix.meta.get("source")
    if src and src.lower().endswith(".pdf") and os.path.exists(src):
        new = eng.index_pdf(src, ix.name, tree=False); ix.emb = new.emb; ix.meta["image_pages"] = new.meta.get("image_pages", 0)
    else:
        ix.emb = eng.embed_docs(ix.pages)
    ix.meta["embed"] = eng.embed_model; ix.save(d); n += len(ix.pages)
for root in ("/data/ragnarok/demo", "/data/ragnarok/store"):
    for d in sorted(os.listdir(root)):
        if os.path.exists(f"{root}/{d}/pages.json"):
            redo(f"{root}/{d}"); print(f"{root}/{d} ({n} pages, {time.time()-t0:.0f}s)", flush=True)
FB = "/data/jev-rag/financebench"; os.makedirs(f"{FB}/emb_gemma", exist_ok=True)
for f in sorted(os.listdir(f"{FB}/pages")):
    d = f[:-5]
    if os.path.exists(f"{FB}/emb_gemma/{d}.npy"): continue
    np.save(f"{FB}/emb_gemma/{d}.npy", eng.index_pdf(f"{FB}/repo/pdfs/{d}.pdf", d, tree=False).emb); n += 1
print("done", n, f"{time.time()-t0:.0f}s")
