"""FinanceBench: 84 冊の 10-K のページ本文と、ページ単位の埋め込みを作る (ベクトル RAG の土台)。"""
import json, os, sys, time
os.environ.setdefault("HF_HUB_CACHE", "/data/openvons/choice_spec/hf_cache")
import numpy as np, pymupdf
FB = "/data/jev-rag/financebench"; os.makedirs(f"{FB}/pages", exist_ok=True); os.makedirs(f"{FB}/emb", exist_ok=True)
rows = [json.loads(l) for l in open(f"{FB}/repo/data/financebench_open_source.jsonl")]
docs = sorted({r["doc_name"] for r in rows})
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda"); m.max_seq_length = 2048
t0 = time.time(); tot = 0
for d in docs:
    pj = f"{FB}/pages/{d}.json"; ej = f"{FB}/emb/{d}.npy"
    if os.path.exists(ej):
        continue
    with pymupdf.open(f"{FB}/repo/pdfs/{d}.pdf") as f:
        pages = [p.get_text() for p in f]
    json.dump(pages, open(pj, "w"))
    emb = m.encode([p[:6000] or " " for p in pages], batch_size=8, normalize_embeddings=True, show_progress_bar=False)
    np.save(ej, emb.astype(np.float32)); tot += len(pages)
    print(f"{d:40s} {len(pages):4d} pages  ({tot} total, {time.time()-t0:.0f}s)", flush=True)
