"""スキャン 48 問: OCR を一切使わない構成。Nemotron Embed のページ画像埋め込みで上位 20 頁 → Nemotron Rerank (画像のみ) で並べ替え。
  比較: 同じ候補を埋め込み順のまま / OCR を使う最良 (scan_rerank.py)。
    CUDA_VISIBLE_DEVICES=0 /data/jev-rag/venv/bin/python bench/scan_ocrfree.py [nemo1b|nemo8b]
"""
import glob, json, os, sys, time
os.environ.setdefault("HF_HUB_CACHE", "/data/openvons/choice_spec/hf_cache")
import numpy as np, pymupdf, torch
from PIL import Image
from sentence_transformers import CrossEncoder, SentenceTransformer

TAG = sys.argv[1] if len(sys.argv) > 1 else "nemo1b"; TOPN = 20
S = "/data/ragnarok/scans"; NM = f"/data/ragnarok/nemo/{TAG}"
rows = [json.loads(l) for l in open(f"{S}/qa.jsonl")]
DOCS = sorted({r["doc"] for r in rows})
PDF = {}
for f in glob.glob(f"{S}/*.pdf"):
    PDF[os.path.basename(f)[:-4].split("-")[0] if "town" in f else "hiroshima_" + os.path.basename(f).split("_")[1]] = f
I = {d: np.load(f"{NM}/scan_img/{d}.npy") for d in DOCS}
em = SentenceTransformer({"nemo1b": "nvidia/Nemotron-3.5-Embed-1B-BF16", "nemo8b": "nvidia/Nemotron-3.5-Embed-8B-BF16"}[TAG], device="cuda", trust_remote_code=True,
                         model_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa"})
qv = em.encode_query([r["question"] for r in rows], normalize_embeddings=True)
cands = [[int(i) for i in np.argsort(-(I[r["doc"]] @ v))[:TOPN]] for r, v in zip(rows, qv)]
del em; torch.cuda.empty_cache()


def img(d, p, long=1120):
    with pymupdf.open(PDF[d]) as doc:
        pg = doc[p]; z = long / max(pg.rect.width, pg.rect.height); px = pg.get_pixmap(matrix=pymupdf.Matrix(z, z))
        return Image.frombytes("RGB", (px.width, px.height), px.samples)


rr = CrossEncoder("nvidia/Nemotron-3.5-Rerank-8B-BF16", device="cuda", trust_remote_code=True, model_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa"})
t0 = time.time(); ranks = []
for r, c in zip(rows, cands):
    res = rr.rank(r["question"], [{"image": img(r["doc"], p)} for p in c], batch_size=2, show_progress_bar=False)
    ranks.append([c[x["corpus_id"]] for x in res])
dt = (time.time() - t0) / len(rows)
out = {}
for grp, sel in (("英語", lambda d: not d.startswith("hiroshima")), ("日本語", lambda d: d.startswith("hiroshima"))):
    idx = [i for i, r in enumerate(rows) if sel(r["doc"])]
    for name, R in ((f"{TAG} 画像埋め込みのみ", cands), (f"{TAG} 画像埋め込み → Rerank 画像", ranks)):
        rec = {k: float(np.mean([rows[i]["page"] in R[i][:k] for i in idx])) for k in (1, 3, 5)}
        print(f"{grp} {name:32s} recall@1 {rec[1]:.3f} @3 {rec[3]:.3f} @5 {rec[5]:.3f}", flush=True)
        out.setdefault(name, {})[grp] = rec
print(f"Rerank {dt:.1f}s/問 (20 頁の画像)")
json.dump(out, open(f"{S}/ocrfree_{TAG}.json", "w"), ensure_ascii=False, indent=1)
json.dump({str(k): {"img_top3": R[:3], "emb_top3": c[:3]} for k, (R, c) in enumerate(zip(ranks, cands))}, open(f"{S}/retrieval_ocrfree_{TAG}.json", "w"))
