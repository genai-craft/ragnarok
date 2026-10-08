"""テキスト埋め込み + ページ画像埋め込みの両方を持ち、類似度の最大 (または重み付き和) で候補を取る方式の効果 (embeddinggemma-2)。"""
import json, os, sys, time
os.environ.setdefault("HF_HUB_CACHE", "/data/openvons/choice_spec/hf_cache")
import numpy as np, torch, pymupdf
from PIL import Image
from sentence_transformers import SentenceTransformer
FB = "/data/jev-rag/financebench"; DEMO = "/data/ragnarok/demo"; OUT = "/data/ragnarok/lake/emb_img"; os.makedirs(OUT, exist_ok=True)
fb = [json.loads(l) for l in open(f"{FB}/repo/data/financebench_open_source.jsonl")]; ya = [json.loads(l) for l in open("/data/ragnarok/edinet/qa.jsonl")]
fb_docs = sorted({r["doc_name"] for r in fb}); ya_docs = sorted({r["doc"] for r in ya})
edmeta = {os.path.basename(m["file"])[:-4]: m["file"] for m in json.load(open("/data/ragnarok/edinet/meta.json"))}
def pdf_of(d):
    if d in fb_docs: return f"{FB}/repo/pdfs/{d}.pdf"
    co_, ym = d.split("_", 1)[1].rsplit("_", 1); return next(v for k, v in edmeta.items() if k.startswith(co_ + "_" + ym))
gm = SentenceTransformer("google/embeddinggemma-2", device="cuda", model_kwargs={"torch_dtype": torch.bfloat16}); gm.max_seq_length = 2048
E_t = {d: np.load(f"{FB}/emb_gemma/{d}.npy") for d in fb_docs}; E_t.update({d: np.load(f"{DEMO}/{d}/emb.npy") for d in ya_docs})
t0 = time.time(); E_i = {}; n = 0
for d in fb_docs + ya_docs:
    f = f"{OUT}/{d}.npy"
    if os.path.exists(f): E_i[d] = np.load(f); continue
    imgs = []
    with pymupdf.open(pdf_of(d)) as doc:
        for pg in doc:
            px = pg.get_pixmap(matrix=pymupdf.Matrix(1.0, 1.0)); imgs.append(Image.frombytes("RGB", (px.width, px.height), px.samples))
    E_i[d] = gm.encode(imgs, prompt_name="document", batch_size=4, normalize_embeddings=True, show_progress_bar=False).astype(np.float32); np.save(f, E_i[d]); n += len(imgs)
    if n and n % 2000 < 300: print(f"  {n} 画像ページ {time.time()-t0:.0f}s", flush=True)
print(f"画像埋め込み {n} ページ {time.time()-t0:.0f}s")
Q = {"FinanceBench": [(r["question"], r["doc_name"], {int(e["evidence_page_num"]) for e in r["evidence"]}) for r in fb], "有報 QA": [(r["question"], r["doc"], {r["page"]}) for r in ya]}
def run(title, qs, fuse):
    qv = gm.encode([q for q, _, _ in qs], prompt_name="query", normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    rec = {k: 0 for k in (1, 5, 10, 50)}
    for (q, d, ev), v in zip(qs, qv):
        n_pg = min(len(E_t[d]), len(E_i[d])); s = fuse(E_t[d][:n_pg] @ v, E_i[d][:n_pg] @ v); order = np.argsort(-s)
        for k in rec: rec[k] += bool(ev & set(int(i) for i in order[:k]))
    n = len(qs); print(f"  {title:34s} recall@1 {rec[1]/n:.3f} @5 {rec[5]/n:.3f} @10 {rec[10]/n:.3f} @50 {rec[50]/n:.3f}", flush=True)
for name, qs in Q.items():
    print(name)
    run("テキストのみ", qs, lambda t, i: t); run("画像のみ", qs, lambda t, i: i)
    run("max(テキスト, 画像)", qs, lambda t, i: np.maximum(t, i)); run("0.7 テキスト + 0.3 画像", qs, lambda t, i: 0.7 * t + 0.3 * i); run("0.5 + 0.5", qs, lambda t, i: 0.5 * t + 0.5 * i)
    run("テキスト + 0.5·画像 (加算)", qs, lambda t, i: t + 0.5 * i)
