"""埋め込みモデルの比較: Qwen3-Embedding-0.6B (今の既定) vs google/embeddinggemma-2 (768d / 256d MRL / ページ画像)。
  単一文書: 根拠ページの recall@1/5/10/50 (FinanceBench 150 問、有報 QA 130 問)
  横断 (データレイク): 文書 hit@1/5、根拠ページ hit@5 (横断 top-5、1 文書最大 10)
  画像: 一部の文書でページ画像だけを埋め込み、テキスト質問で引けるか (スキャン PDF 向け)"""
import json, os, sys, time, io
os.environ.setdefault("HF_HUB_CACHE", "/data/openvons/choice_spec/hf_cache")
import numpy as np, torch
FB = "/data/jev-rag/financebench"; DEMO = "/data/ragnarok/demo"
fb = [json.loads(l) for l in open(f"{FB}/repo/data/financebench_open_source.jsonl")]
ya = [json.loads(l) for l in open("/data/ragnarok/edinet/qa.jsonl")]
fb_docs = sorted({r["doc_name"] for r in fb}); ya_docs = sorted({r["doc"] for r in ya})
pages = {d: json.load(open(f"{FB}/pages/{d}.json")) for d in fb_docs}
for d in ya_docs: pages[d] = json.load(open(f"{DEMO}/{d}/pages.json"))["pages"]
Q_FB = [(r["question"], r["doc_name"], {int(e["evidence_page_num"]) for e in r["evidence"]}) for r in fb]
Q_YA = [(r["question"], r["doc"], {r["page"]}) for r in ya]

def evaluate(name, E, qe_fn, qs, docs):
    """E: {doc: (n_pages, dim)}; qe_fn: 質問列 → (n, dim)。単一文書 recall と横断の文書/ページ hit。"""
    qv = qe_fn([q for q, _, _ in qs])
    rec = {k: 0 for k in (1, 5, 10, 50)}; d1 = d5 = p5 = 0
    mat = np.concatenate([E[d] for d in docs]); rows = [(d, p) for d in docs for p in range(len(E[d]))]
    for (q, d, ev), v in zip(qs, qv):
        s = E[d] @ v; order = np.argsort(-s)
        for k in rec: rec[k] += bool(ev & set(int(i) for i in order[:k]))
        sc = mat @ v; top = np.argsort(-sc)[:400]; out, cnt = [], {}
        for i in top:
            dd, pp = rows[i]
            if cnt.get(dd, 0) >= 10: continue
            cnt[dd] = cnt.get(dd, 0) + 1; out.append((dd, pp))
            if len(out) >= 5: break
        d1 += out[0][0] == d; d5 += any(dd == d for dd, _ in out); p5 += any(dd == d and pp in ev for dd, pp in out)
    n = len(qs)
    print(f"  {name:34s} recall@1 {rec[1]/n:.3f} @5 {rec[5]/n:.3f} @10 {rec[10]/n:.3f} @50 {rec[50]/n:.3f} | 横断: 文書@1 {d1/n:.3f} 文書@5 {d5/n:.3f} 根拠@5 {p5/n:.3f}", flush=True)

from sentence_transformers import SentenceTransformer
print("== Qwen3-Embedding-0.6B (既定)")
qm = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda")
E_q = {d: np.load(f"{FB}/emb/{d}.npy") for d in fb_docs}; E_q.update({d: np.load(f"{DEMO}/{d}/emb.npy") for d in ya_docs})
qf = lambda qs: qm.encode(qs, prompt_name="query", normalize_embeddings=True, show_progress_bar=False)
print("FinanceBench:"); evaluate("Qwen3-Embedding-0.6B 1024d", E_q, qf, Q_FB, fb_docs)
print("有報 QA:"); evaluate("Qwen3-Embedding-0.6B 1024d", E_q, qf, Q_YA, ya_docs)
del qm; torch.cuda.empty_cache()

print("== google/embeddinggemma-2")
gm = SentenceTransformer("google/embeddinggemma-2", device="cuda", model_kwargs={"torch_dtype": torch.bfloat16}); gm.max_seq_length = 2048
t0 = time.time(); E_g = {}; n_pages = 0
for d in fb_docs + ya_docs:
    E_g[d] = gm.encode([p[:6000] or " " for p in pages[d]], prompt_name="document", batch_size=16, normalize_embeddings=True, show_progress_bar=False).astype(np.float32); n_pages += len(pages[d])
print(f"  {n_pages} ページの埋め込み {time.time()-t0:.0f}s ({1000*(time.time()-t0)/n_pages:.0f} ms/ページ)")
gf = lambda qs: gm.encode(qs, prompt_name="query", normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
print("FinanceBench:"); evaluate("embeddinggemma-2 768d", E_g, gf, Q_FB, fb_docs)
print("有報 QA:"); evaluate("embeddinggemma-2 768d", E_g, gf, Q_YA, ya_docs)
def trunc(E, k):
    out = {}
    for d, m in E.items():
        x = m[:, :k]; out[d] = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    return out
gf256 = lambda qs: (lambda v: v[:, :256] / (np.linalg.norm(v[:, :256], axis=1, keepdims=True) + 1e-9))(gf(qs))
print("FinanceBench:"); evaluate("embeddinggemma-2 256d (MRL)", trunc(E_g, 256), gf256, Q_FB, fb_docs)
print("有報 QA:"); evaluate("embeddinggemma-2 256d (MRL)", trunc(E_g, 256), gf256, Q_YA, ya_docs)
np.save("/data/ragnarok/lake/emb_gemma2_meta.npy", np.array([0]))
# ---- 画像: ページ画像だけで引けるか (スキャン PDF 想定)。FB 5 冊 + 有報 2 冊
import pymupdf
from PIL import Image
sub_fb = [d for d in ["3M_2022_10K", "AMAZON_2019_10K", "NIKE_2023_10K", "VERIZON_2022_10K", "WALMART_2020_10K"] if d in E_g]
sub_ya = [d for d in ["edinet_任天堂_2026-03", "edinet_三菱商事_2024-03"] if d in E_g]
def render(pdf):
    out = []
    with pymupdf.open(pdf) as doc:
        for pg in doc:
            px = pg.get_pixmap(matrix=pymupdf.Matrix(1.0, 1.0)); out.append(Image.frombytes("RGB", (px.width, px.height), px.samples))
    return out
edmeta = {os.path.basename(m["file"])[:-4]: m["file"] for m in json.load(open("/data/ragnarok/edinet/meta.json"))}
def ya_pdf(d):
    co_, ym = d.split("_", 1)[1].rsplit("_", 1); return next(v for k, v in edmeta.items() if k.startswith(co_ + "_" + ym))
E_img = {}; t0 = time.time(); n_img = 0
for d in sub_fb + sub_ya:
    imgs = render(f"{FB}/repo/pdfs/{d}.pdf" if d in sub_fb else ya_pdf(d)); n_img += len(imgs)
    E_img[d] = gm.encode(imgs, prompt_name="document", batch_size=4, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
print(f"  画像 {n_img} ページの埋め込み {time.time()-t0:.0f}s ({1000*(time.time()-t0)/n_img:.0f} ms/ページ)")
qfb = [x for x in Q_FB if x[1] in sub_fb]; qya = [x for x in Q_YA if x[1] in sub_ya]
print(f"画像のみ (FinanceBench {len(qfb)} 問 / 有報 {len(qya)} 問の部分集合):")
evaluate("embeddinggemma-2 画像 768d", E_img, gf, qfb, sub_fb); evaluate("embeddinggemma-2 画像 768d", E_img, gf, qya, sub_ya)
print("同じ部分集合のテキスト:"); evaluate("embeddinggemma-2 テキスト 768d", {d: E_g[d] for d in sub_fb}, gf, qfb, sub_fb); evaluate("embeddinggemma-2 テキスト 768d", {d: E_g[d] for d in sub_ya}, gf, qya, sub_ya)
evaluate("Qwen3-Embedding-0.6B テキスト", {d: E_q[d] for d in sub_fb}, lambda qs: SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda").encode(qs, prompt_name="query", normalize_embeddings=True, show_progress_bar=False), qfb, sub_fb)
