"""スキャン 48 問: 埋め込み (gemma、VL-8B OCR 文字 + 0.3·画像 = 今の最良) の上位 20 頁を Nemotron-3.5-Rerank-8B で並べ替える。
  入力の種類: OCR 文字 (VL-8B / Parse) / ページ画像のみ / 画像 + OCR 文字。recall@1/3/5 を英日別に。
    CUDA_VISIBLE_DEVICES=0 /data/jev-rag/venv/bin/python bench/scan_rerank.py
"""
import glob, json, os, time
os.environ.setdefault("HF_HUB_CACHE", "/data/openvons/choice_spec/hf_cache")
import numpy as np, pymupdf, torch
from PIL import Image
from sentence_transformers import CrossEncoder, SentenceTransformer

S = "/data/ragnarok/scans"; TOPN = 20
rows = [json.loads(l) for l in open(f"{S}/qa.jsonl")]
DOCS = sorted({r["doc"] for r in rows})
OCR = {d: json.load(open(f"{S}/ocr/{d}.json"))["pages"] for d in DOCS}
NP = {d: json.load(open(f"{S}/ocr_nparse_vllm/{d}.json"))["pages"] for d in DOCS}
IMG = {d: np.load(f"{S}/img/{d}.npy") for d in DOCS}
PDF = {}
for f in glob.glob(f"{S}/*.pdf"):
    name = os.path.basename(f)[:-4].split("-")[0] if "town" in f else "hiroshima_" + os.path.basename(f).split("_")[1]
    PDF[name] = f

gm = SentenceTransformer("google/embeddinggemma-2", device="cuda", model_kwargs={"torch_dtype": torch.bfloat16}); gm.max_seq_length = 2048
E = {d: gm.encode([(t or " ")[:6000] for t in OCR[d]], prompt_name="document", batch_size=8, normalize_embeddings=True, show_progress_bar=False) for d in DOCS}
qv = gm.encode([r["question"] for r in rows], prompt_name="query", normalize_embeddings=True, show_progress_bar=False)
cands = []
for r, v in zip(rows, qv):
    s = E[r["doc"]] @ v + 0.3 * (IMG[r["doc"]] @ v); cands.append([int(i) for i in np.argsort(-s)[:TOPN]])
del gm; torch.cuda.empty_cache()


def img(d, p, long=1120):
    with pymupdf.open(PDF[d]) as doc:
        pg = doc[p]; z = long / max(pg.rect.width, pg.rect.height); px = pg.get_pixmap(matrix=pymupdf.Matrix(z, z))
        return Image.frombytes("RGB", (px.width, px.height), px.samples)


rr = CrossEncoder("nvidia/Nemotron-3.5-Rerank-8B-BF16", device="cuda", trust_remote_code=True, model_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa"})
conds = {"埋め込みのみ (gemma VL-8B OCR + 0.3·画像)": None, "Rerank: VL-8B OCR 文字": "ocr", "Rerank: Parse OCR 文字": "np", "Rerank: 画像のみ": "img", "Rerank: 画像 + VL-8B OCR 文字": "it"}
out = {}
for name, kind in conds.items():
    t0 = time.time(); ranks = []
    for r, c in zip(rows, cands):
        d = r["doc"]
        if kind is None:
            ranks.append(c); continue
        docs = {"ocr": lambda: [(OCR[d][p] or " ")[:6000] for p in c], "np": lambda: [(NP[d][p] or " ")[:6000] for p in c],
                "img": lambda: [{"image": img(d, p)} for p in c], "it": lambda: [{"image": img(d, p), "text": (OCR[d][p] or " ")[:6000]} for p in c]}[kind]()
        res = rr.rank(r["question"], docs, batch_size=8 if kind in ("ocr", "np") else 2, show_progress_bar=False)
        ranks.append([c[x["corpus_id"]] for x in res])
    dt = time.time() - t0
    for grp, sel in (("英語", lambda d: not d.startswith("hiroshima")), ("日本語", lambda d: d.startswith("hiroshima"))):
        idx = [i for i, r in enumerate(rows) if sel(r["doc"])]
        rec = {k: np.mean([rows[i]["page"] in ranks[i][:k] for i in idx]) for k in (1, 3, 5)}
        print(f"{grp} {name:36s} recall@1 {rec[1]:.3f} @3 {rec[3]:.3f} @5 {rec[5]:.3f}" + (f"  ({dt/len(rows):.1f}s/問)" if kind else ""), flush=True)
        out.setdefault(name, {})[grp] = {f"r@{k}": round(float(v), 3) for k, v in rec.items()}
    if kind: out[name]["s_per_q"] = round(dt / len(rows), 2)
json.dump(out, open(f"{S}/rerank_eval.json", "w"), ensure_ascii=False, indent=1)
