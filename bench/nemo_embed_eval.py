"""埋め込みの比較: embeddinggemma-2 (今の既定) vs Nemotron-3.5-Embed 1B / 8B。nemo_embed.py の結果を読む。
  FinanceBench 150 問 / 有報 QA 130 問: 根拠ページ recall@1/5/10/50 (文書を教えた場合) と 横断 (文書 hit@1/5、根拠 hit@5)
  スキャン 48 問 (英 24 / 日 24): recall@1/5/10
  条件: 本文 / 画像のみ / 本文 + 0.3·画像 (今の ragnarok の融合) / 画像 + 本文を 1 文書 (Nemotron のみ)
    CUDA_VISIBLE_DEVICES=0 /data/jev-rag/venv/bin/python bench/nemo_embed_eval.py [gemma,nemo1b,nemo8b]
"""
import json, os, sys
os.environ.setdefault("HF_HUB_CACHE", "/data/openvons/choice_spec/hf_cache")
import numpy as np, torch
from sentence_transformers import SentenceTransformer

FB = "/data/jev-rag/financebench"; DEMO = "/data/ragnarok/demo"; SC = "/data/ragnarok/scans"; NM = "/data/ragnarok/nemo"
MODELS = (sys.argv[1] if len(sys.argv) > 1 else "gemma,nemo1b,nemo8b").split(",")
fb = [json.loads(l) for l in open(f"{FB}/repo/data/financebench_open_source.jsonl")]
ya = [json.loads(l) for l in open("/data/ragnarok/edinet/qa.jsonl")]
sc = [json.loads(l) for l in open(f"{SC}/qa.jsonl")]
Q = {"fb": [(r["question"], r["doc_name"], {int(e["evidence_page_num"]) for e in r["evidence"]}) for r in fb],
     "ya": [(r["question"], r["doc"], {r["page"]}) for r in ya],
     "scan": [(r["question"], r["doc"], {r["page"]}) for r in sc]}
DOCS = {k: sorted({d for _, d, _ in v}) for k, v in Q.items()}


def norm(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def load(model, s, kind):
    """{doc: (pages, dim)} か None。"""
    out = {}
    for d in DOCS[s]:
        if model == "gemma":
            if s == "fb": f = {"text": f"{FB}/emb_gemma/{d}.npy", "img": f"/data/ragnarok/lake/emb_img/{d}.npy"}.get(kind)
            elif s == "ya": f = {"text": f"{DEMO}/{d}/emb.npy", "img": f"{DEMO}/{d}/emb_img.npy"}.get(kind)
            else: f = {"img": f"{SC}/img/{d}.npy"}.get(kind)
        else:
            f = f"{NM}/{model}/{s}_{kind}/{d}.npy"
        if not f or not os.path.exists(f): return None
        out[d] = norm(np.load(f).astype(np.float32))
    return out


def scan_text_gemma(m, src):
    """スキャンの OCR 文字を gemma で埋め込む (保存していないのでその場で)。"""
    out = {}
    for d in DOCS["scan"]:
        pages = json.load(open(f"{SC}/{src}/{d}.json"))["pages"]
        out[d] = m.encode([(p or " ")[:6000] for p in pages], prompt_name="document", batch_size=8, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    return out


def evaluate(qs, qv, E, docs, cross=True):
    rec = {k: 0 for k in (1, 5, 10, 50)}; d1 = d5 = p5 = 0
    if cross:
        mat = np.concatenate([E[d] for d in docs]); rows = [(d, p) for d in docs for p in range(len(E[d]))]
    for (q, d, ev), v in zip(qs, qv):
        order = np.argsort(-(E[d] @ v))
        for k in rec: rec[k] += bool(ev & set(int(i) for i in order[:k]))
        if cross:
            top = np.argsort(-(mat @ v))[:400]; out, cnt = [], {}
            for i in top:
                dd, pp = rows[i]
                if cnt.get(dd, 0) >= 10: continue
                cnt[dd] = cnt.get(dd, 0) + 1; out.append((dd, pp))
                if len(out) >= 5: break
            d1 += out[0][0] == d; d5 += any(dd == d for dd, _ in out); p5 += any(dd == d and pp in ev for dd, pp in out)
    n = len(qs); r = {f"r@{k}": round(v / n, 3) for k, v in rec.items()}
    if cross: r.update({"doc@1": round(d1 / n, 3), "doc@5": round(d5 / n, 3), "ev@5": round(p5 / n, 3)})
    return r


def fuse(A, B, w=0.3):
    return {d: A[d] + w * B[d][:len(A[d])] if len(B[d]) >= len(A[d]) else A[d] for d in A}


results = []
for model in MODELS:
    name = {"gemma": "google/embeddinggemma-2", "nemo1b": "nvidia/Nemotron-3.5-Embed-1B-BF16", "nemo8b": "nvidia/Nemotron-3.5-Embed-8B-BF16"}[model]
    m = SentenceTransformer(name, device="cuda", trust_remote_code=True, model_kwargs={"dtype": torch.bfloat16} if model == "gemma" else {"dtype": torch.bfloat16, "attn_implementation": "sdpa"})
    if model == "gemma": m.max_seq_length = 2048
    qenc = (lambda qs: m.encode(qs, prompt_name="query", normalize_embeddings=True, show_progress_bar=False)) if model == "gemma" else (lambda qs: m.encode_query(qs, normalize_embeddings=True, show_progress_bar=False))
    for s in ("fb", "ya", "scan"):
        qs = Q[s]
        if s == "scan":
            groups = [("英語", [i for i, x in enumerate(qs) if not x[1].startswith("hiroshima")]), ("日本語", [i for i, x in enumerate(qs) if x[1].startswith("hiroshima")])]
        else:
            groups = [("", list(range(len(qs))))]
        qv = norm(np.asarray(qenc([q for q, _, _ in qs]), dtype=np.float32))
        conds = {}
        T = load(model, s, "text"); I = load(model, s, "img"); IT = load(model, s, "it")
        if s == "scan":
            T = scan_text_gemma(m, "ocr") if model == "gemma" else T                     # 本文 = VL-8B の OCR
            TP = scan_text_gemma(m, "ocr_nparse_vllm") if model == "gemma" else load(model, s, "textparse")
            if TP: conds["OCR (Nemotron Parse) の文字"] = TP
            if TP and I: conds["Parse OCR + 0.3·画像"] = fuse(TP, I)
        if T: conds["本文" if s != "scan" else "OCR (VL-8B) の文字"] = T
        if I: conds["画像のみ"] = I
        if T and I: conds["本文 + 0.3·画像" if s != "scan" else "VL-8B OCR + 0.3·画像"] = fuse(T, I)
        if IT: conds["画像 + 本文を 1 文書" if s != "scan" else "画像 + VL-8B OCR を 1 文書"] = IT
        for cname, E in conds.items():
            for g, idx in groups:
                sub = [qs[i] for i in idx]; docs = sorted({d for _, d, _ in sub})
                r = evaluate(sub, qv[idx], E, docs, cross=(s != "scan"))
                row = {"model": model, "set": s + (f"・{g}" if g else ""), "cond": cname, **r}; results.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    del m; torch.cuda.empty_cache()
os.makedirs(NM, exist_ok=True)
old = json.load(open(f"{NM}/eval.json")) if os.path.exists(f"{NM}/eval.json") else []
keep = [r for r in old if r["model"] not in MODELS]
json.dump(keep + results, open(f"{NM}/eval.json", "w"), ensure_ascii=False, indent=1)
