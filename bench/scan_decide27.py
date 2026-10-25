"""スキャン 48 問: scan_rerank.py と同じ埋め込み上位 20 頁を、今の ragnarok の確率判定 (27B、選択肢 + どれでもない) で並べた場合。
  候補の説明は ragnarok と同じ「質問に合う窓」(Engine.snip_for) で、OCR 文字は VL-8B / Parse の 2 通り。
  1 段 (20 択を 1 回) と ragnarok 既定の 2 段 (10 + 10 → 上位 5 + 5 → 最終) を比べる。
    CUDA_VISIBLE_DEVICES=0 /data/jev-rag/venv/bin/python bench/scan_decide27.py
"""
import asyncio, json, os, sys
os.environ.setdefault("HF_HUB_CACHE", "/data/openvons/choice_spec/hf_cache")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch
from openvons.core.primitives import Option, Question
from openvons.lm.backends.llm_backend import LLMBackend
from ragnarok.engine import Engine as _E
from sentence_transformers import SentenceTransformer

S = "/data/ragnarok/scans"; TOPN = 20
rows = [json.loads(l) for l in open(f"{S}/qa.jsonl")]
DOCS = sorted({r["doc"] for r in rows})
TXT = {"VL-8B": {d: json.load(open(f"{S}/ocr/{d}.json"))["pages"] for d in DOCS},
       "Parse": {d: json.load(open(f"{S}/ocr_nparse_vllm/{d}.json"))["pages"] for d in DOCS}}
IMG = {d: np.load(f"{S}/img/{d}.npy") for d in DOCS}
gm = SentenceTransformer("google/embeddinggemma-2", device="cuda", model_kwargs={"torch_dtype": torch.bfloat16}); gm.max_seq_length = 2048
E = {d: gm.encode([(t or " ")[:6000] for t in TXT["VL-8B"][d]], prompt_name="document", batch_size=8, normalize_embeddings=True, show_progress_bar=False) for d in DOCS}
qv = gm.encode([r["question"] for r in rows], prompt_name="query", normalize_embeddings=True, show_progress_bar=False)
cands = [[int(i) for i in np.argsort(-(E[r["doc"]] @ v + 0.3 * (IMG[r["doc"]] @ v)))[:TOPN]] for r, v in zip(rows, qv)]
del gm; torch.cuda.empty_cache()
B27 = LLMBackend("http://127.0.0.1:8310/v1", "qwen27b", mode="logprob", concurrency=12)


async def decide(r, cand, txt):
    ja = r["doc"].startswith("hiroshima")
    opts = [Option(f"p{p}", f"page {p+1}: {_E.snip_for(r['question'], txt[r['doc']][p] or '')}") for p in cand]
    opts.append(Option("none", "None of these pages contains the answer"))
    q = Question("choice", f"Which page contains the information needed to answer this question?\nQuestion: {r['question']}", opts)
    d = (await B27.adecide(f"Document: a scanned {'Japanese fisheries research bulletin' if ja else 'town annual report'} (OCR text)", [q]))[0]
    return dict(zip(cand, [float(x) for x in d.probs[:-1]]))


async def rank(r, cand, txt, two_stage):
    if not two_stage:
        sc = await decide(r, cand, txt)
    else:
        pool = []
        for i in range(0, len(cand), 10):
            s1 = await decide(r, cand[i:i + 10], txt); pool += [p for p, _ in sorted(s1.items(), key=lambda x: -x[1])[:5]]
        sc = await decide(r, pool, txt)
    return [p for p, _ in sorted(sc.items(), key=lambda x: -x[1])]


async def main():
    out = {}
    for src in ("VL-8B", "Parse"):
        for two in (False, True):
            name = f"27B 確率判定 ({'2 段' if two else '1 段 20 択'}): {src} OCR 窓"
            ranks = await asyncio.gather(*[rank(r, c, TXT[src], two) for r, c in zip(rows, cands)])
            for grp, sel in (("英語", lambda d: not d.startswith("hiroshima")), ("日本語", lambda d: d.startswith("hiroshima"))):
                idx = [i for i, r in enumerate(rows) if sel(r["doc"])]
                rec = {k: float(np.mean([rows[i]["page"] in ranks[i][:k] for i in idx])) for k in (1, 3, 5)}
                print(f"{grp} {name:36s} recall@1 {rec[1]:.3f} @3 {rec[3]:.3f} @5 {rec[5]:.3f}", flush=True)
                out.setdefault(name, {})[grp] = {f"r@{k}": round(v, 3) for k, v in rec.items()}
    json.dump(out, open(f"{S}/decide27_eval.json", "w"), ensure_ascii=False, indent=1)


asyncio.run(main())
