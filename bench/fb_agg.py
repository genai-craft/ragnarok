"""集約型の質問 (「X に触れている箇所を全部」) の網羅性を測る。GraphRAG のグローバル検索が狙う「取りこぼさない」問題を、
判定の積み重ねで安く解けるかの実験。
  正解 (oracle): 文書の全ページに 27B で yes/no (そのページはこの話題に実質的に触れているか)。
  手法: emb top-k (ベクトルの常套) / キーワード一致 / emb top-50 → 4B or 27B の yes/no 判定で絞る (jev-rag) / 全ページ 4B 判定 (網羅、GraphRAG の抽出の代わり)
  指標: oracle の正ページ集合に対する再現率・適合率・F1、1 問あたりの呼び出し回数。"""
import asyncio, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_CACHE", "/data/openvons/choice_spec/hf_cache")
import numpy as np
from openvons.core.primitives import Option, Question
from openvons.lm.backends.llm_backend import LLMBackend
FB = "/data/jev-rag/financebench"
DOCS = ["3M_2022_10K", "AMCOR_2023_10K", "PAYPAL_2022_10K", "VERIZON_2022_10K", "WALMART_2020_10K", "ADOBE_2022_10K"]
TOPICS = [("share repurchases or buybacks of the company's own stock", "repurchas"), ("goodwill or intangible asset impairment", "impairment"),
          ("climate change, emissions or environmental sustainability", "climate|emission|sustainab"), ("cybersecurity risks or data breaches", "cyber|data breach"),
          ("pension or other post-retirement benefit plans", "pension|postretirement|post-retirement"), ("litigation, lawsuits or legal proceedings", "litigation|lawsuit|legal proceeding"),
          ("stock-based or share-based compensation expense", "stock-based|share-based"), ("foreign currency exchange rate effects", "foreign currency|exchange rate")]
B4 = LLMBackend("http://127.0.0.1:8311/v1", "qwen3-4b", mode="logprob", concurrency=48); B27 = LLMBackend("http://127.0.0.1:8310/v1", "qwen27b", mode="logprob", concurrency=16)
MODE = sys.argv[1] if len(sys.argv) > 1 else "oracle"

def pages(d): return json.load(open(f"{FB}/pages/{d}.json"))
def emb(d): return np.load(f"{FB}/emb/{d}.npy")
_qm = None
def qemb(q):
    global _qm
    if _qm is None:
        from sentence_transformers import SentenceTransformer
        _qm = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda")
    return _qm.encode([q], prompt_name="query", normalize_embeddings=True, show_progress_bar=False)[0]

async def judge_pages(d, topic, idx, backend):
    """ページごとの yes/no (実質的に触れているか)。戻り値 {page: p_yes}"""
    pg = pages(d)
    async def one(p):
        q = Question("choice", f"Does this page substantively discuss {topic} (not just a passing mention in a table of contents or boilerplate)?",
                     [Option("yes", "yes"), Option("no", "no")])
        dec = (await backend.adecide(f"Document: {d} (10-K). Page {p+1}:\n{re.sub(r'[ \\t]+', ' ', pg[p])[:3500]}", [q]))[0]
        return p, float(dec.probs[0])
    return dict(await asyncio.gather(*[one(p) for p in idx]))

async def main():
    out_f = f"{FB}/agg_oracle.json"
    if MODE == "oracle":
        oracle = json.load(open(out_f)) if os.path.exists(out_f) else {}
        for d in DOCS:
            n = len(pages(d))
            for topic, kw in TOPICS:
                key = f"{d}||{topic}"
                if key in oracle: continue
                t = time.time(); sc = await judge_pages(d, topic, list(range(n)), B27)
                oracle[key] = {str(p): v for p, v in sc.items()}
                json.dump(oracle, open(out_f, "w"))
                print(f"{d:18s} {topic[:40]:40s} pages {n} 正 {sum(1 for v in sc.values() if v >= 0.5)} ({time.time()-t:.0f}s)", flush=True)
        return
    oracle = json.load(open(out_f))
    rows = []
    for d in DOCS:
        pg = pages(d); n = len(pg); E = emb(d)
        for topic, kw in TOPICS:
            key = f"{d}||{topic}"
            if key not in oracle: continue
            gold = {int(p) for p, v in oracle[key].items() if v >= 0.5}
            if len(gold) < 2: continue
            s = E @ qemb(f"Which pages discuss {topic}?"); order = [int(i) for i in np.argsort(-s)]
            res = {"emb@10": set(order[:10]), "emb@20": set(order[:20]), "emb@|gold|": set(order[: len(gold)]),
                   "keyword": {p for p in range(n) if re.search(kw, pg[p], re.I)}}
            calls = {"emb@10": 0, "emb@20": 0, "emb@|gold|": 0, "keyword": 0}
            for name, backend in (("emb50→4B", B4), ("emb50→27B", B27)):
                sc = await judge_pages(d, topic, order[:50], backend); res[name] = {p for p, v in sc.items() if v >= 0.5}; calls[name] = 50
            sc = await judge_pages(d, topic, list(range(n)), B4); res["全頁→4B"] = {p for p, v in sc.items() if v >= 0.5}; calls["全頁→4B"] = n
            for name, sel in res.items():
                tp = len(sel & gold); rows.append({"doc": d, "topic": topic, "method": name, "gold": len(gold), "sel": len(sel), "recall": tp / len(gold), "precision": tp / max(len(sel), 1), "calls": calls[name]})
            print(f"{d:18s} {topic[:30]:30s} gold {len(gold):3d} | " + " ".join(f"{m}:R{r['recall']:.2f}/P{r['precision']:.2f}" for m in res for r in rows[-len(res):] if r["method"] == m), flush=True)
    json.dump(rows, open(f"{FB}/agg_results.json", "w"), ensure_ascii=False, indent=1)
    print("\n手法            再現率  適合率  F1    呼び出し/問")
    for m in ["emb@10", "emb@20", "emb@|gold|", "keyword", "emb50→4B", "emb50→27B", "全頁→4B"]:
        rr = [r for r in rows if r["method"] == m]; R = np.mean([r["recall"] for r in rr]); P = np.mean([r["precision"] for r in rr])
        print(f"{m:14s} {R:.3f}  {P:.3f}  {2*R*P/max(R+P,1e-9):.3f}  {np.mean([r['calls'] for r in rr]):.0f}")
asyncio.run(main())
