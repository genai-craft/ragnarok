"""木の探索の比較: 合成 QA (正解 = 葉の節) で、葉の hit@1/3/5 とページ hit@1、呼び出し回数、待ち時間。
方法: gen27b (PageIndex 方式: 木全体を 27B に見せて id 生成) / dec4b (4B 確率判定のビーム) / dec27b / emb (節のベクトル検索) / hybrid (埋め込みで絞って 4B 判定) / hybrid+sim"""
import asyncio, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_CACHE", "/data/openvons/choice_spec/hf_cache")
from openvons.lm.backends.llm_backend import LLMBackend
from ragnarok.tree import Doc
from ragnarok.navigate import DecisionNavigator, GenerativeNavigator, EmbeddingRetriever, HybridNavigator, RerankRetriever

QA = [json.loads(l) for l in open("/data/jev-rag/bench/qa.jsonl")]
DOCS = {n: Doc.load(f"/data/jev-rag/trees/{n}.json") for n in sorted({r["doc"] for r in QA})}
only = sys.argv[1].split(",") if len(sys.argv) > 1 else None

def pages_hit(hit, r):
    return hit is not None and not (hit.node.end < r["pages"][0] or hit.node.start > r["pages"][1])

async def run(name, nav, conc=8):
    sem = asyncio.Semaphore(conc); t0 = time.time()
    async def one(r):
        async with sem:
            try:
                return await nav.search(r["question"], DOCS[r["doc"]])
            except Exception as e:
                print("ERR", name, str(e)[:120]); return None
    res = await asyncio.gather(*[one(r) for r in QA]); wall = time.time() - t0
    m = {"hit@1": 0, "hit@3": 0, "hit@5": 0, "page@1": 0, "calls": 0, "lat": 0.0, "n": 0}; per = {}
    for r, x in zip(QA, res):
        if x is None: continue
        ids = [h.node.id for h in x.hits]; m["n"] += 1
        d = per.setdefault(r["doc"], {"hit@1": 0, "n": 0}); d["n"] += 1
        if ids[:1] == [r["leaf"]]: m["hit@1"] += 1; d["hit@1"] += 1
        if r["leaf"] in ids[:3]: m["hit@3"] += 1
        if r["leaf"] in ids[:5]: m["hit@5"] += 1
        if pages_hit(x.hits[0] if x.hits else None, r): m["page@1"] += 1
        m["calls"] += x.calls; m["lat"] += x.latency_s
    n = max(m["n"], 1)
    print(f"{name:12s} hit@1 {m['hit@1']/n:.3f} hit@3 {m['hit@3']/n:.3f} hit@5 {m['hit@5']/n:.3f} page@1 {m['page@1']/n:.3f} calls/q {m['calls']/n:.1f} lat/q {m['lat']/n:.2f}s wall {wall:.0f}s n={m['n']} | " +
          " ".join(f"{k[:8]}:{v['hit@1']/v['n']:.2f}" for k, v in per.items()), flush=True)
    json.dump({"name": name, "metrics": m, "per_doc": per, "wall": wall, "preds": [[h.node.id for h in x.hits] if x else None for x in res]},
              open(f"/data/jev-rag/bench/nav_{name}.json", "w"))

async def main():
    b4 = LLMBackend("http://127.0.0.1:8311/v1", "qwen3-4b", mode="logprob", concurrency=32)
    b27 = LLMBackend("http://127.0.0.1:8310/v1", "qwen27b", mode="logprob", concurrency=16)
    emb = None
    def E():
        nonlocal emb
        if emb is None:
            emb = EmbeddingRetriever(device="cuda")
            for d in DOCS.values(): emb.index(d)
        return emb
    arms = {
        "gen27b": lambda: (GenerativeNavigator("http://127.0.0.1:8310/v1", "qwen27b"), 8),
        "dec4b": lambda: (DecisionNavigator(b4), 16),
        "dec27b": lambda: (DecisionNavigator(b27), 8),
        "emb": lambda: (E(), 1),
        "hybrid": lambda: (HybridNavigator(b4, E(), prune=8), 16),
        "hybrid+sim": lambda: (HybridNavigator(b4, E(), prune=8, w_sim=0.3), 16),
        "hybrid27b": lambda: (HybridNavigator(b27, E(), prune=8), 8),
        "rerank4b": lambda: (RerankRetriever(b4, E(), topn=10), 16),
        "rerank4b+sim": lambda: (RerankRetriever(b4, E(), topn=10, w_sim=0.3), 16),
        "rerank27b": lambda: (RerankRetriever(b27, E(), topn=10), 8),
    }
    print(f"{len(QA)} questions over {len(DOCS)} docs", flush=True)
    for name, mk in arms.items():
        if only and name not in only: continue
        nav, conc = mk(); await run(name, nav, conc)

asyncio.run(main())
