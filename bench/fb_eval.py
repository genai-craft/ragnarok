"""FinanceBench (open-source 150 問、10-K 84 冊) で検索方式を比べる。
検索 → 上位 k ページ → 27B が回答 → 27B が正解と照合 (LLM 判定)。指標: 根拠ページ hit@k、回答正解率。
方式: emb (ページ埋め込み) / rerank27b, rerank4b (埋め込み top-10 → 確率判定 1 回) / tree_dec27b, tree_gen27b (PageIndex の木) / rerank27b+tree (候補に節の経路を添える)
使い方: fb_eval.py emb,rerank27b [k]"""
import asyncio, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_CACHE", "/data/openvons/choice_spec/hf_cache")
import httpx, numpy as np
from openvons.core.primitives import Option, Question
from openvons.lm.backends.llm_backend import LLMBackend
from jevrag.tree import Doc, Node, tree_from_pageindex, outline_text

FB = "/data/jev-rag/financebench"
ROWS = [json.loads(l) for l in open(f"{FB}/repo/data/financebench_open_source.jsonl")]
METHODS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["emb"]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 5
C27 = httpx.AsyncClient(timeout=600); B4 = LLMBackend("http://127.0.0.1:8311/v1", "qwen3-4b", mode="logprob", concurrency=32); B27 = LLMBackend("http://127.0.0.1:8310/v1", "qwen27b", mode="logprob", concurrency=12)
SEM27 = asyncio.Semaphore(12)
_pages, _emb, _trees = {}, {}, {}

def pages(d):
    if d not in _pages: _pages[d] = json.load(open(f"{FB}/pages/{d}.json"))
    return _pages[d]
def emb(d):
    if d not in _emb: _emb[d] = np.load(f"{FB}/emb/{d}.npy")
    return _emb[d]
def tree(d) -> Doc | None:
    if d not in _trees:
        f = f"/data/jev-rag/exp/tree2_swap_{d}.json"
        if not os.path.exists(f): _trees[d] = None
        else:
            doc = tree_from_pageindex(f, f"{FB}/repo/pdfs/{d}.pdf", d); doc.pages = pages(d); _trees[d] = doc
    return _trees[d]

_qm = None
def qemb(q):
    global _qm
    if _qm is None:
        from sentence_transformers import SentenceTransformer
        _qm = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda")
    return _qm.encode([q], prompt_name="query", normalize_embeddings=True, show_progress_bar=False)[0]

def snippet(d, p, n=300):
    return re.sub(r"\s+", " ", pages(d)[p])[:n]

def sec_path(doc: Doc, p: int) -> str:
    """ページ p (0 始まり) を含む最も深い節の経路。"""
    best = []
    def walk(n, path):
        nonlocal best
        if n.id != "root" and n.start - 1 <= p <= n.end - 1:
            if len(path) + 1 > len(best): best = path + [n.title]
        for c in n.children: walk(c, path + ([n.title] if n.id != "root" else []))
    walk(doc.root, [])
    return " > ".join(best)

async def ret_emb(r):
    d = r["doc_name"]; s = emb(d) @ qemb(r["question"]); return [int(i) for i in np.argsort(-s)[:K]], 0

async def ret_rerank(r, backend, with_tree=False, topn=10):
    d = r["doc_name"]; s = emb(d) @ qemb(r["question"]); cand = [int(i) for i in np.argsort(-s)[:topn]]
    t = tree(d) if with_tree else None
    opts = [Option(f"p{p}", (f"[{sec_path(t, p)}] " if t else "") + f"page {p+1}: {snippet(d, p)}") for p in cand]
    opts.append(Option("none", "None of these pages contains the answer"))
    q = Question("choice", f"Which page contains the information needed to answer this question?\nQuestion: {r['question']}", opts)
    dec = (await backend.adecide(f"Document: {d} (an annual 10-K filing of {r['company']})", [q]))[0]
    ps = list(dec.probs[:-1]); order = sorted(range(len(cand)), key=lambda i: -ps[i])
    return [cand[i] for i in order[:K]], 1

async def ret_tree_dec(r, backend):
    from jevrag.navigate import DecisionNavigator
    t = tree(r["doc_name"])
    if t is None: return await ret_emb(r)
    res = await DecisionNavigator(backend, beam=3, topk=K).search(r["question"], t)
    out = []
    for h in res.hits:
        for p in range(h.node.start - 1, h.node.end):
            if p not in out: out.append(p)
            if len(out) >= K: break
        if len(out) >= K: break
    return out[:K], res.calls

async def ret_tree_gen(r):
    from jevrag.navigate import GenerativeNavigator
    t = tree(r["doc_name"])
    if t is None: return await ret_emb(r)
    res = await GenerativeNavigator("http://127.0.0.1:8310/v1", "qwen27b", topk=K).search(r["question"], t)
    out = []
    for h in res.hits:
        for p in range(h.node.start - 1, h.node.end):
            if p not in out: out.append(p)
            if len(out) >= K: break
        if len(out) >= K: break
    return out[:K] or (await ret_emb(r))[0], 1

async def chat27(prompt, max_tokens=300):
    async with SEM27:
        r = await C27.post("http://127.0.0.1:8310/v1/chat/completions", json={"model": "qwen27b", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
    return r.json()["choices"][0]["message"]["content"].strip()

async def answer(r, pgs):
    d = r["doc_name"]; ctx = "\n\n".join(f"=== page {p+1} ===\n{pages(d)[p][:6000]}" for p in pgs)
    return await chat27(f"You are a financial analyst. Using only the excerpts below from the 10-K filing of {r['company']} ({d}), answer the question. "
                        f"Be concise; give the number with units when the question asks for a number, and show the calculation briefly if needed. If the excerpts do not contain the answer, say \"Not found\".\n\n{ctx}\n\nQuestion: {r['question']}\nAnswer:")

async def judge(r, ans):
    j = await chat27(f"Compare a model's answer with the gold answer to a financial question. Say CORRECT if the model's answer conveys the same fact/number as the gold answer "
                     f"(numbers within 1% or rounding, units/sign consistent; extra explanation is fine). Otherwise say INCORRECT.\n\nQuestion: {r['question']}\nGold answer: {r['answer']}\n"
                     f"Gold justification: {(r.get('justification') or '')[:500]}\nModel answer: {ans[:1200]}\n\nReply with one word: CORRECT or INCORRECT.", 5)
    return j.strip().upper().startswith("CORRECT")

async def run(method):
    t0 = time.time()
    async def one(r):
        try:
            if method == "emb": pgs, calls = await ret_emb(r)
            elif method == "rerank27b": pgs, calls = await ret_rerank(r, B27)
            elif method == "rerank4b": pgs, calls = await ret_rerank(r, B4)
            elif method == "rerank27b+tree": pgs, calls = await ret_rerank(r, B27, with_tree=True)
            elif method == "tree_dec27b": pgs, calls = await ret_tree_dec(r, B27)
            elif method == "tree_dec4b": pgs, calls = await ret_tree_dec(r, B4)
            elif method == "tree_gen27b": pgs, calls = await ret_tree_gen(r)
            else: raise ValueError(method)
            ev = {int(e["evidence_page_num"]) for e in r["evidence"]}
            hit = bool(ev & set(pgs)); hit1 = bool(ev & set(pgs[:1]))
            ans = await answer(r, pgs); ok = await judge(r, ans)
            return {"id": r["financebench_id"], "pages": pgs, "evidence": sorted(ev), "hit": hit, "hit1": hit1, "answer": ans, "correct": ok, "calls": calls, "type": r["question_type"]}
        except Exception as e:
            print("ERR", method, r["financebench_id"], str(e)[:150], flush=True); return None
    res = await asyncio.gather(*[one(r) for r in ROWS]); res = [x for x in res if x]
    n = len(res); by = {}
    for x in res: by.setdefault(x["type"], []).append(x["correct"])
    print(f"{method:16s} k={K} 根拠 hit@1 {sum(x['hit1'] for x in res)/n:.3f} hit@{K} {sum(x['hit'] for x in res)/n:.3f} 正解率 {sum(x['correct'] for x in res)/n:.3f} n={n} wall {time.time()-t0:.0f}s | "
          + " ".join(f"{k[:8]}:{sum(v)/len(v):.2f}" for k, v in by.items()), flush=True)
    json.dump(res, open(f"{FB}/eval_{method}_k{K}.json", "w"), ensure_ascii=False, indent=1)

async def main():
    for m in METHODS: await run(m)
asyncio.run(main())
