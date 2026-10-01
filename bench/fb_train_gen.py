"""4B rerank の学習データ: FinanceBench の評価に使わない 10-K (284 冊のうち 150 冊) のページから 27B に質問を作らせ、
埋め込み top-25 を候補にした「どのページか」の問題 (正解 = 出典ページ) を openvons の Sample 形式で書く。27B の確率 (教師) も付ける。"""
import asyncio, json, os, random, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_CACHE", "/data/openvons/choice_spec/hf_cache")
import httpx, numpy as np, pymupdf
from openvons.core.primitives import Option, Question
from openvons.core.formats import Sample
from openvons.lm.backends.llm_backend import LLMBackend
random.seed(0)
FB = "/data/jev-rag/financebench"; OUT = os.path.expanduser("~/dev/openvons/data/store")
N_DOCS, PER_DOC, TOPN = int(os.environ.get("N_DOCS", 150)), int(os.environ.get("PER_DOC", 10)), 25
rows = [json.loads(l) for l in open(f"{FB}/repo/data/financebench_open_source.jsonl")]
used = {r["doc_name"] for r in rows}
alld = sorted(f[:-4] for f in os.listdir(f"{FB}/repo/pdfs") if f.endswith(".pdf") and f[:-4] not in used)
random.shuffle(alld); docs = alld[:N_DOCS]
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda"); m.max_seq_length = 2048
C = httpx.AsyncClient(timeout=300); SEM = asyncio.Semaphore(16); B27 = LLMBackend("http://127.0.0.1:8310/v1", "qwen27b", mode="logprob", concurrency=12)

def snippet(t, n=300): return re.sub(r"\s+", " ", t)[:n]

async def make_q(doc, page, text):
    prompt = (f"Below is one page of the 10-K filing \"{doc}\". Write ONE specific question that a financial analyst might ask and that can be answered from this page "
              f"(a number, a date, a policy, a reason). Do not mention the page or section. Reply with JSON only: {{\"question\": \"...\"}}\n\nPage:\n{text[:5000]}")
    async with SEM:
        r = await C.post("http://127.0.0.1:8310/v1/chat/completions", json={"model": "qwen27b", "messages": [{"role": "user", "content": prompt}], "max_tokens": 120, "temperature": 0.5, "chat_template_kwargs": {"enable_thinking": False}})
    mm = re.search(r'"question"\s*:\s*"((?:[^"\\]|\\.)*)"', r.json()["choices"][0]["message"]["content"])
    return mm.group(1) if mm else None

async def proc_doc(d):
    with pymupdf.open(f"{FB}/repo/pdfs/{d}.pdf") as f:
        pages = [p.get_text() for p in f]
    good = [i for i, t in enumerate(pages) if len(t) > 1200 and sum(ch.isdigit() for ch in t) > 40]
    if len(good) < PER_DOC: return []
    emb = m.encode([p[:6000] or " " for p in pages], batch_size=8, normalize_embeddings=True, show_progress_bar=False)
    picks = random.sample(good, PER_DOC)
    qs = await asyncio.gather(*[make_q(d, i, pages[i]) for i in picks])
    out = []
    for i, q in zip(picks, qs):
        if not q: continue
        qe = m.encode([q], prompt_name="query", normalize_embeddings=True, show_progress_bar=False)[0]
        cand = [int(x) for x in np.argsort(-(emb @ qe))[:TOPN]]
        if i not in cand: continue
        random.shuffle(cand)
        opts = [Option(f"p{p}", f"page {p+1}: {snippet(pages[p])}") for p in cand] + [Option("none", "None of these pages contains the answer")]
        question = Question("choice", f"Which page contains the information needed to answer this question?\nQuestion: {q}", opts)
        state = f"Document: {d} (an annual 10-K filing)"
        dec = (await B27.adecide(state, [question]))[0]
        out.append(Sample(id=f"{d}_p{i}", state=state, question=question, label=cand.index(i), target_probs=[float(x) for x in dec.probs], meta={"doc": d, "page": i, "q": q}))
    return out

async def main():
    t0 = time.time(); samples = []
    for k in range(0, len(docs), 4):
        res = await asyncio.gather(*[proc_doc(d) for d in docs[k:k+4]])
        for r in res: samples += r
        print(f"{k+4}/{len(docs)} docs, {len(samples)} samples, {time.time()-t0:.0f}s", flush=True)
    random.shuffle(samples)
    n = len(samples); split = {"test": samples[: n // 10], "valid": samples[n // 10 : n // 5], "train": samples[n // 5 :]}
    for name, ss in split.items():
        os.makedirs(f"{OUT}/processed/fb_rerank", exist_ok=True); os.makedirs(f"{OUT}/generated/fb_rerank/qwen27b_logprob", exist_ok=True)
        with open(f"{OUT}/processed/fb_rerank/{name}.jsonl", "w") as f:
            for s in ss: f.write(s.to_json() + "\n")
        with open(f"{OUT}/generated/fb_rerank/qwen27b_logprob/{name}.jsonl", "w") as f:
            for s in ss: f.write(s.to_json() + "\n")
    agree = sum(1 for s in samples if int(np.argmax(s.target_probs[:-1])) == s.label) / n
    print(f"{n} samples; 27B 教師の正答率 {agree:.3f}; -> {OUT}/processed/fb_rerank/")
asyncio.run(main())
