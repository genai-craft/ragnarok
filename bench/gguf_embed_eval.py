"""言箱 (kotohako) の CPU 埋め込みと同じ経路 (GGUF + llama.cpp) で、Qwen3-Embedding-0.6B (言箱の今) と Nemotron-3.5-Embed-1B を比べる。
  言箱の LlamaEmbedder と同じ前置き・集約: Qwen3 = 質問だけ指示文を前置き・最後のトークン、Nemotron = 質問 "query: " / 文書 "passage: "・平均。
  計算は GPU の llama-server (-ngl 99) で速く回す (CPU と数値は同じ。CPU との一致は別途確認済み: 余弦 0.9998)。
  注意: 双方向の埋め込みモデルを llama-server で同時に流す (--parallel > 1) と別の文書が混ざって壊れる。必ず 1 件ずつ。
  FinanceBench 150 問 (84 冊) / 有報 QA 130 問 (26 冊) の recall@1/5/10/50 と横断の根拠@5。
    CUDA_VISIBLE_DEVICES=2 /data/jev-rag/venv/bin/python bench/gguf_embed_eval.py
"""
import json, os, signal, subprocess, sys, time
import httpx, numpy as np

LS = os.environ.get("LLAMA_SERVER", "/data/openvons/choice_spec/tools/llama.cpp/build/bin/llama-server")
G = "/data/ragnarok/gguf_embed"; FB = "/data/jev-rag/financebench"; DEMO = "/data/ragnarok/demo"; OUT = "/data/ragnarok/gguf_embed/eval"
QI = "Instruct: Given a search query, retrieve relevant passages that answer the query\nQuery: "      # 言箱 store/ragnarok/server.py と同じ
MODELS = {
    "Qwen3-Embedding-0.6B Q8 (言箱の今)": ("Qwen3-Embedding-0.6B-Q8_0.gguf", "last", QI, ""),
    "Nemotron-3.5-Embed-1B Q8": ("Nemotron-3.5-Embed-1B-Q8_0.gguf", "mean", "query: ", "passage: "),
}
fb = [json.loads(l) for l in open(f"{FB}/repo/data/financebench_open_source.jsonl")]
ya = [json.loads(l) for l in open("/data/ragnarok/edinet/qa.jsonl")]
SETS = {"FinanceBench": [(r["question"], r["doc_name"], {int(e["evidence_page_num"]) for e in r["evidence"]}) for r in fb],
        "有報": [(r["question"], r["doc"], {r["page"]}) for r in ya]}
PAGES = {}
for d in sorted({r["doc_name"] for r in fb}): PAGES[d] = json.load(open(f"{FB}/pages/{d}.json"))
for d in sorted({r["doc"] for r in ya}): PAGES[d] = json.load(open(f"{DEMO}/{d}/pages.json"))["pages"]


def zwnj(t):
    return t.replace("[", "[‌").replace("<", "<‌")


def serve(gguf, pooling, port=8653):
    p = subprocess.Popen([LS, "-m", f"{G}/{gguf}", "--embeddings", "--pooling", pooling, "-ngl", "99", "-c", "8192", "-b", "8192", "-ub", "8192", "--parallel", "1", "--port", str(port)],
                         stdout=subprocess.DEVNULL, stderr=open(f"{OUT}/server_{gguf}.log", "w"))
    for _ in range(180):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200: return p, port
        except Exception: pass
        time.sleep(1)
    raise RuntimeError("server did not start")


def embed(port, texts, conc=1):
    """1 件ずつ。双方向 (Nemotron) を llama.cpp で同時に流すと別の文書が混ざる (余弦 0.67 まで落ちる、2026-10-27 確認)。言箱も 1 件ずつ。"""
    import concurrent.futures as cf
    c = httpx.Client(timeout=600)
    def one(t):
        v = np.asarray(c.post(f"http://127.0.0.1:{port}/v1/embeddings", json={"input": t}).json()["data"][0]["embedding"], dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-9)
    with cf.ThreadPoolExecutor(conc) as ex:
        return np.stack(list(ex.map(one, texts)))


def evaluate(qs, qv, E, docs):
    rec = {k: 0 for k in (1, 5, 10, 50)}; p5 = 0
    mat = np.concatenate([E[d] for d in docs]); rows = [(d, p) for d in docs for p in range(len(E[d]))]
    for (q, d, ev), v in zip(qs, qv):
        order = np.argsort(-(E[d] @ v))
        for k in rec: rec[k] += bool(ev & set(int(i) for i in order[:k]))
        top = np.argsort(-(mat @ v))[:400]; out, cnt = [], {}
        for i in top:
            dd, pp = rows[i]
            if cnt.get(dd, 0) >= 10: continue
            cnt[dd] = cnt.get(dd, 0) + 1; out.append((dd, pp))
            if len(out) >= 5: break
        p5 += any(dd == d and pp in ev for dd, pp in out)
    n = len(qs); return {**{f"r@{k}": round(v / n, 3) for k, v in rec.items()}, "横断根拠@5": round(p5 / n, 3)}


os.makedirs(OUT, exist_ok=True); results = []
for name, (gguf, pooling, qpre, dpre) in MODELS.items():
    p, port = serve(gguf, pooling)
    try:
        E = {}
        for s, qs in SETS.items():
            docs = sorted({d for _, d, _ in qs})
            for d in docs:
                if d not in E:
                    E[d] = embed(port, [dpre + zwnj((t or " ")[:6000]) if dpre else (t or " ")[:6000] for t in PAGES[d]])
            qv = embed(port, [qpre + (zwnj(q) if dpre else q) for q, _, _ in qs])
            r = {"model": name, "set": s, **evaluate(qs, qv, E, docs)}; results.append(r); print(json.dumps(r, ensure_ascii=False), flush=True)
    finally:
        p.send_signal(signal.SIGTERM); p.wait()
json.dump(results, open(f"{OUT}/results.json", "w"), ensure_ascii=False, indent=1)
