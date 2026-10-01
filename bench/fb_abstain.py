"""棄却の評価: 各質問を (a) 正しい 10-K、(b) 別会社の 10-K にぶつけ、rerank 最終段の「どれでもない」確率で見分けられるか (AUROC)。
比較: 埋め込みの最大類似度で閾値を引く。"""
import asyncio, json, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_CACHE", "/data/openvons/choice_spec/hf_cache")
import numpy as np
sys.argv = ["x", "emb"]
import importlib.util
spec = importlib.util.spec_from_file_location("fb", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fb_eval.py")); fb = importlib.util.module_from_spec(spec); spec.loader.exec_module(fb)
random.seed(0)
ROWS = fb.ROWS; docs = sorted({r["doc_name"] for r in ROWS})

async def none_prob(r, d, backend):
    s = fb.emb(d) @ fb.qemb(r["question"]); cand = [int(i) for i in np.argsort(-s)[:50]]
    pool = []; nones = []
    for i in range(0, len(cand), 25):
        grp = cand[i:i+25]
        opts = [fb.Option(f"p{p}", f"page {p+1}: {fb.snippet(d, p)}") for p in grp] + [fb.Option("none", "None of these pages contains the answer")]
        q = fb.Question("choice", f"Which page contains the information needed to answer this question?\nQuestion: {r['question']}", opts)
        dec = (await backend.adecide(f"Document: {d} (an annual 10-K filing)", [q]))[0]
        ps = [float(x) for x in dec.probs[:-1]]; nones.append(float(dec.probs[-1]))
        pool += [grp[k] for k in sorted(range(len(grp)), key=lambda k: -ps[k])[:6]]
    opts = [fb.Option(f"p{p}", f"page {p+1}: {fb.snippet(d, p)}") for p in pool] + [fb.Option("none", "None of these pages contains the answer")]
    q = fb.Question("choice", f"Which page contains the information needed to answer this question?\nQuestion: {r['question']}", opts)
    dec = (await backend.adecide(f"Document: {d} (an annual 10-K filing)", [q]))[0]
    return float(dec.probs[-1]), min(nones), float(np.max(s))

def auroc(pos, neg):
    return float(np.mean([[1.0 if a > b else 0.5 if a == b else 0.0 for b in neg] for a in pos]))

async def main():
    for name, backend in (("27B", fb.B27), ("4B", fb.B4)):
        sem = asyncio.Semaphore(8)
        MODE = os.environ.get("ABSTAIN_MODE", "other_company")
        import re
        def year(d):
            m = re.search(r"(?:19|20)\d\d", d); return int(m.group(0)) if m else 0
        def pick_wrong(r):
            co = r["doc_name"].rsplit("_", 2)[0]
            same = [d for d in docs if d.rsplit("_", 2)[0] == co and d != r["doc_name"] and abs(year(d) - year(r["doc_name"])) >= 3] if MODE == "same_company" else []
            return random.choice(same) if same else (random.choice([d for d in docs if d.rsplit("_", 2)[0] != co]) if MODE != "same_company" else None)
        async def one(r):
            wrong = pick_wrong(r)
            if wrong is None: return None
            async with sem:
                a = await none_prob(r, r["doc_name"], backend); b = await none_prob(r, wrong, backend)
            return a, b
        res = [x for x in await asyncio.gather(*[one(r) for r in ROWS]) if x]
        print(f"[{name}] mode={MODE} n={len(res)}")
        n_true = [a[0] for a, b in res]; n_wrong = [b[0] for a, b in res]
        s_true = [a[2] for a, b in res]; s_wrong = [b[2] for a, b in res]
        print(f"[{name}] 最終段の none 確率: 正しい文書 中央値 {np.median(n_true):.2f} / 別文書 {np.median(n_wrong):.2f} | 別文書を見分ける AUROC {auroc(n_wrong, n_true):.3f} | "
              f"埋め込み最大類似度での AUROC {auroc(s_true, s_wrong):.3f} | none>0.5 で棄却: 正しい文書の誤棄却 {np.mean(np.array(n_true)>0.5):.2f}, 別文書の棄却 {np.mean(np.array(n_wrong)>0.5):.2f}", flush=True)
        json.dump({"true": [a for a, b in res], "wrong": [b for a, b in res]}, open(f"{fb.FB}/abstain_{name}_{MODE}.json", "w"))
asyncio.run(main())
