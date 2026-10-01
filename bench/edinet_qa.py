"""有価証券報告書 (EDINET、10 社 26 期) から日本語の QA ベンチを作って評価する。
  gen  : 各有報の本文ページから 27B に「その期の、そのページでしか答えられない質問」を 5 問ずつ作らせる (正解 = 出典ページ) → /data/ragnarok/edinet/qa.jsonl
  eval : emb top-5 / rerank 2 段 (27B・4B) の根拠的中、27B 回答 + 27B 判定の正解率、棄却 (同じ会社の別期) の AUROC
使い方: edinet_qa.py gen | eval"""
import asyncio, json, os, random, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_CACHE", "/data/openvons/choice_spec/hf_cache")
import httpx, numpy as np
from ragnarok.engine import Engine, Index
random.seed(0)
DEMO = "/data/ragnarok/demo"; ED = "/data/ragnarok/edinet"; QA = f"{ED}/qa.jsonl"
DOCS = sorted(d for d in os.listdir(DEMO) if d.startswith("edinet_"))
C = httpx.AsyncClient(timeout=300); SEM = asyncio.Semaphore(16)
async def chat(prompt, max_tokens=300, think=False):
    async with SEM:
        r = await C.post("http://127.0.0.1:8310/v1/chat/completions", json={"model": "qwen27b", "messages": [{"role": "user", "content": prompt}], "max_tokens": 3000 if think else max_tokens, "temperature": 0, "chat_template_kwargs": {"enable_thinking": think}})
    m = r.json()["choices"][0]["message"]; t = (m.get("content") or "").strip(); return t.split("</think>")[-1].strip() if "</think>" in t else t

async def gen():
    rows = []
    for d in DOCS:
        ix = Index.load(f"{DEMO}/{d}"); n = len(ix.pages); name = ix.name
        good = [i for i in range(8, n) if len(ix.pages[i]) > 900 and sum(ch.isdigit() for ch in ix.pages[i]) > 30 and "目次" not in ix.pages[i][:200]]
        picks = random.sample(good, min(6, len(good)))
        async def one(i):
            txt = await chat(f"以下は「{name}」の 1 ページです。このページの内容だけで答えられ、かつ他の期の報告書では答えが変わる (この期に固有の数値・出来事・方針・セグメントの数値など) 具体的な質問を 1 つ作ってください。"
                             f"冒頭の『主要な経営指標等の推移』のような複数期の一覧表に載る数字 (売上高・従業員数の合計など) は避けてください。ページや節への言及はしないこと。短い答えも付けてください。\n\nページ:\n{ix.pages[i][:5000]}\n\nJSON のみで返答: {{\"question\": \"...\", \"answer\": \"...\"}}")
            q = re.search(r'"question"\s*:\s*"((?:[^"\\]|\\.)*)"', txt); a = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', txt)
            return {"doc": d, "name": name, "company": ix.meta["group"], "year": ix.meta["year"], "page": i, "question": q.group(1), "answer": a.group(1) if a else ""} if q else None
        res = [r for r in await asyncio.gather(*[one(i) for i in picks]) if r][:5]; rows += res
        print(f"{name:40s} {len(res)} 問  例: {res[0]['question'][:50] if res else ''}", flush=True)
    with open(QA, "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(rows), "問 ->", QA)

async def eval_():
    rows = [json.loads(l) for l in open(QA)]; eng = Engine(); IX = {d: Index.load(f"{DEMO}/{d}") for d in DOCS}
    sem = asyncio.Semaphore(8)
    async def one(r):
        ix = IX[r["doc"]]
        async with sem:
            qe = eng.emb.encode([r["question"]], prompt_name="query", normalize_embeddings=True, show_progress_bar=False)[0]; order = [int(i) for i in np.argsort(-(ix.emb @ qe))]
            ret = await eng.retrieve(ix, r["question"], k=5)
            ret4 = await eng.retrieve(ix, r["question"], k=5, fast_first=True)
            ans = await eng.answer(ix, r["question"], ret.pages, think=True)
            j = await chat(f"金融の質問について、モデルの回答が正解と同じ事実・数値を述べているか判定してください (数値は丸めや単位の違いを許容、説明が多いのは可)。同じなら CORRECT、違えば INCORRECT と 1 語で。\n\n質問: {r['question']}\n正解: {r['answer']}\nモデルの回答: {ans[:1200]}", 5)
            # 棄却: 同じ会社の別期 (最も離れた期) にぶつける
            others = [d for d in DOCS if IX[d].meta["group"] == r["company"] and d != r["doc"]]
            none_wrong = sim_wrong = None
            if others:
                od = max(others, key=lambda d: abs(int(IX[d].meta["year"]) - int(r["year"]))); ox = IX[od]
                rw = await eng.retrieve(ox, r["question"], k=5); none_wrong = rw.none_final; sim_wrong = float(max(rw.emb_scores.values()))
        return {**r, "emb5": r["page"] in order[:5], "emb1": order[0] == r["page"], "rr5": r["page"] in ret.pages, "rr1": ret.pages[:1] == [r["page"]], "rr5_4b": r["page"] in ret4.pages,
                "none_true": ret.none_final, "sim_true": float(max(ret.emb_scores.values())), "none_wrong": none_wrong, "sim_wrong": sim_wrong, "correct": j.strip().upper().startswith("CORRECT"), "model_answer": ans[:400]}
    t0 = time.time(); res = await asyncio.gather(*[one(r) for r in rows]); n = len(res)
    def auroc(pos, neg): return float(np.mean([[1.0 if a > b else 0.5 if a == b else 0.0 for b in neg] for a in pos]))
    nt = [x["none_true"] for x in res if x["none_wrong"] is not None]; nw = [x["none_wrong"] for x in res if x["none_wrong"] is not None]
    st = [x["sim_true"] for x in res if x["sim_wrong"] is not None]; sw = [x["sim_wrong"] for x in res if x["sim_wrong"] is not None]
    print(f"\n有報 QA {n} 問 ({len(DOCS)} 冊)  wall {time.time()-t0:.0f}s")
    print(f"根拠 hit@1 / hit@5:  emb {np.mean([x['emb1'] for x in res]):.3f} / {np.mean([x['emb5'] for x in res]):.3f}   rerank 2 段 (27B) {np.mean([x['rr1'] for x in res]):.3f} / {np.mean([x['rr5'] for x in res]):.3f}   rerank 2 段 (1 段目 4B) - / {np.mean([x['rr5_4b'] for x in res]):.3f}")
    print(f"回答正解率 (27B 思考あり、27B 判定): {np.mean([x['correct'] for x in res]):.3f}")
    print(f"棄却 (同じ会社の別期、n={len(nt)}): none 確率 AUROC {auroc(nw, nt):.3f} (正しい期の中央値 {np.median(nt):.2f} / 別期 {np.median(nw):.2f})、埋め込み最大類似度 AUROC {auroc(st, sw):.3f}; none>0.5: 誤棄却 {np.mean(np.array(nt)>0.5):.2f} / 棄却 {np.mean(np.array(nw)>0.5):.2f}")
    json.dump(res, open(f"{ED}/qa_eval.json", "w"), ensure_ascii=False, indent=1)

asyncio.run(gen() if sys.argv[1] == "gen" else eval_())
