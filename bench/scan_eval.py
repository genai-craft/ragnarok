"""スキャン PDF (文字層なし想定) の検索評価。6 冊 (NH の町の年次報告 3 冊 1964-73、広島県水産試験場研究報告 3 冊 1974-81)。
  正解: Qwen3-VL-8B の OCR テキストから 27B が作った質問 (出典ページ)。
  条件: 画像埋め込みのみ (OCR なし) / IA の OCR 文字層 / VLM の OCR テキスト / 画像 + VLM テキストの融合。
  phase=gen|retrieve|judge。回答は scan_answer_vlm.py (VLM が画像を読む) と 27B (OCR テキスト) を比べる。"""
import asyncio, json, os, re, sys, glob
os.environ.setdefault("HF_HUB_CACHE", "/data/openvons/choice_spec/hf_cache")
import numpy as np, httpx
S = "/data/ragnarok/scans"; PH = sys.argv[1] if len(sys.argv) > 1 else "gen"
DOCS = sorted(os.path.basename(f)[:-5] for f in glob.glob(f"{S}/ocr/*.json"))
OCR = {d: json.load(open(f"{S}/ocr/{d}.json"))["pages"] for d in DOCS}; IA = {d: json.load(open(f"{S}/img/{d}_ia.json"))["pages"] for d in DOCS}; IMG = {d: np.load(f"{S}/img/{d}.npy") for d in DOCS}
C = httpx.AsyncClient(timeout=300); SEM = asyncio.Semaphore(12)
async def chat(p, max_tokens=300, think=False):
    async with SEM:
        r = await C.post("http://127.0.0.1:8310/v1/chat/completions", json={"model": "qwen27b", "messages": [{"role": "user", "content": p}], "max_tokens": 3000 if think else max_tokens, "temperature": 0, "chat_template_kwargs": {"enable_thinking": think}})
    t = (r.json()["choices"][0]["message"].get("content") or "").strip(); return t.split("</think>")[-1].strip() if "</think>" in t else t
def ja(d): return d.startswith("hiroshima")
async def gen():
    import random; random.seed(0); rows = []
    for d in DOCS:
        good = [i for i, t in enumerate(OCR[d]) if len(t) > 500 and sum(ch.isdigit() for ch in t) > 15]
        picks = random.sample(good, min(10, len(good)))
        async def one(i):
            if ja(d):
                p = f"以下は古い報告書 (スキャン) の 1 ページを OCR した文章です。このページの内容だけで答えられる具体的な質問を 1 つ作り、短い答えも付けてください (数値・固有名詞・条件など)。ページへの言及はしないこと。\n\n{OCR[d][i][:4000]}\n\nJSON のみ: {{\"question\": \"...\", \"answer\": \"...\"}}"
            else:
                p = f"Below is the OCR text of one page of an old scanned town annual report. Write ONE specific question answerable only from this page (a number, a name, a date, an amount) and its short answer. Do not mention the page.\n\n{OCR[d][i][:4000]}\n\nJSON only: {{\"question\": \"...\", \"answer\": \"...\"}}"
            t = await chat(p); q = re.search(r'"question"\s*:\s*"((?:[^"\\]|\\.)*)"', t); a = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', t)
            return {"doc": d, "page": i, "question": q.group(1), "answer": a.group(1) if a else ""} if q else None
        res = [r for r in await asyncio.gather(*[one(i) for i in picks]) if r][:8]; rows += res; print(d, len(res), "問", res[0]["question"][:60] if res else "")
    with open(f"{S}/qa.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(rows), "問")
def retrieve():
    import torch
    from sentence_transformers import SentenceTransformer
    gm = SentenceTransformer("google/embeddinggemma-2", device="cuda", model_kwargs={"torch_dtype": torch.bfloat16}); gm.max_seq_length = 2048
    emb = lambda ts: gm.encode([t[:6000] or " " for t in ts], prompt_name="document", batch_size=8, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    E_ia = {d: emb(IA[d]) for d in DOCS}; E_ocr = {d: emb(OCR[d]) for d in DOCS}
    rows = [json.loads(l) for l in open(f"{S}/qa.jsonl")]; qv = gm.encode([r["question"] for r in rows], prompt_name="query", normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    conds = {"画像のみ (OCR なし)": lambda d, v: IMG[d] @ v, "IA の OCR 文字層": lambda d, v: E_ia[d] @ v, "VLM (Qwen3-VL-8B) の OCR": lambda d, v: E_ocr[d] @ v,
             "画像 + 0.3·IA OCR": lambda d, v: IMG[d] @ v + 0.3 * (E_ia[d] @ v), "VLM OCR + 0.3·画像": lambda d, v: E_ocr[d] @ v + 0.3 * (IMG[d] @ v)}
    out = {}
    for grp, sel in (("英語 (NH 町の年次報告)", lambda d: not ja(d)), ("日本語 (広島県水産試験場)", ja)):
        idx = [k for k, r in enumerate(rows) if sel(r["doc"])]; print(f"\n{grp}: {len(idx)} 問")
        for name, fn in conds.items():
            rec = {k: 0 for k in (1, 5, 10)}
            for k in idx:
                r = rows[k]; s = fn(r["doc"], qv[k]); order = [int(i) for i in np.argsort(-s)]
                for kk in rec: rec[kk] += r["page"] in order[:kk]
                if name == "画像のみ (OCR なし)": out.setdefault(str(k), {})["img_top3"] = order[:3]
                if name == "VLM (Qwen3-VL-8B) の OCR": out.setdefault(str(k), {})["ocr_top3"] = order[:3]
            n = len(idx); print(f"  {name:24s} recall@1 {rec[1]/n:.3f} @5 {rec[5]/n:.3f} @10 {rec[10]/n:.3f}")
    json.dump(out, open(f"{S}/retrieval.json", "w"))
async def judge():
    rows = [json.loads(l) for l in open(f"{S}/qa.jsonl")]; ret = json.load(open(f"{S}/retrieval.json")); vlm = json.load(open(f"{S}/answers_vlm.json")) if os.path.exists(f"{S}/answers_vlm.json") else {}
    async def one(k, r):
        pages = ret[str(k)]["ocr_top3"]; ctx = "\n\n".join(f"=== page {p+1} ===\n{OCR[r['doc']][p][:5000]}" for p in pages)
        a27 = await chat((f"以下は文書の抜粋 (OCR) です。抜粋だけを根拠に質問に簡潔に答えてください。\n\n{ctx}\n\n質問: {r['question']}\n答え:" if ja(r["doc"]) else f"Using only the excerpts below (OCR of a scanned report), answer briefly.\n\n{ctx}\n\nQuestion: {r['question']}\nAnswer:"), think=True)
        async def jd(ans):
            if not ans: return False
            t = await chat(f"Does the model answer state the same fact/number as the gold answer? Reply CORRECT or INCORRECT.\nQuestion: {r['question']}\nGold: {r['answer']}\nModel: {ans[:800]}", 5); return t.upper().startswith("CORRECT")
        return {"doc": r["doc"], "ocr27b": await jd(a27), "vlm": await jd(vlm.get(str(k), "")), "has_vlm": str(k) in vlm}
    res = await asyncio.gather(*[one(k, r) for k, r in enumerate(rows)])
    for grp, sel in (("英語", lambda d: not ja(d)), ("日本語", ja)):
        rr = [x for x in res if sel(x["doc"])]; n = len(rr)
        print(f"{grp} (n={n}): 回答正解率  VLM OCR テキスト → 27B 思考あり {np.mean([x['ocr27b'] for x in rr]):.3f}" + (f"   画像検索 → Qwen3-VL-8B が画像を読んで回答 {np.mean([x['vlm'] for x in rr if x['has_vlm']]):.3f}" if any(x["has_vlm"] for x in rr) else ""))
if PH == "gen": asyncio.run(gen())
elif PH == "retrieve": retrieve()
else: asyncio.run(judge())
