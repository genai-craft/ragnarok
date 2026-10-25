"""FinanceBench: 回答器に渡すページ本文を pymupdf の平文から Nemotron-3.5-Parse の Markdown (表は HTML) に替えたら正解率が上がるか。
  検索は固定: 比較相手 (eval_rerank27b_2st_k5_think_gemma_dual.json、埋め込み gemma 本文 + 0.3 画像 → 27B 2 段判定) が選んだ 5 ページ。
  回答・採点は fb_eval.py と同じ 27B・同じプロンプト (思考あり、ANS_TOKENS 既定 2500)。Parse は vLLM サーバー (:8641)。
  条件: pymupdf (再回答、乱れの確認用) / Parse / pymupdf + Parse の表だけ追記
    /data/jev-rag/venv/bin/python bench/fb_parse_answer.py
"""
import asyncio, base64, io, json, os, re, sys, time
os.environ.setdefault("HF_HUB_CACHE", "/data/openvons/choice_spec/hf_cache")
import httpx, pymupdf
from PIL import Image
from huggingface_hub import snapshot_download

sys.path.insert(0, snapshot_download("nvidia/Nemotron-3.5-Parse", allow_patterns=["nemotron_parse/*", "*.py"]))
from nemotron_parse.postprocessing import postprocess_document  # noqa: E402

FB = "/data/jev-rag/financebench"; PURL = os.environ.get("PARSE_URL", "http://127.0.0.1:8641/v1")
REF = os.environ.get("REF", f"{FB}/eval_rerank27b_2st_k5_think_gemma_dual.json")
CACHE = f"{FB}/parse_pages"; os.makedirs(CACHE, exist_ok=True)
rows = {r["financebench_id"]: r for r in (json.loads(l) for l in open(f"{FB}/repo/data/financebench_open_source.jsonl"))}
ref = json.load(open(REF))
C = httpx.AsyncClient(timeout=900); SEM27 = asyncio.Semaphore(12); SEMP = asyncio.Semaphore(48)
PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"
_pages = {}


def pages(d):
    if d not in _pages: _pages[d] = json.load(open(f"{FB}/pages/{d}.json"))
    return _pages[d]


async def parse_page(d, p):
    f = f"{CACHE}/{d}__{p}.json"
    if os.path.exists(f): return json.load(open(f))
    with pymupdf.open(f"{FB}/repo/pdfs/{d}.pdf") as doc:
        pg = doc[p]; z = min(1664 / pg.rect.width, 2048 / pg.rect.height); px = pg.get_pixmap(matrix=pymupdf.Matrix(z, z))
        img = Image.frombytes("RGB", (px.width, px.height), px.samples)
    buf = io.BytesIO(); img.save(buf, format="PNG"); url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    async with SEMP:
        j = (await C.post(f"{PURL}/chat/completions", json={"model": "nemotron-parse", "max_tokens": 8744, "temperature": 0.0, "repetition_penalty": 1.1, "top_k": 1,
              "skip_special_tokens": False, "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT}, {"type": "image_url", "image_url": {"url": url}}]}]})).json()
    doc = postprocess_document(j["choices"][0]["message"]["content"], image_size=img.size)
    out = {"markdown": doc["markdown"], "tables": [b["content"] for b in doc["blocks"] if b["class"] == "Table"], "tokens": j.get("usage", {}).get("completion_tokens", 0)}
    json.dump(out, open(f, "w"), ensure_ascii=False); return out


async def chat27(prompt, max_tokens=300, think=False):
    async with SEM27:
        r = await C.post("http://127.0.0.1:8310/v1/chat/completions", json={"model": "qwen27b", "messages": [{"role": "user", "content": prompt}],
                         "max_tokens": (int(os.environ.get("ANS_TOKENS", "2500")) if think else max_tokens), "temperature": 0, "chat_template_kwargs": {"enable_thinking": think}})
    m = r.json()["choices"][0]["message"]; txt = (m.get("content") or "").strip()
    return txt.split("</think>")[-1].strip() if "</think>" in txt else txt


async def answer(r, ctx):
    return await chat27(think=True, prompt=f"You are a financial analyst. Using only the excerpts below from the 10-K filing of {r['company']} ({r['doc_name']}), answer the question. "
                        f"Be concise; give the number with units when the question asks for a number, and show the calculation briefly if needed. If the excerpts do not contain the answer, say \"Not found\".\n\n{ctx}\n\nQuestion: {r['question']}\nAnswer:")


async def judge(r, ans):
    j = await chat27(f"Compare a model's answer with the gold answer to a financial question. Say CORRECT if the model's answer conveys the same fact/number as the gold answer "
                     f"(numbers within 1% or rounding, units/sign consistent; extra explanation is fine). Otherwise say INCORRECT.\n\nQuestion: {r['question']}\nGold answer: {r['answer']}\n"
                     f"Gold justification: {(r.get('justification') or '')[:500]}\nModel answer: {ans[:1200]}\n\nReply with one word: CORRECT or INCORRECT.", 5)
    return j.strip().upper().startswith("CORRECT")


async def main():
    t0 = time.time(); need = sorted({(x["id"], p) for x in ref for p in x["pages"]})
    P = {}
    async def get(i, p):
        d = rows[i]["doc_name"]; P[(d, p)] = await parse_page(d, p)
    await asyncio.gather(*[get(i, p) for i, p in need])
    print(f"Parse: {len(P)} pages in {time.time()-t0:.0f}s", flush=True)
    res = []
    async def one(x):
        r = rows[x["id"]]; d = r["doc_name"]; pgs = x["pages"]
        ctx_py = "\n\n".join(f"=== page {p+1} ===\n{pages(d)[p][:6000]}" for p in pgs)
        ctx_np = "\n\n".join(f"=== page {p+1} ===\n{P[(d, p)]['markdown'][:6000]}" for p in pgs)
        ctx_mix = "\n\n".join(f"=== page {p+1} ===\n{pages(d)[p][:6000]}" + ("\n\n[tables on this page]\n" + "\n".join(P[(d, p)]["tables"])[:4000] if P[(d, p)]["tables"] else "") for p in pgs)
        out = {"id": x["id"], "type": r["question_type"], "ref_correct": x["correct"]}
        for k, ctx in (("pymupdf", ctx_py), ("parse", ctx_np), ("pymupdf+tables", ctx_mix)):
            a = await answer(r, ctx); out[k] = await judge(r, a); out[k + "_answer"] = a
        res.append(out)
    await asyncio.gather(*[one(x) for x in ref])
    n = len(res)
    print(f"n={n}  比較相手の記録 {sum(x['ref_correct'] for x in res)/n:.3f}")
    for k in ("pymupdf", "parse", "pymupdf+tables"):
        by = {}
        for x in res: by.setdefault(x["type"], []).append(x[k])
        print(f"  {k:16s} 正解率 {sum(x[k] for x in res)/n:.3f}  " + " ".join(f"{t[:8]}:{sum(v)/len(v):.2f}" for t, v in by.items()), flush=True)
    both = sum(1 for x in res if x["pymupdf"] and x["parse"]); only_p = sum(1 for x in res if x["parse"] and not x["pymupdf"]); only_m = sum(1 for x in res if x["pymupdf"] and not x["parse"])
    print(f"  両方正解 {both} / Parse だけ正解 {only_p} / pymupdf だけ正解 {only_m}")
    json.dump(res, open(f"{FB}/eval_parse_answer.json", "w"), ensure_ascii=False, indent=1)


asyncio.run(main())
