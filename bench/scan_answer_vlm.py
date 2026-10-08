"""画像検索 (OCR なし) で取った上位 3 ページの画像を Qwen3-VL-8B (vLLM :8312) に見せて回答する (スキャン文書を端から端まで OCR なしで)。"""
import asyncio, base64, io, json, os
import httpx, pymupdf
from PIL import Image
S = "/data/ragnarok/scans"; rows = [json.loads(l) for l in open(f"{S}/qa.jsonl")]; ret = json.load(open(f"{S}/retrieval.json")); src = {d: json.load(open(f"{S}/ocr/{d}.json"))["source"] for d in {r["doc"] for r in rows}}
docs = {d: pymupdf.open(p) for d, p in src.items()}; C = httpx.AsyncClient(timeout=600); SEM = asyncio.Semaphore(8)
def b64(img):
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=85); return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
async def one(k, r):
    imgs = []
    for p in ret[str(k)]["img_top3"]:
        px = docs[r["doc"]][p].get_pixmap(matrix=pymupdf.Matrix(1.3, 1.3)); imgs.append(Image.frombytes("RGB", (px.width, px.height), px.samples))
    ja = r["doc"].startswith("hiroshima")
    content = [{"type": "image_url", "image_url": {"url": b64(im)}} for im in imgs] + [{"type": "text", "text": (f"これらは古い報告書のページです。ページの内容だけを根拠に、質問に簡潔に答えてください。\n質問: {r['question']}" if ja else f"These are pages of an old scanned report. Using only what is on the pages, answer briefly.\nQuestion: {r['question']}")}]
    async with SEM:
        res = await C.post("http://127.0.0.1:8312/v1/chat/completions", json={"model": "qwen3-vl-8b", "max_tokens": 300, "temperature": 0, "messages": [{"role": "user", "content": content}]})
    return str(k), res.json()["choices"][0]["message"]["content"]
async def main():
    out = dict(await asyncio.gather(*[one(k, r) for k, r in enumerate(rows)])); json.dump(out, open(f"{S}/answers_vlm.json", "w"), ensure_ascii=False, indent=1); print("done", len(out))
asyncio.run(main())
