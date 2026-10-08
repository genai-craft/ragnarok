"""スキャンの読解評価: (a) 正解ページ 1 枚を VLM に渡す上限、(b) OCR テキスト検索 top-3 の画像を VLM が読む、(c) 画像検索 top-3 の画像を VLM が読む。VLM_URL / VLM_MODEL で指定。"""
import asyncio, base64, io, json, os, httpx, pymupdf
from PIL import Image
S="/data/ragnarok/scans"; URL=os.environ.get("VLM_URL","http://127.0.0.1:8312/v1"); MODEL=os.environ.get("VLM_MODEL","qwen3-vl-8b")
rows=[json.loads(l) for l in open(f"{S}/qa.jsonl")]; ret=json.load(open(f"{S}/retrieval.json")); src={d: json.load(open(f"{S}/ocr/{d}.json"))["source"] for d in {r["doc"] for r in rows}}; docs={d: pymupdf.open(p) for d,p in src.items()}
C=httpx.AsyncClient(timeout=900); SEM=asyncio.Semaphore(4)
def b64(img):
    buf=io.BytesIO(); img.save(buf, format="JPEG", quality=85); return "data:image/jpeg;base64,"+base64.b64encode(buf.getvalue()).decode()
async def ans(r, pages):
    imgs=[]
    z = 1.5 if len(pages) == 1 else 1.1   # 複数枚は token 数を抑える
    for p in pages:
        px=docs[r["doc"]][p].get_pixmap(matrix=pymupdf.Matrix(z,z)); imgs.append(Image.frombytes("RGB",(px.width,px.height),px.samples))
    ja=r["doc"].startswith("hiroshima")
    body={"model":MODEL,"max_tokens":400,"temperature":0,"chat_template_kwargs":{"enable_thinking":False},"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":b64(im)}} for im in imgs]+[{"type":"text","text":(f"これらのページの内容だけを根拠に、質問に簡潔に答えてください。\n質問: {r['question']}" if ja else f"Using only these pages, answer briefly.\nQuestion: {r['question']}")}]}]}
    for attempt in range(3):
        try:
            async with SEM:
                res=await C.post(f"{URL}/chat/completions", json=body)
            j=res.json(); return (j.get("choices") or [{}])[0].get("message",{}).get("content","") or j.get("error",{}).get("message","")
        except Exception as e:
            if attempt == 2: return f"(error: {str(e)[:60]})"
            await asyncio.sleep(3)
async def judge(r,a):
    async with SEM:
        res=await C.post("http://127.0.0.1:8310/v1/chat/completions", json={"model":"qwen27b","max_tokens":5,"temperature":0,"chat_template_kwargs":{"enable_thinking":False},"messages":[{"role":"user","content":f"Does the model answer state the same fact/number as the gold answer? Reply CORRECT or INCORRECT.\nQuestion: {r['question']}\nGold: {r['answer']}\nModel: {a[:800]}"}]})
    return res.json()["choices"][0]["message"]["content"].strip().upper().startswith("CORRECT")
async def main():
    for label, pages_of in (("正解ページ 1 枚 (読解の上限)", lambda k,r: [r["page"]]), ("OCR テキスト検索 top-3 → 画像を読む", lambda k,r: ret[str(k)]["ocr_top3"]), ("画像検索 top-3 → 画像を読む", lambda k,r: ret[str(k)]["img_top3"])):
        A=await asyncio.gather(*[ans(r, pages_of(k,r)) for k,r in enumerate(rows)]); J=await asyncio.gather(*[judge(r,a) for r,a in zip(rows,A)]); out=[]
        for grp,sel in (("英語",lambda d: not d.startswith("hiroshima")),("日本語",lambda d: d.startswith("hiroshima"))):
            idx=[i for i,r in enumerate(rows) if sel(r["doc"])]; out.append(f"{grp} {sum(J[i] for i in idx)/len(idx):.3f}")
        print(f"[{MODEL}] {label:34s} " + " / ".join(out), flush=True)
asyncio.run(main())
