"""vLLM 上の Qwen3-VL-8B (:8312) で全ページを並列 OCR。/data/ragnarok/scans/ocr/<doc>.json"""
import asyncio, base64, glob, io, json, os, time
import httpx, pymupdf
from PIL import Image
S = "/data/ragnarok/scans"; C = httpx.AsyncClient(timeout=600); SEM = asyncio.Semaphore(16)
MODEL = os.environ.get("VLM_MODEL", "qwen3-vl-8b"); OCR_DIR = os.environ.get("OCR_DIR", f"{S}/ocr"); os.makedirs(OCR_DIR, exist_ok=True)
def b64(img):
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=85); return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
async def ocr(img):
    async with SEM:
        r = await C.post("http://127.0.0.1:8312/v1/chat/completions", json={"model": MODEL, "max_tokens": 1800, "temperature": 0,
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": b64(img)}}, {"type": "text", "text": "Transcribe all the text on this page exactly as printed, preserving tables as rows (cells separated by ' | '). Output the text only."}]}]})
    return r.json()["choices"][0]["message"]["content"]
async def main():
    for f in sorted(glob.glob(f"{S}/*.pdf")):
        name = os.path.basename(f)[:-4].split("-")[0] if "town" in f else "hiroshima_" + os.path.basename(f).split("_")[1]
        out = f"{OCR_DIR}/{name}.json"
        if os.path.exists(out): continue
        t0 = time.time(); imgs = []
        with pymupdf.open(f) as d:
            for pg in d:
                px = pg.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)); imgs.append(Image.frombytes("RGB", (px.width, px.height), px.samples))
        pages = await asyncio.gather(*[ocr(im) for im in imgs])
        json.dump({"name": name, "source": f, "pages": pages}, open(out, "w"), ensure_ascii=False); print(f"{name}: {len(pages)} pages {time.time()-t0:.0f}s", flush=True)
asyncio.run(main())
