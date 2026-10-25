"""スキャン PDF 6 冊 (534 頁) を Nemotron-3.5-Parse の vLLM サーバー (公式の推奨経路) で読む。速さ (頁/秒) と OCR 文字を記録。
  サーバー: /data/ragnarok/venv_nparse (vllm 0.32 + Parse の plugin) で `vllm serve <snapshot> --served-model-name nemotron-parse --port 8641 ...`
  出力: /data/ragnarok/scans/ocr_nparse_vllm/<name>.json = {"pages": [Markdown], "blocks": [...], "raw_tokens": [...]}
    /data/jev-rag/venv/bin/python bench/scan_ocr_parse_vllm.py [URL] [並列数]
"""
import asyncio, base64, glob, io, json, os, sys, time
os.environ.setdefault("HF_HUB_CACHE", "/data/openvons/choice_spec/hf_cache")
import httpx, pymupdf
from PIL import Image
from huggingface_hub import snapshot_download

sys.path.insert(0, snapshot_download("nvidia/Nemotron-3.5-Parse", allow_patterns=["nemotron_parse/*", "*.py"]))
from nemotron_parse.postprocessing import postprocess_document  # noqa: E402

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8641/v1"
CONC = int(sys.argv[2]) if len(sys.argv) > 2 else 64
S = "/data/ragnarok/scans"; OUT = os.environ.get("OUT", f"{S}/ocr_nparse_vllm"); os.makedirs(OUT, exist_ok=True)
PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"


def render(page):
    w, h = page.rect.width, page.rect.height; z = min(1664 / w, 2048 / h)
    px = page.get_pixmap(matrix=pymupdf.Matrix(z, z)); return Image.frombytes("RGB", (px.width, px.height), px.samples)


def b64png(img):
    buf = io.BytesIO(); img.save(buf, format="PNG"); return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


async def main():
    jobs = []
    for f in sorted(glob.glob(f"{S}/*.pdf")):
        name = os.path.basename(f)[:-4].split("-")[0] if "town" in f else "hiroshima_" + os.path.basename(f).split("_")[1]
        with pymupdf.open(f) as d:
            for i, pg in enumerate(d):
                img = render(pg); jobs.append((name, i, img.size, b64png(img)))
    print(f"{len(jobs)} pages rendered", flush=True)
    sem = asyncio.Semaphore(CONC); res = {}
    async with httpx.AsyncClient(timeout=600) as c:
        async def one(name, i, size, url):
            async with sem:
                r = await c.post(f"{URL}/chat/completions", json={
                    "model": "nemotron-parse", "max_tokens": 8744, "temperature": 0.0,
                    "repetition_penalty": 1.1, "top_k": 1, "skip_special_tokens": False,
                    "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT}, {"type": "image_url", "image_url": {"url": url}}]}]})
                j = r.json(); raw = j["choices"][0]["message"]["content"]; n = j.get("usage", {}).get("completion_tokens", 0)
            doc = postprocess_document(raw, image_size=size)
            res[(name, i)] = (doc["markdown"], [{"class": x["class"], "bbox": x["bbox"], "content": x["content"]} for x in doc["blocks"]], n)
        t0 = time.time()
        await asyncio.gather(*[one(*j) for j in jobs])
        dt = time.time() - t0
    for name in dict.fromkeys(n for n, *_ in jobs):
        idx = sorted(i for n, i in res if n == name)
        json.dump({"name": name, "pages": [res[(name, i)][0] for i in idx], "blocks": [res[(name, i)][1] for i in idx],
                   "raw_tokens": [res[(name, i)][2] for i in idx]}, open(f"{OUT}/{name}.json", "w"), ensure_ascii=False)
    toks = sum(v[2] for v in res.values())
    print(f"{len(jobs)} pages {dt:.1f}s = {len(jobs)/dt:.2f} pages/s, {toks} tokens ({toks/dt:.0f} tok/s), concurrency {CONC}", flush=True)
    capped = sum(1 for v in res.values() if v[2] >= 8700); print(f"上限 (8744 トークン) に達した頁: {capped}")


asyncio.run(main())
