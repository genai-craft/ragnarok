"""スキャン PDF の各ページを Qwen3-VL-8B に読ませてテキスト化 (高品質 OCR = 正解テキストの源)。/data/ragnarok/scans/ocr/<doc>.json"""
import glob, json, os, sys, time
os.environ.setdefault("HF_HUB_CACHE", "/data/openvons/choice_spec/hf_cache")
import pymupdf, torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
M = "Qwen/Qwen3-VL-8B-Instruct"
proc = AutoProcessor.from_pretrained(M); model = AutoModelForImageTextToText.from_pretrained(M, torch_dtype=torch.bfloat16, device_map="cuda").eval()
for f in sorted(glob.glob("/data/ragnarok/scans/*.pdf")):
    name = os.path.basename(f)[:-4].split("-")[0] if "town" in f else "hiroshima_" + os.path.basename(f).split("_")[1]
    out = f"/data/ragnarok/scans/ocr/{name}.json"
    if os.path.exists(out): continue
    pages = []; t0 = time.time()
    with pymupdf.open(f) as d:
        for i, pg in enumerate(d):
            px = pg.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)); img = Image.frombytes("RGB", (px.width, px.height), px.samples)
            msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "Transcribe all the text on this page exactly as printed, preserving tables as rows (cells separated by ' | '). Output the text only."}]}]
            inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                ids = model.generate(**inp, max_new_tokens=1800, do_sample=False)
            txt = proc.batch_decode(ids[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            pages.append(txt)
            if i % 20 == 0: print(f"  {name} p{i+1}/{len(d)} {time.time()-t0:.0f}s", flush=True)
    json.dump({"name": name, "source": f, "pages": pages}, open(out, "w"), ensure_ascii=False)
    print(f"{name}: {len(pages)} pages OCR {time.time()-t0:.0f}s", flush=True)
