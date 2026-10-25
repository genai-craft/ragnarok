"""スキャン PDF 6 冊 (534 頁) を NVIDIA Nemotron-3.5-Parse (0.63B、OpenMDW-1.1) で読む。VLM の OCR (scan_ocr_vllm.py) との比較用。
  出力: /data/ragnarok/scans/ocr_nparse/<name>.json = {"pages": [Markdown], "blocks": [[{class, bbox, content}]], "seconds": [...]}
  経路は transformers (trust_remote_code)。まとめて推論 (BATCH) して 1 頁あたりの時間も記録する。
    CUDA_VISIBLE_DEVICES=2 /data/jev-rag/venv/bin/python bench/scan_ocr_parse.py
"""
import glob, json, os, sys, time
os.environ.setdefault("HF_HUB_CACHE", "/data/openvons/choice_spec/hf_cache")
import pymupdf, torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoProcessor, GenerationConfig

P = snapshot_download("nvidia/Nemotron-3.5-Parse"); sys.path.insert(0, P)
from nemotron_parse.postprocessing import postprocess_document  # noqa: E402

S = "/data/ragnarok/scans"; OUT = os.environ.get("OUT", f"{S}/ocr_nparse"); os.makedirs(OUT, exist_ok=True)
BATCH = int(os.environ.get("BATCH", "8"))
PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"
model = AutoModel.from_pretrained(P, trust_remote_code=True, dtype=torch.bfloat16).to("cuda").eval()
proc = AutoProcessor.from_pretrained(P, trust_remote_code=True)
gc = GenerationConfig.from_pretrained(P, trust_remote_code=True); gc.max_new_tokens = 8744; gc.do_sample = False; gc.num_beams = 1; gc.repetition_penalty = 1.1
if proc.tokenizer.pad_token_id is None:
    proc.tokenizer.pad_token = proc.tokenizer.eos_token
proc.tokenizer.padding_side = "left"


def render(page):
    """推奨の上限 1664 × 2048 (幅 × 高さ) に収まる倍率で描画。"""
    w, h = page.rect.width, page.rect.height; z = min(1664 / w, 2048 / h)
    px = page.get_pixmap(matrix=pymupdf.Matrix(z, z)); return Image.frombytes("RGB", (px.width, px.height), px.samples)


def run(imgs):
    inp = proc(images=imgs, text=[PROMPT] * len(imgs), return_tensors="pt", add_special_tokens=False, padding=True).to("cuda")
    with torch.inference_mode():
        out = model.generate(**inp, generation_config=gc)
    run.last_tokens = int(out.shape[1])
    return proc.batch_decode(out, skip_special_tokens=False)


tot_pages = 0; tot_s = 0.0
for f in sorted(glob.glob(f"{S}/*.pdf")):
    name = os.path.basename(f)[:-4].split("-")[0] if "town" in f else "hiroshima_" + os.path.basename(f).split("_")[1]
    if os.path.exists(f"{OUT}/{name}.json"):
        print(name, "skip (done)"); continue
    with pymupdf.open(f) as d:
        imgs = [render(pg) for pg in d]
    pages, blocks, secs = [], [], []
    t_doc = time.time()
    for i in range(0, len(imgs), BATCH):
        b = imgs[i:i + BATCH]; t0 = time.time()
        try:
            raws = run(b)
        except Exception as e:                       # まとめて流せないときは 1 枚ずつ
            print("batch failed, fallback to 1:", str(e)[:120], flush=True); raws = [run([x])[0] for x in b]
        dt = time.time() - t0
        print(f"  {name} {i + len(b)}/{len(imgs)}  {dt:.1f}s / {len(b)} 頁  最長 {getattr(run, 'last_tokens', 0)} トークン", flush=True)
        for img, raw in zip(b, raws):
            doc = postprocess_document(raw, image_size=img.size)
            pages.append(doc["markdown"]); blocks.append([{"class": x["class"], "bbox": x["bbox"], "content": x["content"]} for x in doc["blocks"]])
            secs.append(dt / len(b))
    el = time.time() - t_doc; tot_pages += len(imgs); tot_s += el
    json.dump({"name": name, "pages": pages, "blocks": blocks, "seconds": secs, "batch": BATCH}, open(f"{OUT}/{name}.json", "w"), ensure_ascii=False)
    print(f"{name}: {len(imgs)} pages {el:.0f}s ({len(imgs)/el:.2f} pages/s)", flush=True)
if tot_pages:
    print(f"total {tot_pages} pages {tot_s:.0f}s = {tot_pages/tot_s:.2f} pages/s (transformers, batch {BATCH})")
