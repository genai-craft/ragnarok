"""NVIDIA Nemotron-3.5-Embed (1B / 8B、OpenMDW-1.1) でページを埋め込む。embeddinggemma-2 (今の既定) との比較用。
  集合: fb (FinanceBench 84 冊、12,013 頁) / ya (有報 26 冊) / scan (スキャン 6 冊 534 頁)
  種類: text (ページ本文) / img (ページ画像のみ) / it (画像 + 本文を 1 つの文書として)
  スキャンは本文が無いので text = OCR の文字 (VL-8B の OCR と Nemotron Parse の OCR の両方)。
  出力: /data/ragnarok/nemo/<tag>/<set>_<kind>/<doc>.npy と timing.json
    CUDA_VISIBLE_DEVICES=0 /data/jev-rag/venv/bin/python bench/nemo_embed.py 1b fb,ya,scan text,img,it
"""
import json, os, sys, time
os.environ.setdefault("HF_HUB_CACHE", "/data/openvons/choice_spec/hf_cache")
import numpy as np, pymupdf, torch
from PIL import Image
from sentence_transformers import SentenceTransformer

TAG = sys.argv[1] if len(sys.argv) > 1 else "1b"
SETS = (sys.argv[2] if len(sys.argv) > 2 else "fb,ya,scan").split(",")
KINDS = (sys.argv[3] if len(sys.argv) > 3 else "text,img,it").split(",")
MODEL = {"1b": "nvidia/Nemotron-3.5-Embed-1B-BF16", "8b": "nvidia/Nemotron-3.5-Embed-8B-BF16"}[TAG]
OUT = f"/data/ragnarok/nemo/nemo{TAG}"
FB = "/data/jev-rag/financebench"; DEMO = "/data/ragnarok/demo"; SC = "/data/ragnarok/scans"
LONG = 1120                                                   # モデル既定の画像の長辺

m = SentenceTransformer(MODEL, device="cuda", trust_remote_code=True, model_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa"})


def render(pdf, sel=None):
    out = []
    with pymupdf.open(pdf) as d:
        for i, pg in enumerate(d):
            if sel is not None and i not in sel: continue
            z = LONG / max(pg.rect.width, pg.rect.height)
            px = pg.get_pixmap(matrix=pymupdf.Matrix(z, z)); out.append(Image.frombytes("RGB", (px.width, px.height), px.samples))
    return out


def docs_of(s):
    """(名前, ページ本文のリスト, PDF のパス)。"""
    if s == "fb":
        fb = [json.loads(l) for l in open(f"{FB}/repo/data/financebench_open_source.jsonl")]
        for d in sorted({r["doc_name"] for r in fb}):
            yield d, json.load(open(f"{FB}/pages/{d}.json")), f"{FB}/repo/pdfs/{d}.pdf"
    elif s == "ya":
        ya = [json.loads(l) for l in open("/data/ragnarok/edinet/qa.jsonl")]
        meta = {os.path.basename(x["file"])[:-4]: x["file"] for x in json.load(open("/data/ragnarok/edinet/meta.json"))}
        for d in sorted({r["doc"] for r in ya}):
            co, ym = d.split("_", 1)[1].rsplit("_", 1)
            yield d, json.load(open(f"{DEMO}/{d}/pages.json"))["pages"], next(v for k, v in meta.items() if k.startswith(co + "_" + ym))
    elif s == "scan":
        import glob
        for f in sorted(glob.glob(f"{SC}/*.pdf")):
            name = os.path.basename(f)[:-4].split("-")[0] if "town" in f else "hiroshima_" + os.path.basename(f).split("_")[1]
            yield name, json.load(open(f"{SC}/ocr/{name}.json"))["pages"], f


def enc(items, bs):
    return m.encode_document(items, batch_size=bs, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True).astype(np.float32)


timing = json.load(open(f"{OUT}/timing.json")) if os.path.exists(f"{OUT}/timing.json") else {}
for s in SETS:
    for kind in KINDS:
        sub = f"{s}_{kind}"; os.makedirs(f"{OUT}/{sub}", exist_ok=True); t0 = time.time(); n = 0
        for d, pages, pdf in docs_of(s):
            f = f"{OUT}/{sub}/{d}.npy"
            if os.path.exists(f): continue
            texts = [(p or " ")[:6000] for p in pages]
            if kind == "text":
                e = enc(texts, 8)
            else:
                imgs = render(pdf)
                if len(imgs) != len(texts):                       # ページ数が本文と食い違う PDF は短い方に揃える
                    k = min(len(imgs), len(texts)); imgs, texts = imgs[:k], texts[:k]
                items = [{"image": im} for im in imgs] if kind == "img" else [{"image": im, "text": t} for im, t in zip(imgs, texts)]
                e = np.concatenate([enc(items[i:i + 32], 4) for i in range(0, len(items), 32)])
            np.save(f, e); n += len(e)
            if s == "scan" and kind == "text":                     # スキャンは Parse の OCR 文字でも
                pp = f"{SC}/ocr_nparse_vllm/{d}.json"
                if os.path.exists(pp):
                    os.makedirs(f"{OUT}/scan_textparse", exist_ok=True)
                    np.save(f"{OUT}/scan_textparse/{d}.npy", enc([(p or ' ')[:6000] for p in json.load(open(pp))["pages"]], 8))
        if n:
            dt = time.time() - t0; timing[sub] = {"pages": n, "seconds": round(dt, 1), "ms_per_page": round(1000 * dt / n, 1)}
            print(f"{TAG} {sub}: {n} pages {dt:.0f}s ({1000*dt/n:.0f} ms/page)", flush=True)
            json.dump(timing, open(f"{OUT}/timing.json", "w"), indent=1)
print("done", TAG)
