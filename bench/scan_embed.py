"""スキャン PDF の全ページ画像を embeddinggemma-2 で埋め込む (OCR なしの検索用)。IA の OCR 文字層のテキストも保存。"""
import glob, json, os, time
os.environ.setdefault("HF_HUB_CACHE", "/data/openvons/choice_spec/hf_cache")
import numpy as np, pymupdf, torch
from PIL import Image
from sentence_transformers import SentenceTransformer
gm = SentenceTransformer("google/embeddinggemma-2", device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
for f in sorted(glob.glob("/data/ragnarok/scans/*.pdf")):
    name = os.path.basename(f)[:-4].split("-")[0] if "town" in f else "hiroshima_" + os.path.basename(f).split("_")[1]
    t0 = time.time(); imgs = []; ia = []
    with pymupdf.open(f) as d:
        for pg in d:
            px = pg.get_pixmap(matrix=pymupdf.Matrix(1.0, 1.0)); imgs.append(Image.frombytes("RGB", (px.width, px.height), px.samples)); ia.append(pg.get_text())
    e = gm.encode(imgs, prompt_name="document", batch_size=4, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    np.save(f"/data/ragnarok/scans/img/{name}.npy", e); json.dump({"name": name, "pages": ia}, open(f"/data/ragnarok/scans/img/{name}_ia.json", "w"), ensure_ascii=False)
    print(f"{name}: {len(imgs)} pages image-embedded {time.time()-t0:.0f}s", flush=True)
