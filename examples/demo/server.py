"""ragnarok デモ: 文書を選ぶ/アップロード → 質問 → 候補 50 → 確率判定 2 段 → 回答、を段階ごとに見せる。網羅モード (全ページ判定) も。
起動: CUDA_VISIBLE_DEVICES=5 python -m examples.demo.server --port 8608   (vLLM: 27B :8310, 4B :8311 が必要)"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
os.environ.setdefault("HF_HUB_CACHE", "/data/openvons/choice_spec/hf_cache")

from fastapi import FastAPI, File, Form, UploadFile  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from ragnarok.engine import Engine, Index  # noqa: E402

DATA = Path(os.environ.get("RAGNAROK_DATA", "/data/ragnarok/demo"))
app = FastAPI(title="ragnarok demo")
G: dict = {"docs": {}}


def load_docs():
    docs = {}
    if DATA.exists():
        for d in sorted(DATA.iterdir()):
            if (d / "pages.json").exists():
                try:
                    ix = Index.load(str(d)); docs[d.name] = ix
                except Exception as e:  # noqa: BLE001
                    print("skip", d, e)
    return docs


@app.on_event("startup")
def _start():
    G["engine"] = Engine(); G["docs"] = load_docs()
    print(f"{len(G['docs'])} docs loaded", flush=True)


@app.get("/healthz")
def healthz():
    return {"ok": True, "docs": len(G["docs"])}


@app.get("/api/docs")
def docs():
    out = []
    for k, ix in G["docs"].items():
        src = ix.meta.get("source") or ""
        out.append({"id": k, "name": ix.name, "pages": len(ix.pages), "lang": ix.meta.get("lang", "en"), "kind": ix.meta.get("kind", ""), "group": ix.meta.get("group", ""),
                    "tree": bool(ix.tree), "samples": ix.meta.get("samples", []), "pdf": bool(src.lower().endswith(".pdf") and os.path.exists(src)), "unit": ix.meta.get("unit", "page")})
    return {"docs": out}


@app.get("/api/page")
def page(doc: str, p: int):
    ix = G["docs"].get(doc)
    if ix is None or not (0 <= p < len(ix.pages)):
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"doc": doc, "p": p, "text": ix.pages[p][:8000], "section": ix.section_path(p)}


@app.get("/api/page_image")
def page_image(doc: str, p: int, zoom: float = 1.6):
    """PDF のページを画像で返す (元 PDF がある文書のみ)。"""
    ix = G["docs"].get(doc); src = ix.meta.get("source") if ix else None
    if ix is None or not src or not src.lower().endswith(".pdf") or not os.path.exists(src) or not (0 <= p < len(ix.pages)):
        return JSONResponse({"error": "no pdf"}, status_code=404)
    import pymupdf
    cache = DATA / "_pagecache"; cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"{re.sub(r'[^A-Za-z0-9_-]', '_', doc)}_{p}_{zoom}.png"
    if not f.exists():
        with pymupdf.open(src) as d:
            f.write_bytes(d[p].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png"))
    return Response(f.read_bytes(), media_type="image/png", headers={"Cache-Control": "max-age=86400"})


@app.get("/api/tree")
def tree(doc: str):
    ix = G["docs"].get(doc)
    if ix is None or ix.tree is None:
        return {"tree": None}
    return {"tree": ix.tree.root.to_dict()}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), name: str = Form("")):
    data = await file.read()
    if len(data) > 100 * 1024 * 1024:
        return JSONResponse({"error": "PDF は 100MB まで"}, status_code=413)
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False); tmp.write(data); tmp.close()
    nm = re.sub(r"[^\w\-぀-鿿]+", "_", (name or Path(file.filename or "doc").stem))[:60]
    t0 = time.time()
    ix = await asyncio.to_thread(G["engine"].index_pdf, tmp.name, nm)
    os.unlink(tmp.name)
    ix.meta["kind"] = "uploaded document"; ix.meta["lang"] = "ja" if re.search(r"[ぁ-んァ-ン一-龥]", "".join(ix.pages[:3])) else "en"
    key = f"up_{int(time.time())}_{nm}"; G["docs"][key] = ix
    return {"id": key, "name": ix.name, "pages": len(ix.pages), "tree": bool(ix.tree), "elapsed_s": time.time() - t0}


def sse(obj) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


@app.get("/api/ask")
async def ask(doc: str, q: str, k: int = 5, think: int = 1, fast: int = 0, path: int = 0):
    """SSE: emb → stage1 (25+25) → final → answer。段ごとに流す。"""
    ix = G["docs"].get(doc)
    if ix is None:
        return JSONResponse({"error": "doc not found"}, status_code=404)
    eng: Engine = G["engine"]

    async def gen():
        t0 = time.time()
        import numpy as np
        qe = eng.emb.encode([q], prompt_name="query", normalize_embeddings=True, show_progress_bar=False)[0]
        sims = ix.emb @ qe; order = [int(i) for i in np.argsort(-sims)[:50]]
        yield sse({"stage": "emb", "candidates": [{"p": p, "sim": float(sims[p]), "section": ix.section_path(p), "snip": eng._snip(ix.pages[p], 160)} for p in order], "t": time.time() - t0})
        pool = []
        if len(order) > 25:
            b = eng.fast if fast else eng.judge
            res = await asyncio.gather(*[eng._decide(ix, q, order[i:i+25], b, bool(path)) for i in range(0, len(order), 25)])
            for j, st in enumerate(res):
                pool += [p for p, _ in sorted(zip(st.candidates, st.probs), key=lambda x: -x[1])[:5]]
                yield sse({"stage": f"stage1-{j+1}", "candidates": st.candidates, "probs": st.probs, "none": st.none, "latency": st.latency_s, "t": time.time() - t0})
        else:
            pool = order
        fin = await eng._decide(ix, q, pool, eng.judge, bool(path))
        ranked = sorted(zip(fin.candidates, fin.probs), key=lambda x: -x[1])
        pages = [p for p, _ in ranked[:k]]
        yield sse({"stage": "final", "candidates": fin.candidates, "probs": fin.probs, "none": fin.none, "latency": fin.latency_s, "pages": pages,
                   "abstain": fin.none >= 0.5, "sections": {str(p): ix.section_path(p) for p in pages}, "t": time.time() - t0})
        if fin.none >= 0.5:
            yield sse({"stage": "answer", "text": "この文書には、この質問に答える箇所が見つかりませんでした (「どれでもない」の確率 %.0f%%)。" % (fin.none * 100), "abstained": True, "t": time.time() - t0})
        else:
            yield sse({"stage": "answer_start", "pages": pages, "think": bool(think), "t": time.time() - t0})
            acc = ""; nthink = 0
            async for kind, piece in eng.answer_stream(ix, q, pages, think=bool(think)):
                if kind == "think":
                    nthink += piece; yield sse({"stage": "answer_think", "chars": nthink, "t": time.time() - t0})
                else:
                    acc += piece; yield sse({"stage": "answer_delta", "delta": piece, "t": time.time() - t0})
            yield sse({"stage": "answer", "text": acc, "abstained": False, "t": time.time() - t0})
        yield sse({"stage": "done", "t": time.time() - t0})
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/sweep")
async def sweep(doc: str, topic: str, fast: int = 1, summarize: int = 1):
    """SSE: 全ページ yes/no 判定 (網羅) → 該当ページ一覧 → 要約。"""
    ix = G["docs"].get(doc)
    if ix is None:
        return JSONResponse({"error": "doc not found"}, status_code=404)
    eng: Engine = G["engine"]

    async def gen():
        t0 = time.time(); n = len(ix.pages); done = {}
        chunk = 32
        for i in range(0, n, chunk):
            part = await eng.sweep(ix, topic, list(range(i, min(n, i + chunk))), use_fast=bool(fast))
            done.update(part)
            yield sse({"stage": "sweep", "progress": min(n, i + chunk), "total": n, "probs": {str(p): v for p, v in part.items()}, "t": time.time() - t0})
        hits = sorted([p for p, v in done.items() if v >= 0.5], key=lambda p: -done[p])
        yield sse({"stage": "hits", "pages": hits, "sections": {str(p): ix.section_path(p) for p in hits}, "t": time.time() - t0})
        if summarize and hits:
            s = await eng.summarize_hits(ix, topic, sorted(hits))
            yield sse({"stage": "summary", "text": s, "t": time.time() - t0})
        yield sse({"stage": "done", "t": time.time() - t0})
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/", response_class=HTMLResponse)
def root():
    return (HERE / "static" / "index.html").read_text(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8608); ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
