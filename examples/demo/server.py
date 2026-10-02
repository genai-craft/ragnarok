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
from ragnarok.corpus import Corpus  # noqa: E402

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


def build_corpus():
    """読み込み済みの全文書を 1 つのコーパスとして横断検索できるようにする (store は使わず docs を直接注入)。"""
    co = Corpus(str(DATA / "_corpus"), G["engine"]); co.manifest = {}; co.docs = {}
    for k, ix in G["docs"].items():
        co.manifest[k] = {"path": ix.meta.get("source", ""), "rel": ix.name, "name": ix.name, "pages": len(ix.pages)}; co.docs[k] = ix
    co._mat = None; return co


@app.on_event("startup")
def _start():
    G["engine"] = Engine(); G["docs"] = load_docs(); G["corpus"] = build_corpus()
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
        return JSONResponse({"error": "ファイルは 100MB まで"}, status_code=413)
    ext = Path(file.filename or "doc.pdf").suffix.lower()
    if ext not in (".pdf", ".docx", ".pptx", ".html", ".htm", ".txt", ".md"):
        return JSONResponse({"error": f"対応形式: PDF / DOCX / PPTX / HTML / TXT / MD ({ext} は未対応)"}, status_code=415)
    nm = re.sub(r"[^\w\-\u3040-\u9fff]+", "_", (name or Path(file.filename or "doc").stem))[:60]
    key = f"up_{int(time.time())}_{nm}"
    up = DATA / "_uploads"; up.mkdir(parents=True, exist_ok=True)
    path = up / f"{key}{ext}"; path.write_bytes(data)          # PDF プレビューのために元ファイルを残す
    t0 = time.time()
    ix = await asyncio.to_thread(G["engine"].index_file, str(path), nm)
    ix.meta.update({"kind": f"uploaded {ext.lstrip('.')}", "lang": "ja" if re.search(r"[ぁ-んァ-ン一-龥]", "".join(ix.pages[:3])) else "en", "source": str(path)})
    G["docs"][key] = ix; G["corpus"] = build_corpus()
    return {"id": key, "name": ix.name, "pages": len(ix.pages), "tree": bool(ix.tree), "pdf": ext == ".pdf", "unit": ix.meta.get("unit", "page"), "elapsed_s": time.time() - t0}


def sse(obj) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


@app.get("/api/ask_all")
async def ask_all(q: str, k: int = 5, think: int = 1):
    """SSE: 全文書を横断 (データレイク版)。候補は「文書:ページ」。"""
    co: Corpus = G["corpus"]; eng: Engine = G["engine"]
    short = {did: (ix.name[:18] + ("…" if len(ix.name) > 18 else "")) for did, ix in G["docs"].items()}
    async def gen():
        t0 = time.time()
        cand = co.search_pages(q, 50, per_doc=10)
        yield sse({"stage": "emb", "corpus": True, "candidates": [{"p": p, "doc": d, "label": f"{short[d]} p.{p+1}", "sim": s, "section": G["docs"][d].section_path(p), "snip": eng.snip_for(q, G["docs"][d].pages[p], 200)} for d, p, s in cand], "docs": len({d for d, _, _ in cand}), "t": time.time() - t0})
        pairs = [(d, p) for d, p, _ in cand]; pool = []
        for j in range(0, len(pairs), 25):
            st = await co._decide(q, pairs[j:j+25], eng.judge)
            for c, _ in sorted(zip(st.candidates, st.probs), key=lambda x: -x[1])[:5]:
                d_, p_ = c.rsplit(":", 1); pool.append((d_, int(p_)))
            yield sse({"stage": f"stage1-{j//25+1}", "corpus": True, "candidates": [{"doc": c.rsplit(':', 1)[0], "p": int(c.rsplit(':', 1)[1]), "label": f"{short[c.rsplit(':', 1)[0]]} p.{int(c.rsplit(':', 1)[1])+1}"} for c in st.candidates], "probs": st.probs, "none": st.none, "latency": st.latency_s, "t": time.time() - t0})
        fin = await co._decide(q, pool, eng.judge)
        ranked = sorted(zip(fin.candidates, fin.probs), key=lambda x: -x[1]); hits = [(c.rsplit(":", 1)[0], int(c.rsplit(":", 1)[1])) for c, _ in ranked[:k]]
        yield sse({"stage": "final", "corpus": True, "candidates": [{"doc": c.rsplit(':', 1)[0], "p": int(c.rsplit(':', 1)[1]), "label": f"{short[c.rsplit(':', 1)[0]]} p.{int(c.rsplit(':', 1)[1])+1}"} for c in fin.candidates], "probs": fin.probs, "none": fin.none,
                   "hits": [{"doc": d, "p": p, "name": G["docs"][d].name} for d, p in hits], "abstain": fin.none >= 0.5, "t": time.time() - t0})
        if fin.none >= 0.5:
            yield sse({"stage": "answer", "text": "どの文書にも、この質問に答える箇所が見つかりませんでした (「どれでもない」の確率 %.0f%%)。" % (fin.none * 100), "abstained": True, "t": time.time() - t0})
        else:
            yield sse({"stage": "answer_start", "pages": [], "labels": [f"{G['docs'][d].name} p.{p+1}" for d, p in hits], "think": bool(think), "t": time.time() - t0})
            from ragnarok.corpus import CorpusHit
            ans = await co.answer(q, [CorpusHit(d, p, 0.0) for d, p in hits], think=bool(think))
            yield sse({"stage": "answer", "text": ans, "abstained": False, "t": time.time() - t0})
        yield sse({"stage": "done", "t": time.time() - t0})
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


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
        yield sse({"stage": "emb", "candidates": [{"p": p, "sim": float(sims[p]), "section": ix.section_path(p), "snip": eng.snip_for(q, ix.pages[p], 200)} for p in order], "t": time.time() - t0})
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
