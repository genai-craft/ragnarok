"""複数文書 (フォルダ / NAS / データレイク) をまとめて扱う。
  Corpus.index_dir(root)  : 配下の PDF/DOCX/PPTX/HTML/TXT/MD を再帰的に索引化 (更新時刻とサイズで差分のみ再索引)。store/<id>/ に保存
  Corpus.retrieve(q)      : 全文書のページ埋め込みを横断して候補 top-N → 確率判定 2 段 (選択肢は「文書名 p.N」) → ページ + P(どれでもない)
  Corpus.sweep(topic)     : 文書を埋め込みで絞り、ページごとの yes/no 判定で網羅
ページ埋め込みは全文書を 1 つの行列にまとめて持つ (12k ページで数十 MB、100 万ページ級なら faiss を入れると自動で使う)。"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field

import numpy as np

from openvons.core.primitives import Option, Question

from .engine import Engine, Index, Stage

EXTS = (".pdf", ".docx", ".pptx", ".html", ".htm", ".txt", ".md")


@dataclass
class CorpusHit:
    doc: str
    page: int
    score: float


@dataclass
class CorpusRetrieval:
    hits: list[CorpusHit]
    stages: list[Stage]
    none_final: float
    abstain: bool
    candidates: list[tuple[str, int, float]]      # (doc, page, 類似度) 埋め込みの候補
    docs_considered: int = 0
    extra: dict = field(default_factory=dict)


class Corpus:
    def __init__(self, store: str, engine: Engine):
        self.store = store; self.eng = engine; os.makedirs(store, exist_ok=True)
        self.manifest_path = os.path.join(store, "manifest.json")
        self.manifest: dict[str, dict] = json.load(open(self.manifest_path)) if os.path.exists(self.manifest_path) else {}
        self.docs: dict[str, Index] = {}
        self._mat = None; self._rows: list[tuple[str, int]] = []; self._faiss = None

    # ---------------- 索引 ----------------
    @staticmethod
    def doc_id(path: str) -> str:
        return hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:16]

    def index_dir(self, root: str, limit: int | None = None, progress=print) -> dict:
        """root 配下を再帰的に索引化。manifest の mtime/size が同じものは飛ばす。戻り値: 件数。"""
        files = []
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if fn.lower().endswith(EXTS) and not fn.startswith((".", "~$")):
                    files.append(os.path.join(dp, fn))
        files.sort(); n_new = n_skip = n_fail = 0; t0 = time.time()
        for k, f in enumerate(files[:limit] if limit else files):
            st = os.stat(f); did = self.doc_id(f); m = self.manifest.get(did)
            if m and m.get("mtime") == st.st_mtime and m.get("size") == st.st_size and os.path.exists(os.path.join(self.store, did, "pages.json")):
                n_skip += 1; continue
            try:
                ix = self.eng.index_file(f, os.path.splitext(os.path.basename(f))[0])
                ix.meta.update({"source": f, "rel": os.path.relpath(f, root), "mtime": st.st_mtime, "size": st.st_size})
                ix.save(os.path.join(self.store, did))
                self.manifest[did] = {"path": f, "rel": os.path.relpath(f, root), "name": ix.name, "pages": len(ix.pages), "mtime": st.st_mtime, "size": st.st_size, "indexed_at": time.time()}
                n_new += 1
                if progress: progress(f"[{k+1}/{len(files)}] {ix.name} ({len(ix.pages)} pages)")
            except Exception as e:  # noqa: BLE001
                n_fail += 1
                if progress: progress(f"[{k+1}/{len(files)}] FAIL {f}: {str(e)[:120]}")
            if (n_new % 10) == 0: self.save_manifest()
        # 消えたファイルは manifest から外す
        for did in [d for d, m in self.manifest.items() if not os.path.exists(m["path"])]:
            del self.manifest[did]
        self.save_manifest(); self._mat = None
        return {"files": len(files), "indexed": n_new, "skipped": n_skip, "failed": n_fail, "seconds": time.time() - t0}

    def save_manifest(self):
        json.dump(self.manifest, open(self.manifest_path, "w"), ensure_ascii=False, indent=1)

    def load(self, did: str) -> Index:
        if did not in self.docs:
            self.docs[did] = Index.load(os.path.join(self.store, did))
        return self.docs[did]

    def _matrix(self):
        """全文書のページ埋め込みを 1 つの行列に (遅延構築)。"""
        if self._mat is not None: return
        mats, rows = [], []
        for did in self.manifest:
            ix = self.load(did); mats.append(ix.emb.astype(np.float32)); rows += [(did, p) for p in range(len(ix.pages))]
        self._mat = np.concatenate(mats) if mats else np.zeros((0, 1), np.float32); self._rows = rows
        if len(rows) > 200_000:
            try:
                import faiss
                idx = faiss.IndexFlatIP(self._mat.shape[1]); idx.add(self._mat); self._faiss = idx
            except Exception:
                self._faiss = None

    def search_pages(self, q: str, topn: int = 50, per_doc: int | None = None) -> list[tuple[str, int, float]]:
        """全文書を横断してページ候補。per_doc で 1 文書あたりの上限 (多様性)。"""
        self._matrix()
        qe = self.eng.embed_query(q)
        if self._faiss is not None:
            sc, ii = self._faiss.search(qe[None], topn * 4); order = [(int(i), float(s)) for s, i in zip(sc[0], ii[0]) if i >= 0]
        else:
            s = self._mat @ qe; top = np.argpartition(-s, min(topn * 4, len(s) - 1))[: topn * 4]; order = sorted([(int(i), float(s[i])) for i in top], key=lambda x: -x[1])
        out, cnt = [], {}
        for i, sim in order:
            did, p = self._rows[i]
            if per_doc and cnt.get(did, 0) >= per_doc: continue
            cnt[did] = cnt.get(did, 0) + 1; out.append((did, p, sim))
            if len(out) >= topn: break
        return out

    # ---------------- 検索 ----------------
    async def _decide(self, q: str, cand: list[tuple[str, int]], backend, with_doc: bool = True) -> Stage:
        t0 = time.perf_counter(); opts = []
        for did, p in cand:
            ix = self.load(did); label = f"{ix.name} p.{p+1}" if with_doc else f"page {p+1}"
            opts.append(Option(f"{did}:{p}", f"[{label}] {self.eng.snip_for(q, ix.pages[p])}"))
        opts.append(Option("none", "None of these pages contains the answer"))
        qq = Question("choice", f"Which page (from any of the documents) contains the information needed to answer this question?\nQuestion: {q}", opts)
        d = (await backend.adecide("Corpus: multiple documents (the document name is shown with each page)", [qq]))[0]
        ps = [float(x) for x in d.probs[:-1]]; s = sum(ps) or 1.0
        return Stage("", [f"{a}:{b}" for a, b in cand], [p / s for p in ps], float(d.probs[-1]), time.perf_counter() - t0)

    async def retrieve(self, q: str, k: int = 5, topn: int = 50, group: int = 25, pool_each: int = 5, per_doc: int | None = 10, fast_first: bool = False, abstain_th: float = 0.5) -> CorpusRetrieval:
        cand = self.search_pages(q, topn, per_doc); pairs = [(d, p) for d, p, _ in cand]
        stages: list[Stage] = []; pool: list[tuple[str, int]] = []
        if len(pairs) > group:
            b = self.eng.fast if fast_first else self.eng.judge
            res = await asyncio.gather(*[self._decide(q, pairs[i:i+group], b) for i in range(0, len(pairs), group)])
            for j, st in enumerate(res):
                st.name = f"stage1-{j+1}"; stages.append(st)
                for c, _ in sorted(zip(st.candidates, st.probs), key=lambda x: -x[1])[:pool_each]:
                    d_, p_ = c.rsplit(":", 1); pool.append((d_, int(p_)))
        else:
            pool = pairs
        fin = await self._decide(q, pool, self.eng.judge); fin.name = "final"; stages.append(fin)
        ranked = sorted(zip(fin.candidates, fin.probs), key=lambda x: -x[1])
        hits = [CorpusHit(c.rsplit(":", 1)[0], int(c.rsplit(":", 1)[1]), pr) for c, pr in ranked[:k]]
        return CorpusRetrieval(hits, stages, fin.none, fin.none >= abstain_th, cand, len({d for d, _, _ in cand}))

    async def sweep(self, topic: str, max_docs: int = 20, max_pages: int = 2000, use_fast: bool = True) -> dict[tuple[str, int], float]:
        """文書を埋め込みで絞り (ページ候補の多い文書から)、各文書の全ページを yes/no 判定。"""
        cand = self.search_pages(topic, topn=400, per_doc=None); order: list[str] = []
        for did, _, _ in cand:
            if did not in order: order.append(did)
        out = {}; budget = max_pages
        for did in order[:max_docs]:
            ix = self.load(did)
            if budget <= 0: break
            pages = list(range(min(len(ix.pages), budget))); budget -= len(pages)
            sc = await self.eng.sweep(ix, topic, pages, use_fast=use_fast); out.update({(did, p): v for p, v in sc.items()})
        return out

    async def answer(self, q: str, hits: list[CorpusHit], think: bool = True) -> str:
        ctx = "\n\n".join(f"=== {self.load(h.doc).name} — page {h.page+1} ===\n{self.load(h.doc).pages[h.page][:5000]}" for h in hits)
        ja = bool(re.search(r"[ぁ-んァ-ン一-龥]", q))
        inst = ("以下は複数の文書からの抜粋です。抜粋だけを根拠に質問に答え、根拠には文書名とページ番号を添えてください。抜粋に答えが無ければ「文書中に見つかりません」と答えてください。" if ja else
                "Using only the excerpts below (from several documents), answer the question and cite document name and page for each fact. If the excerpts do not contain the answer, say \"Not found\".")
        return await self.eng.chat(f"{inst}\n\n{ctx}\n\nQuestion: {q}\nAnswer:", think=think)
