"""木の探索: 質問に対してどの節に降りるかを決める。
  DecisionNavigator  : 各階層で「子の節 A〜 / どれでもない」を openvons の確率判定 (小さいモデルの logprob) で選び、確率でビーム探索
  GenerativeNavigator: 木全体を見せて節 id を生成させる (PageIndex 方式)
  EmbeddingRetriever : 節の本文の埋め込み類似度 (= 節単位のベクトル RAG)
  HybridNavigator    : 埋め込みで子を絞ってから確率判定 (+ 類似度でスコアを混ぜる)"""
from __future__ import annotations

import asyncio
import json
import math
import re
import time
from dataclasses import dataclass, field

from openvons.core.primitives import Option, Question
from openvons.lm.backends.llm_backend import LLMBackend

from .tree import Doc, Node, outline_text

NONE_ID = "none"


@dataclass
class Hit:
    node: Node
    score: float
    path: list[str] = field(default_factory=list)


@dataclass
class Result:
    hits: list[Hit]
    calls: int = 0
    latency_s: float = 0.0
    none_root: float = 0.0        # 根での「どれでもない」= 文書に無い可能性
    extra: dict = field(default_factory=dict)


def _question(q: str, node: Node, doc: Doc, children: list[Node], snippet_chars: int) -> Question:
    opts = [Option(c.id, f"{c.title} (pp.{c.start}-{c.end}): {c.summary or doc.snippet(c, snippet_chars)}") for c in children]
    opts.append(Option(NONE_ID, "None of these sections (the answer is not under this part of the document)"))
    return Question("choice", f"Which section contains the answer to this question?\nQuestion: {q}", opts, key=node.id)


class DecisionNavigator:
    def __init__(self, backend: LLMBackend, beam: int = 3, min_mass: float = 0.03, max_opts: int = 25, snippet_chars: int = 240, topk: int = 5):
        self.b = backend; self.beam = beam; self.min_mass = min_mass; self.max_opts = max_opts; self.snippet_chars = snippet_chars; self.topk = topk

    async def _children_probs(self, q: str, doc: Doc, node: Node, children: list[Node]) -> tuple[list[float], float]:
        """子ごとの確率と「どれでもない」の確率。子が多いときは max_opts ずつに分けて各束で聞き、束内の確率を束の (1 - none) で重み付けして繋ぐ。"""
        if len(children) <= self.max_opts:
            d = (await self.b.adecide(f"Document: {doc.name}\nCurrent section: {node.title}", [_question(q, node, doc, children, self.snippet_chars)]))[0]
            ps = [float(p) for p in d.probs[:-1]]; s = sum(ps) or 1.0
            return [p / s for p in ps], float(d.probs[-1])   # 子の中での相対確率 (none は別に返す)
        probs: list[float] = []; nones = []
        for i in range(0, len(children), self.max_opts):
            ch = children[i : i + self.max_opts]
            d = (await self.b.adecide(f"Document: {doc.name}\nCurrent section: {node.title}", [_question(q, node, doc, ch, self.snippet_chars)]))[0]
            nones.append(float(d.probs[-1])); probs += [float(p) for p in d.probs[:-1]]
        none = min(nones)
        s = sum(probs) or 1.0
        return [p / s for p in probs], none

    async def search(self, q: str, doc: Doc) -> Result:
        t0 = time.perf_counter(); calls = 0
        frontier = [(1.0, doc.root, [])]; leaves: list[Hit] = []; none_root = 0.0
        while frontier:
            frontier.sort(key=lambda x: -x[0])
            nxt = []
            for score, node, path in frontier[: self.beam]:
                if node.leaf:
                    leaves.append(Hit(node, score, path)); continue
                probs, none = await self._children_probs(q, doc, node, node.children); calls += 1
                if node.id == "root":
                    none_root = none
                elif none >= 0.5:   # どの子でもない = この節自身の本文 (子の前のページ) が答え
                    leaves.append(Hit(node, score * none, path + [node.id]))
                for c, p in sorted(zip(node.children, probs), key=lambda x: -x[1])[: self.beam]:
                    if score * p >= self.min_mass or not nxt:
                        nxt.append((score * p, c, path + [node.id]))
            # ビームに入らなかった葉はそのまま結果に
            for score, node, path in frontier[self.beam :]:
                if node.leaf:
                    leaves.append(Hit(node, score, path))
            frontier = nxt
        leaves.sort(key=lambda h: -h.score)
        return Result(leaves[: self.topk], calls, time.perf_counter() - t0, none_root)


class GenerativeNavigator:
    """PageIndex 方式: 木の目次を全部見せ、関係する節 id を JSON で出させる。"""
    def __init__(self, base_url: str, model: str, topk: int = 5, timeout: float = 180.0):
        import httpx
        self.base = base_url.rstrip("/"); self.model = model; self.topk = topk
        self.client = httpx.AsyncClient(timeout=timeout)

    async def search(self, q: str, doc: Doc) -> Result:
        t0 = time.perf_counter()
        prompt = (f"You are given the table of contents of a document and a question. Choose the sections (up to {self.topk}, most relevant first) "
                  f"that most likely contain the answer. If no section is relevant, return an empty list.\n\nDocument: {doc.name}\n\nTable of contents:\n{outline_text(doc.root)}\n\n"
                  f"Question: {q}\n\nReply with JSON only: {{\"node_ids\": [\"n0001\", ...]}}")
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 200, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False}}
        r = await self.client.post(f"{self.base}/chat/completions", json=body); r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"]
        ids = re.findall(r"n\d{4}", txt)
        by = doc.by_id(); hits = []
        for k, i in enumerate(dict.fromkeys(ids)):
            if i in by:
                hits.append(Hit(by[i], 1.0 / (k + 1)))
        return Result(hits[: self.topk], 1, time.perf_counter() - t0, 0.0, {"raw": txt[:200]})


class EmbeddingRetriever:
    """節 (葉) の本文の埋め込み類似度で上位を返す = 節単位のベクトル RAG。木の途中の節にも埋め込みを持つ (hybrid で使う)。"""
    def __init__(self, model: str = "Qwen/Qwen3-Embedding-0.6B", device: str = "cuda", max_chars: int = 2000, topk: int = 5):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model, device=device); self.max_chars = max_chars; self.topk = topk; self.cache: dict[str, dict] = {}

    def index(self, doc: Doc) -> None:
        nodes = [n for n in doc.root.walk() if n.id != "root"]
        texts = [f"{n.title}\n{doc.text(n, self.max_chars)}" for n in nodes]
        emb = self.m.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False)
        self.cache[doc.name] = {"ids": [n.id for n in nodes], "emb": emb}

    def sims(self, q: str, doc: Doc) -> dict[str, float]:
        if doc.name not in self.cache:
            self.index(doc)
        c = self.cache[doc.name]
        qe = self.m.encode([q], prompt_name="query", normalize_embeddings=True, show_progress_bar=False)[0]
        return dict(zip(c["ids"], (c["emb"] @ qe).tolist()))

    async def search(self, q: str, doc: Doc) -> Result:
        t0 = time.perf_counter(); s = self.sims(q, doc); by = doc.by_id()
        leaves = sorted([(v, i) for i, v in s.items() if by[i].leaf], reverse=True)
        return Result([Hit(by[i], v) for v, i in leaves[: self.topk]], 0, time.perf_counter() - t0)


class HybridNavigator(DecisionNavigator):
    """埋め込みで各階層の子を上位 prune 個に絞ってから確率判定。最終スコア = 判定の経路確率 × (1-w) + 類似度の順位 w。"""
    def __init__(self, backend: LLMBackend, emb: EmbeddingRetriever, prune: int = 8, w_sim: float = 0.0, **kw):
        super().__init__(backend, **kw); self.emb = emb; self.prune = prune; self.w_sim = w_sim

    async def search(self, q: str, doc: Doc) -> Result:
        t0 = time.perf_counter(); calls = 0
        sims = self.emb.sims(q, doc)
        sub: dict[str, float] = {}
        def fill(n: Node) -> float:   # 部分木の最大類似度 (節自身も含む)
            v = max([sims.get(n.id, -1.0)] + [fill(c) for c in n.children]); sub[n.id] = v; return v
        fill(doc.root)
        frontier = [(1.0, doc.root, [])]; leaves: list[Hit] = []; none_root = 0.0
        while frontier:
            frontier.sort(key=lambda x: -x[0]); nxt = []
            for score, node, path in frontier[: self.beam]:
                if node.leaf:
                    leaves.append(Hit(node, score, path)); continue
                ch = sorted(node.children, key=lambda c: -sub.get(c.id, -1.0))[: self.prune]
                probs, none = await self._children_probs(q, doc, node, ch); calls += 1
                if node.id == "root":
                    none_root = none
                elif none >= 0.5:
                    leaves.append(Hit(node, score * none, path + [node.id]))
                for c, p in sorted(zip(ch, probs), key=lambda x: -x[1])[: self.beam]:
                    if score * p >= self.min_mass or not nxt:
                        nxt.append((score * p, c, path + [node.id]))
            for score, node, path in frontier[self.beam :]:
                if node.leaf:
                    leaves.append(Hit(node, score, path))
            frontier = nxt
        if self.w_sim > 0 and leaves:
            by = doc.by_id(); ranked = sorted([i for i in sims if by[i].leaf], key=lambda i: -sims[i]); rank = {i: k for k, i in enumerate(ranked)}
            for h in leaves:
                h.score = (1 - self.w_sim) * h.score + self.w_sim / (1 + rank.get(h.node.id, 999))
        leaves.sort(key=lambda h: -h.score)
        return Result(leaves[: self.topk], calls, time.perf_counter() - t0, none_root)


class RerankRetriever:
    """ベクトル検索で葉を topn 取り、1 回の確率判定 (候補 + どれでもない) で並べ替える。= ベクトル RAG の rerank 段を生成 LLM から logprob 判定に置き換えたもの。"""
    def __init__(self, backend: LLMBackend, emb: EmbeddingRetriever, topn: int = 10, snippet_chars: int = 300, topk: int = 5, w_sim: float = 0.0):
        self.b = backend; self.emb = emb; self.topn = topn; self.snippet_chars = snippet_chars; self.topk = topk; self.w_sim = w_sim

    async def search(self, q: str, doc: Doc) -> Result:
        t0 = time.perf_counter(); sims = self.emb.sims(q, doc); by = doc.by_id()
        cand = [by[i] for i, _ in sorted(((i, v) for i, v in sims.items() if by[i].leaf), key=lambda x: -x[1])[: self.topn]]
        d = (await self.b.adecide(f"Document: {doc.name}", [_question(q, doc.root, doc, cand, self.snippet_chars)]))[0]
        ps = [float(p) for p in d.probs[:-1]]; none = float(d.probs[-1]); s = sum(ps) or 1.0
        hits = [Hit(c, (1 - self.w_sim) * p / s + self.w_sim / (1 + k)) for k, (c, p) in enumerate(zip(cand, ps))]
        hits.sort(key=lambda h: -h.score)
        return Result(hits[: self.topk], 1, time.perf_counter() - t0, none)
