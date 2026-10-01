"""文書の木 (目次の階層)。PDF の埋め込みアウトラインから作る (LLM なし)。PageIndex の木 JSON も読める。"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


@dataclass
class Node:
    id: str
    title: str
    start: int            # 1 始まりのページ (含む)
    end: int              # 含む
    level: int = 0
    children: list["Node"] = field(default_factory=list)
    summary: str = ""

    @property
    def leaf(self) -> bool:
        return not self.children

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()

    def own_pages(self) -> list[int]:
        """この節自身の本文のページ (0 始まり): 子があれば最初の子の前まで、無ければ全範囲。"""
        end = (self.children[0].start - 1) if self.children else self.end
        end = max(end, self.start)
        return list(range(self.start - 1, end))

    def leaves(self) -> list["Node"]:
        return [n for n in self.walk() if n.leaf and n.id != "root"]

    def to_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "start": self.start, "end": self.end, "level": self.level, "summary": self.summary,
                "children": [c.to_dict() for c in self.children]}

    @staticmethod
    def from_dict(d: dict) -> "Node":
        n = Node(d["id"], d["title"], d["start"], d["end"], d.get("level", 0), summary=d.get("summary", ""))
        n.children = [Node.from_dict(c) for c in d.get("children", [])]
        return n


@dataclass
class Doc:
    name: str
    pages: list[str]      # pages[i] = i+1 ページ目の本文
    root: Node

    def text(self, node: Node, max_chars: int | None = None) -> str:
        t = "\n".join(self.pages[node.start - 1 : node.end])
        return t if max_chars is None else t[:max_chars]

    def snippet(self, node: Node, n: int = 240) -> str:
        """選択肢の説明に使う短い本文: 見出しの直後から n 文字。"""
        t = self.text(node, 4000)
        i = t.lower().find(re.sub(r"^[\dA-Z]+(\.\d+)*\.?\s+", "", node.title).lower()[:30])
        s = t[i + len(node.title) :] if i >= 0 else t
        return re.sub(r"\s+", " ", s).strip()[:n]

    def by_id(self) -> dict[str, Node]:
        return {n.id: n for n in self.root.walk()}

    def save(self, path: str) -> None:
        json.dump({"name": self.name, "pages": self.pages, "root": self.root.to_dict()}, open(path, "w"), ensure_ascii=False)

    @staticmethod
    def load(path: str) -> "Doc":
        d = json.load(open(path))
        return Doc(d["name"], d["pages"], Node.from_dict(d["root"]))


def read_pages(pdf: str) -> list[str]:
    import pymupdf
    with pymupdf.open(pdf) as d:
        return [p.get_text() for p in d]


def tree_from_outline(pdf: str, name: str | None = None, min_entries: int = 5) -> Doc | None:
    """PDF の埋め込みアウトライン (level, title, page) から木を作る。無ければ None。"""
    import pymupdf
    with pymupdf.open(pdf) as d:
        toc = d.get_toc(); n_pages = len(d)
        pages = [p.get_text() for p in d]
    toc = [(lv, t.strip(), p) for lv, t, p in toc if 1 <= p <= n_pages and t.strip()]
    if len(toc) < min_entries:
        return None
    root = Node("root", name or pdf, 1, n_pages, 0)
    stack = [root]
    nodes: list[Node] = []
    for k, (lv, t, p) in enumerate(toc):
        node = Node(f"n{k:04d}", t, p, p, lv); nodes.append(node)
        while len(stack) > 1 and stack[-1].level >= lv:
            stack.pop()
        stack[-1].children.append(node); stack.append(node)
    # 終わりページ: 同じかそれより浅い次の見出しの前まで
    for k, node in enumerate(nodes):
        nxt = next((p for lv2, _, p in toc[k + 1 :] if lv2 <= node.level), None)
        node.end = max(node.start, (nxt - 1) if nxt else n_pages)
        if nxt and nxt == node.start:   # 同じページから次の節が始まる
            node.end = node.start
    return Doc(name or pdf, pages, root)


def tree_from_pageindex(tree_json: str, pdf: str, name: str | None = None) -> Doc:
    """PageIndex の standard モードの出力 {"doc_name", "structure": [{title, start_index, end_index, nodes}]} を読む。"""
    d = json.load(open(tree_json)); pages = read_pages(pdf)
    k = [0]
    def conv(x: dict, lv: int) -> Node:
        n = Node(f"n{k[0]:04d}", x["title"], int(x.get("start_index") or 1), int(x.get("end_index") or x.get("start_index") or 1), lv, summary=x.get("summary", "")); k[0] += 1
        n.children = [conv(c, lv + 1) for c in x.get("nodes") or []]
        return n
    root = Node("root", name or d.get("doc_name", pdf), 1, len(pages), 0)
    root.children = [conv(x, 1) for x in d["structure"]]
    return Doc(root.title, pages, root)


def outline_text(root: Node, max_lines: int = 400) -> str:
    """生成モデルに木を見せるときの文字列 (id, 見出し, ページ)。"""
    lines = []
    for n in root.walk():
        if n.id == "root":
            continue
        lines.append(f"{'  ' * (n.level - 1)}[{n.id}] {n.title} (pp.{n.start}-{n.end})")
    return "\n".join(lines[:max_lines])
