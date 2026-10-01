"""レイアウトから木を作る (LLM なし、PageIndex flash 相当)。PDF のフォント情報 (太字・サイズ) と見出しの型 (PART / Item N / Note N) で見出し行を拾い、階層にする。
10-K のような定型文書に強い。アウトラインが無い PDF の既定の構築法。"""
from __future__ import annotations

import re
from collections import Counter

from .tree import Doc, Node

RE_PART = re.compile(r"^\s*PART\s+[IVX]+\b", re.I)
RE_ITEM = re.compile(r"^\s*ITEM\s+(\d{1,2}[A-C]?)\s*[.:\-–—]?\s*(.{0,90})$", re.I)
RE_NOTE = re.compile(r"^\s*(NOTE|Note)\s+(\d{1,2})\b")
RE_NUMSEC = re.compile(r"^\s*(\d{1,2}(\.\d{1,2}){0,2})\.?\s+([A-Z][^.]{2,80})$")
# 日本の有価証券報告書: 第一部【企業情報】 / 第1【企業の概況】 / 1【主要な経営指標等の推移】 / (1)連結経営指標等
RE_JP_PART = re.compile(r"^\s*第[一二三四五六七八九十]+部\s*【[^】]{1,30}】")
RE_JP_SEC = re.compile(r"^\s*第\s*\d{1,2}\s*【[^】]{1,40}】")
RE_JP_SUB = re.compile(r"^\s*\d{1,2}\s*【[^】]{1,40}】")


def _lines(page) -> list[tuple[str, float, bool, float]]:
    """ページの行 (文字列, フォントサイズ, 太字か, y 座標)。"""
    out = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            spans = [s for s in l["spans"] if s["text"].strip()]
            if not spans:
                continue
            txt = " ".join(s["text"].strip() for s in spans)
            size = max(s["size"] for s in spans)
            bold = all((s["flags"] & 16) or "bold" in s["font"].lower() for s in spans)
            out.append((re.sub(r"\s+", " ", txt).strip(), size, bold, l["bbox"][1]))
    return out


def tree_from_layout(pdf: str, name: str | None = None, max_children: int = 40) -> Doc:
    import pymupdf
    with pymupdf.open(pdf) as d:
        n_pages = len(d)
        pages = [p.get_text() for p in d]
        page_lines = [_lines(p) for p in d]
    # 本文のフォントサイズ (文字数で重み付けした最頻値)
    sizes = Counter()
    for pl in page_lines:
        for t, s, b, y in pl:
            sizes[round(s, 1)] += len(t)
    body = sizes.most_common(1)[0][0] if sizes else 10.0
    # ヘッダー/フッター: 多くのページに同じ行 → 除外
    freq = Counter()
    for pl in page_lines:
        for t in {t for t, *_ in pl}:
            freq[t.lower()] += 1
    boiler = {t for t, c in freq.items() if c >= max(3, 0.25 * n_pages)}
    # 目次ページ (Item が 5 行以上) は見出しとして拾わない
    toc_pages = set()
    for i, pl in enumerate(page_lines):
        if sum(1 for t, *_ in pl if RE_ITEM.match(t)) >= 5:
            toc_pages.add(i)
    heads: list[tuple[int, int, str]] = []   # (level, page0, title)
    seen_items: set[str] = set()
    # 有報は「第1【企業の概況】」の形の目次ページ (見出しが 5 行以上) も飛ばす
    for i, pl in enumerate(page_lines):
        if sum(1 for t, *_ in pl if RE_JP_SEC.match(t) or RE_JP_SUB.match(t)) >= 8:
            toc_pages.add(i)
    seen_jp: set[str] = set()
    for i, pl in enumerate(page_lines):
        if i in toc_pages:
            continue
        for t, s, b, y in pl:
            if t.lower() in boiler or len(t) < 3 or len(t) > 110:
                continue
            if RE_JP_PART.match(t):
                heads.append((1, i, t[:40])); continue
            if RE_JP_SEC.match(t):
                if t in seen_jp: continue
                seen_jp.add(t); heads.append((2, i, t[:60])); continue
            if RE_JP_SUB.match(t):
                if t in seen_jp: continue
                seen_jp.add(t); heads.append((3, i, t[:70])); continue
            if sum(ch.isdigit() for ch in t) > 0.4 * len(t):
                continue   # 数字だらけ (表の行)
            if RE_PART.match(t) and (b or s >= body + 0.5):
                heads.append((1, i, t[:60])); continue
            m = RE_ITEM.match(t)
            if m:
                key = m.group(1).upper()
                if key in seen_items:
                    continue
                seen_items.add(key); heads.append((2, i, f"Item {key}. {m.group(2).strip()}".strip(" .")))
                continue
            if RE_NOTE.match(t) and (b or s >= body + 0.5):
                heads.append((3, i, t[:90])); continue
            if (b and s >= body - 0.1 and not t.endswith((".", ",", ";")) and t[0].isalpha() and t.upper() != t) or s >= body + 2.5:
                if len(t.split()) <= 14:
                    heads.append((3, i, t[:90]))
    # 階層にする。level 3 は直前の level 2 (Item) の下、level 2 は直前の PART の下。子が多すぎる節は刈る (見出し検出の暴走対策)
    root = Node("root", name or pdf, 1, n_pages, 0)
    cur = {0: root}
    nodes: list[Node] = []
    for k, (lv, p, t) in enumerate(heads):
        parent = cur.get(lv - 1) or cur.get(lv - 2) or root
        if lv == 3 and len(parent.children) >= max_children:
            continue
        node = Node(f"n{len(nodes):04d}", t, p + 1, p + 1, lv); nodes.append(node)
        parent.children.append(node); cur[lv] = node
        for deeper in (lv + 1, lv + 2):
            cur.pop(deeper, None)
    # 終わりページ: 次の同階層以上の見出しの前まで (葉は少なくとも自分のページ)
    def fix(n: Node, end: int):
        n.end = max(n.start, end)
        for a, b in zip(n.children, n.children[1:] + [None]):
            fix(a, (b.start - 1 if b and b.start > a.start else (b.start if b else n.end)))
    fix(root, n_pages)
    return Doc(name or pdf, pages, root)
