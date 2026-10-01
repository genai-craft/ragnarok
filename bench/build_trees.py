"""アウトライン付き PDF から木を作って保存する (LLM なし)。"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jevrag.tree import tree_from_outline
OUT = "/data/jev-rag/trees"; os.makedirs(OUT, exist_ok=True)
for pdf in sorted(glob.glob("/data/jev-rag/pdfs/*.pdf")):
    name = os.path.basename(pdf)[:-4]
    doc = tree_from_outline(pdf, name)
    if doc is None:
        print(f"{name:45s} outline なし"); continue
    doc.save(f"{OUT}/{name}.json")
    leaves = doc.root.leaves(); kids = [len(n.children) for n in doc.root.walk() if n.children]
    print(f"{name:45s} pages {len(doc.pages):4d} nodes {sum(1 for _ in doc.root.walk())-1:4d} leaves {len(leaves):4d} max children {max(kids)} depth {max(n.level for n in doc.root.walk())}")
