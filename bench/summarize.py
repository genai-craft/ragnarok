"""木の全ノードに 27B で短い要約を付ける (インデックス時の一回きりのコスト)。葉 = 本文の要約、中間 = 自分のページ + 子の見出し。"""
import asyncio, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import httpx
from ragnarok.tree import Doc
C = httpx.AsyncClient(timeout=300); SEM = asyncio.Semaphore(16)

async def summ(doc, n):
    body = doc.text(n, 6000) if n.leaf else (doc.text(n, 3000) + "\nSubsections: " + "; ".join(c.title for c in n.children))
    if len(body.strip()) < 50:
        return n.title
    prompt = f"Summarize what this section of \"{doc.name}\" covers in at most 50 words, so that a reader could decide whether a question is answered here. Mention key terms, quantities and names.\n\nSection title: {n.title}\n\n{body}\n\nSummary:"
    async with SEM:
        r = await C.post("http://127.0.0.1:8310/v1/chat/completions", json={"model": "qwen27b", "messages": [{"role": "user", "content": prompt}], "max_tokens": 90, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
    return r.json()["choices"][0]["message"]["content"].strip().replace("\n", " ")

async def main():
    for f in sorted(os.listdir("/data/jev-rag/trees")):
        doc = Doc.load(f"/data/jev-rag/trees/{f}"); nodes = [n for n in doc.root.walk() if n.id != "root"]
        t = time.time(); out = await asyncio.gather(*[summ(doc, n) for n in nodes])
        for n, s in zip(nodes, out): n.summary = s
        doc.save(f"/data/jev-rag/trees/{f}")
        print(f"{doc.name:45s} {len(nodes)} nodes {time.time()-t:.0f}s  e.g. {nodes[3].title} -> {out[3][:100]}", flush=True)

asyncio.run(main())
