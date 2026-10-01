"""layout 木の全ノードに 27B の短い要約を付ける (節自身のページの本文から)。"""
import asyncio, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import httpx
from ragnarok.tree import Doc
FB = "/data/jev-rag/financebench"; C = httpx.AsyncClient(timeout=300); SEM = asyncio.Semaphore(24)
async def summ(doc, n):
    body = "\n".join(doc.pages[p] for p in n.own_pages())[:5000]
    if n.children: body += "\nSubsections: " + "; ".join(c.title for c in n.children[:30])
    if len(body.strip()) < 40: return n.title
    prompt = f"This is a section of the 10-K filing \"{doc.name}\". In at most 40 words, say what information it contains (key metrics, tables, topics, years), so one can decide whether a question is answered here.\n\nSection title: {n.title}\n\n{body}\n\nSummary:"
    async with SEM:
        r = await C.post("http://127.0.0.1:8310/v1/chat/completions", json={"model": "qwen27b", "messages": [{"role": "user", "content": prompt}], "max_tokens": 70, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
    return r.json()["choices"][0]["message"]["content"].strip().replace("\n", " ")
async def main():
    t0 = time.time(); done = 0
    for f in sorted(os.listdir(f"{FB}/trees_layout")):
        doc = Doc.load(f"{FB}/trees_layout/{f}")
        if not os.path.exists(f"{FB}/pages/{doc.name}.json"): continue
        doc.pages = json.load(open(f"{FB}/pages/{doc.name}.json"))
        nodes = [n for n in doc.root.walk() if n.id != "root" and not n.summary]
        if not nodes: continue
        out = await asyncio.gather(*[summ(doc, n) for n in nodes])
        for n, s in zip(nodes, out): n.summary = s
        doc.pages = []; doc.save(f"{FB}/trees_layout/{f}"); done += len(nodes)
        print(f"{doc.name:40s} {len(nodes)} nodes ({done} total, {time.time()-t0:.0f}s)", flush=True)
asyncio.run(main())
