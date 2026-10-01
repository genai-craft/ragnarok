"""葉の節ごとに「その節だけで答えられる質問」を 27B に作らせる (正解 = その節)。bench/qa.jsonl に保存。"""
import asyncio, json, os, random, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import httpx
from ragnarok.tree import Doc
random.seed(0)
DOCS = {"PRML": 60, "Regulation Best Interest_proposed rule": 50, "arxiv_gpt3": 29, "arxiv_llama2": 24, "attention-residuals": 16}
OUT = "/data/jev-rag/bench/qa.jsonl"; os.makedirs(os.path.dirname(OUT), exist_ok=True)
C = httpx.AsyncClient(timeout=300); SEM = asyncio.Semaphore(12)

async def ask(doc, leaf):
    text = doc.text(leaf, 5000)
    if len(text.strip()) < 300:
        return None
    prompt = (f"Below is one section of the document \"{doc.name}\". Write ONE specific question that can be answered only from this section, "
              f"the way a reader who has not seen the table of contents would ask it (do not mention the section number or title; do not say 'this section'). "
              f"Also give the short answer.\n\nSection:\n{text}\n\nReply with JSON only: {{\"question\": \"...\", \"answer\": \"...\"}}")
    async with SEM:
        r = await C.post("http://127.0.0.1:8310/v1/chat/completions", json={"model": "qwen27b", "messages": [{"role": "user", "content": prompt}], "max_tokens": 300, "temperature": 0.3,
                                                                           "chat_template_kwargs": {"enable_thinking": False}})
    txt = r.json()["choices"][0]["message"]["content"]
    m = re.search(r'"question"\s*:\s*"((?:[^"\\]|\\.)*)"', txt); a = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', txt)
    if not m:
        return None
    return {"doc": doc.name, "leaf": leaf.id, "title": leaf.title, "pages": [leaf.start, leaf.end], "question": m.group(1), "answer": a.group(1) if a else ""}

async def main():
    rows = []
    for name, n in DOCS.items():
        doc = Doc.load(f"/data/jev-rag/trees/{name}.json")
        leaves = [l for l in doc.root.leaves() if l.end - l.start <= 6]
        random.shuffle(leaves)
        res = await asyncio.gather(*[ask(doc, l) for l in leaves[: int(n * 1.3)]])
        got = [r for r in res if r][:n]; rows += got
        print(f"{name:45s} {len(got)} 問", flush=True)
    with open(OUT, "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(rows), "->", OUT); print(json.dumps(rows[0], ensure_ascii=False)[:300])

asyncio.run(main())
