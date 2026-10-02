"""複数文書 (データレイク) 版の評価: 文書名を教えずにコーパス全体 (10-K 84 冊 + 有報 26 冊 = 110 冊、約 17k ページ) から探す。
  FinanceBench 150 問: 文書 hit@1/@5、根拠ページ hit@5、棄却率。  有報 QA 130 問: 同じ。
  emb (横断ページ埋め込み top-5) vs ragnarok (top-50、1 文書あたり最大 10 → 判定 2 段)。"""
import asyncio, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_CACHE", "/data/openvons/choice_spec/hf_cache")
import numpy as np
from ragnarok.engine import Engine
from ragnarok.corpus import Corpus
eng = Engine(); co = Corpus("/data/ragnarok/store", eng); co._matrix(); print(f"corpus: {len(co.manifest)} docs, {len(co._rows)} pages", flush=True)
name2id = {m["name"]: did for did, m in co.manifest.items()}
fb = [json.loads(l) for l in open("/data/jev-rag/financebench/repo/data/financebench_open_source.jsonl")]
ya = [json.loads(l) for l in open("/data/ragnarok/edinet/qa.jsonl")]
edmeta = {os.path.splitext(os.path.basename(m["file"]))[0]: m for m in json.load(open("/data/ragnarok/edinet/meta.json"))}
def ed_docname(r):  # qa.jsonl の doc = edinet_<社>_<YYYY-MM> → lake の名前 <社>_<periodEnd>
    co_, ym = r["doc"].split("_", 1)[1].rsplit("_", 1); return next((n for n in name2id if n.startswith(co_ + "_" + ym)), None)
TASKS = [("FinanceBench 150", [(r["question"], name2id.get(r["doc_name"]), {int(e["evidence_page_num"]) for e in r["evidence"]}) for r in fb]),
         ("有報 QA 130", [(r["question"], name2id.get(ed_docname(r) or ""), {r["page"]}) for r in ya])]
async def main():
    sem = asyncio.Semaphore(8)
    for title, items in TASKS:
        items = [x for x in items if x[1]]
        async def one(q, did, ev):
            async with sem:
                cand = co.search_pages(q, 50, per_doc=10); r = await co.retrieve(q, k=5)
            emb5 = [(d, p) for d, p, _ in cand[:5]]; rr5 = [(h.doc, h.page) for h in r.hits]
            return {"emb_doc1": emb5[0][0] == did, "emb_doc5": any(d == did for d, _ in emb5), "emb_page5": any(d == did and p in ev for d, p in emb5),
                    "rr_doc1": rr5[0][0] == did if rr5 else False, "rr_doc5": any(d == did for d, _ in rr5), "rr_page5": any(d == did and p in ev for d, p in rr5), "none": r.none_final, "abstain": r.abstain}
        t0 = time.time(); res = await asyncio.gather(*[one(*x) for x in items]); n = len(res); m = lambda k: np.mean([x[k] for x in res])
        print(f"\n{title} (n={n}, {len(co.manifest)} 冊から探す, {time.time()-t0:.0f}s)")
        print(f"  emb (横断 top-5):        文書 hit@1 {m('emb_doc1'):.3f}  文書 hit@5 {m('emb_doc5'):.3f}  根拠ページ hit@5 {m('emb_page5'):.3f}")
        print(f"  ragnarok (判定 2 段):     文書 hit@1 {m('rr_doc1'):.3f}  文書 hit@5 {m('rr_doc5'):.3f}  根拠ページ hit@5 {m('rr_page5'):.3f}   棄却 {m('abstain'):.2f} (none 中央値 {np.median([x['none'] for x in res]):.2f})", flush=True)
        json.dump(res, open(f"/data/ragnarok/lake/eval_{title.split()[0]}.json", "w"), ensure_ascii=False)
asyncio.run(main())
