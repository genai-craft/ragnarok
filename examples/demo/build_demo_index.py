"""デモ用の索引を作る: FinanceBench の 10-K から数社 × 複数年度 (棄却デモ用) と、手元の日本語 PDF。/data/ragnarok/demo/<id>/"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("HF_HUB_CACHE", "/data/openvons/choice_spec/hf_cache")
import numpy as np
from ragnarok.engine import Engine, Index
from ragnarok.layout import tree_from_layout
FB = "/data/jev-rag/financebench"; OUT = "/data/ragnarok/demo"; os.makedirs(OUT, exist_ok=True)
rows = [json.loads(l) for l in open(f"{FB}/repo/data/financebench_open_source.jsonl")]
PICK = ["3M_2018_10K", "3M_2022_10K", "3M_2023_10K", "AMAZON_2017_10K", "AMAZON_2019_10K", "APPLE_2022_10K", "NIKE_2023_10K", "PEPSICO_2023_10K", "WALMART_2020_10K", "VERIZON_2022_10K"]
eng = None
for d in PICK:
    if not os.path.exists(f"{FB}/pages/{d}.json"): print("skip", d); continue
    pages = json.load(open(f"{FB}/pages/{d}.json")); emb = np.load(f"{FB}/emb/{d}.npy")
    tree = tree_from_layout(f"{FB}/repo/pdfs/{d}.pdf", d)
    samples = [{"q": r["question"], "a": r["answer"], "pages": [int(e["evidence_page_num"]) for e in r["evidence"]]} for r in rows if r["doc_name"] == d][:4]
    co, yr = d.split("_")[0], d.split("_")[1]
    ix = Index(d, pages, emb, tree, {"source": f"{FB}/repo/pdfs/{d}.pdf", "kind": "annual 10-K filing", "lang": "en", "group": co, "year": yr, "samples": samples})
    ix.save(f"{OUT}/{d}"); print(d, len(pages), "pages", len(samples), "samples")
# 日本語: 情報通信白書 (令和 7 年版、総務省)
ja = "/data/jev-rag/pdfs/soumu_r7_gaiyo.pdf"
if os.path.exists(ja):
    eng = eng or Engine()
    ix = eng.index_pdf(ja, "情報通信白書 令和7年版 (総務省)")
    ix.meta.update({"kind": "government white paper (Japanese)", "lang": "ja", "group": "総務省", "samples": [{"q": "生成AIの利用率は国別にどのくらいか", "a": "", "pages": []}, {"q": "日本のデータセンターの市場規模の見通しは", "a": "", "pages": []}]})
    ix.save(f"{OUT}/soumu_r7"); print("soumu_r7", len(ix.pages), "pages")
