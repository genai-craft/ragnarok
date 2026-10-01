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

# 有価証券報告書 (EDINET、examples/demo/fetch_edinet.py で取得)
ED = "/data/ragnarok/edinet"
if os.path.exists(f"{ED}/meta.json"):
    eng = eng or Engine()
    QS = ["売上高（売上収益）はいくらか", "従業員数は何人か", "研究開発費はいくらか", "事業等のリスクとして挙げられているものは何か"]
    for m in json.load(open(f"{ED}/meta.json")):
        fy = m["periodEnd"][:4] + ("年3月期" if m["periodEnd"][5:7] == "03" else f"年{int(m['periodEnd'][5:7])}月期")
        key = f"edinet_{m['company']}_{m['periodEnd'][:7]}"
        if os.path.exists(f"{OUT}/{key}/pages.json"): print("skip", key); continue
        ix = eng.index_pdf(m["file"], f"{m['company']} 有価証券報告書 {fy}")
        ix.meta.update({"kind": "有価証券報告書 (EDINET)", "lang": "ja", "group": m["company"], "year": m["periodEnd"][:4], "source": m["file"], "docID": m["docID"],
                        "samples": [{"q": f"{fy}の{q}", "a": "", "pages": []} for q in QS]})
        ix.save(f"{OUT}/{key}"); print(key, len(ix.pages), "pages, tree nodes", sum(1 for _ in ix.tree.root.walk()) - 1 if ix.tree else 0)
