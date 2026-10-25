"""棄却の評価 (fb_abstain.py と同じ組): 各質問を (a) 正しい 10-K、(b) 別会社の 10-K にぶつけ、答えが「無い」方を見分けられるか (AUROC)。
  Nemotron Rerank: 1 段目 (gemma 本文 + 0.3·画像) の上位 50 頁の最高点 (生の logit)。高いほど「ある」。
  Rerank → 27B: Rerank の上位 10 頁を 27B の確率判定に渡したときの「どれでもない」確率 (nrr50_27b と同じ構成)。
  参考: 今の ragnarok (27B の 2 段判定の none 確率) は abstain_27B.json。
    CUDA_VISIBLE_DEVICES=2 RAGNAROK_EMBED=google/embeddinggemma-2 EMB_DIR=emb_gemma EMB_IMG_DIR=/data/ragnarok/lake/emb_img IMG_W=0.3 \\
      /data/jev-rag/venv/bin/python bench/fb_abstain_nrr.py
"""
import asyncio, json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_CACHE", "/data/openvons/choice_spec/hf_cache")
import numpy as np
sys.argv = ["x", "emb"]
import importlib.util
spec = importlib.util.spec_from_file_location("fb", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fb_eval.py")); fb = importlib.util.module_from_spec(spec); spec.loader.exec_module(fb)
random.seed(0)
ROWS = fb.ROWS; docs = sorted({r["doc_name"] for r in ROWS})


MODE = os.environ.get("ABSTAIN_MODE", "other_company")


def year(d):
    import re
    m = re.search(r"(?:19|20)\d\d", d); return int(m.group(0)) if m else 0


def pick_wrong(r):
    """fb_abstain.py と同じ選び方 (same_company: 同じ会社で 3 年以上離れた 10-K、無ければ対象外)。"""
    co = r["doc_name"].rsplit("_", 2)[0]
    same = [d for d in docs if d.rsplit("_", 2)[0] == co and d != r["doc_name"] and abs(year(d) - year(r["doc_name"])) >= 3] if MODE == "same_company" else []
    return random.choice(same) if same else (random.choice([d for d in docs if d.rsplit("_", 2)[0] != co]) if MODE != "same_company" else None)


def auroc(pos, neg):
    return float(np.mean([[1.0 if a > b else 0.5 if a == b else 0.0 for b in neg] for a in pos]))


LOCK = asyncio.Lock()


async def scores(r, d):
    s = fb.sims(d, r["question"]); cand = [int(i) for i in np.argsort(-s)[:50]]
    docs_ = [fb.pages(d)[p][:6000] or " " for p in cand]
    async with LOCK:
        ranked = await asyncio.to_thread(lambda: fb._nrr_model().rank(r["question"], docs_, batch_size=8, show_progress_bar=False))
    order = [cand[x["corpus_id"]] for x in ranked]
    rr = dict(r); rr["doc_name"] = d
    sc = await fb.decide_pages(rr, fb.B27, order[:10])
    return float(ranked[0]["score"]), 1.0 - sum(sc.values()), float(np.max(s))


async def main():
    pairs = [(r, w) for r, w in ((r, pick_wrong(r)) for r in ROWS) if w]   # fb_abstain.py の 27B と同じ乱数の並び
    t0 = time.time()
    res = []
    for k, (r, w) in enumerate(pairs):
        a = await scores(r, r["doc_name"]); b = await scores(r, w); res.append((a, b))
        if (k + 1) % 25 == 0: print(f"  {k+1}/{len(pairs)} {time.time()-t0:.0f}s", flush=True)
    mx_t = [a[0] for a, b in res]; mx_w = [b[0] for a, b in res]
    nn_t = [a[1] for a, b in res]; nn_w = [b[1] for a, b in res]
    sm_t = [a[2] for a, b in res]; sm_w = [b[2] for a, b in res]
    print(f"n={len(res)} ({'同じ会社の別年度' if MODE == 'same_company' else '別会社'} の 10-K)")
    print(f"  Nemotron Rerank の最高点: 正しい文書 中央値 {np.median(mx_t):.2f} / 別文書 {np.median(mx_w):.2f} | AUROC {auroc(mx_t, mx_w):.3f}")
    print(f"  Rerank 上位 10 → 27B の none 確率: 正しい {np.median(nn_t):.2f} / 別 {np.median(nn_w):.2f} | AUROC {auroc(nn_w, nn_t):.3f} | none>0.5 で棄却: 誤棄却 {np.mean(np.array(nn_t)>0.5):.2f}, 別文書の棄却 {np.mean(np.array(nn_w)>0.5):.2f}")
    print(f"  埋め込みの最大類似度: AUROC {auroc(sm_t, sm_w):.3f}")
    rf = f"{fb.FB}/abstain_27B_same_company.json" if MODE == "same_company" else f"{fb.FB}/abstain_27B.json"
    ref = json.load(open(rf)) if os.path.exists(rf) else None
    if ref:
        t = [x[0] for x in ref["true"]]; w = [x[0] for x in ref["wrong"]]
        print(f"  参考 (今の ragnarok、27B 2 段の none 確率、同じ組): AUROC {auroc(w, t):.3f} | 誤棄却 {np.mean(np.array(t)>0.5):.2f}, 別文書の棄却 {np.mean(np.array(w)>0.5):.2f}")
    json.dump({"true": [a for a, b in res], "wrong": [b for a, b in res]}, open(f"{fb.FB}/abstain_nrr_{MODE}.json", "w"))


asyncio.run(main())
