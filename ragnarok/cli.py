"""ragnarok の CLI: フォルダ (NAS のマウントなど) を索引化して質問する。
  ragnarok index /mnt/nas/share --store /data/ragnarok/store
  ragnarok ask   --store /data/ragnarok/store "2024年度の設備投資額は"
  ragnarok sweep --store /data/ragnarok/store "自社株買い" """
from __future__ import annotations

import argparse
import asyncio
import json
import os


def main():
    ap = argparse.ArgumentParser(prog="ragnarok"); sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("index"); a.add_argument("root"); a.add_argument("--store", required=True); a.add_argument("--limit", type=int)
    b = sub.add_parser("ask"); b.add_argument("question"); b.add_argument("--store", required=True); b.add_argument("--k", type=int, default=5); b.add_argument("--no-think", action="store_true"); b.add_argument("--json", action="store_true")
    c = sub.add_parser("sweep"); c.add_argument("topic"); c.add_argument("--store", required=True); c.add_argument("--max-docs", type=int, default=20)
    for p in (a, b, c):
        p.add_argument("--judge-url", default=os.environ.get("RAGNAROK_JUDGE_URL", "http://127.0.0.1:8310/v1")); p.add_argument("--judge-model", default=os.environ.get("RAGNAROK_JUDGE_MODEL", "qwen27b"))
        p.add_argument("--fast-url", default=os.environ.get("RAGNAROK_FAST_URL", "http://127.0.0.1:8311/v1")); p.add_argument("--fast-model", default=os.environ.get("RAGNAROK_FAST_MODEL", "qwen3-4b"))
    args = ap.parse_args()
    from .engine import Engine
    from .corpus import Corpus
    eng = Engine(judge_url=args.judge_url, judge_model=args.judge_model, fast_url=args.fast_url, fast_model=args.fast_model)
    co = Corpus(args.store, eng)
    if args.cmd == "index":
        print(json.dumps(co.index_dir(args.root, args.limit), ensure_ascii=False))
    elif args.cmd == "ask":
        async def run():
            r = await co.retrieve(args.question, k=args.k)
            if args.json:
                print(json.dumps({"abstain": r.abstain, "none": r.none_final, "hits": [{"doc": co.load(h.doc).name, "path": co.manifest[h.doc]["path"], "page": h.page + 1, "p": h.score} for h in r.hits]}, ensure_ascii=False, indent=1)); return
            print(f"どれでもない {r.none_final:.2f}  (候補 {len(r.candidates)} ページ / {r.docs_considered} 文書)")
            for h in r.hits: print(f"  {h.score:.2f}  {co.load(h.doc).name}  p.{h.page+1}  ({co.manifest[h.doc]['rel']})")
            if r.abstain: print("→ 棄却: どの文書にも答えが見つかりません"); return
            print("\n" + await co.answer(args.question, r.hits, think=not args.no_think))
        asyncio.run(run())
    else:
        async def run():
            sc = await co.sweep(args.topic, max_docs=args.max_docs); hits = sorted([(k, v) for k, v in sc.items() if v >= 0.5], key=lambda x: -x[1])
            print(f"{len(sc)} ページ判定、該当 {len(hits)} ページ")
            for (did, p), v in hits[:50]: print(f"  {v:.2f}  {co.load(did).name}  p.{p+1}")
        asyncio.run(run())


if __name__ == "__main__":
    main()
