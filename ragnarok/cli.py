"""ragnarok の CLI: サーバーを立て、フォルダ (NAS のマウントなど) を索引化して質問する。
  ragnarok up                      # 空き VRAM と入っているもの (vllm / llama-server) から段を選んでモデルサーバーを立てる (up full | light | gguf)
  ragnarok status                  # どの段が生きているか
  ragnarok index /mnt/nas/share --store /data/ragnarok/store
  ragnarok ask   --store /data/ragnarok/store "2024年度の設備投資額は"
  ragnarok sweep --store /data/ragnarok/store "自社株買い"
  ragnarok down
段は自動 (生きているサーバーから最良を選ぶ)。固定したければ --profile full|light|gguf|nollm か環境変数 RAGNAROK_PROFILE。"""
from __future__ import annotations

import argparse
import asyncio
import json
import os


def main():
    ap = argparse.ArgumentParser(prog="ragnarok"); sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("up", help="モデルサーバーを立てる"); u.add_argument("tier", nargs="?", default="auto", choices=["auto", "full", "light", "gguf", "nollm"]); u.add_argument("--gpu", type=int); u.add_argument("--judge-only", action="store_true", help="gguf: 判定役だけ (4GB 級)"); u.add_argument("--no-wait", action="store_true")
    sub.add_parser("down", help="up で立てたものを止める"); sub.add_parser("status", help="段ごとのサーバーの生死")
    a = sub.add_parser("index"); a.add_argument("root"); a.add_argument("--store", required=True); a.add_argument("--limit", type=int)
    b = sub.add_parser("ask"); b.add_argument("question"); b.add_argument("--store", required=True); b.add_argument("--k", type=int, default=5); b.add_argument("--no-think", action="store_true"); b.add_argument("--json", action="store_true")
    c = sub.add_parser("sweep"); c.add_argument("topic"); c.add_argument("--store", required=True); c.add_argument("--max-docs", type=int, default=20)
    for p in (a, b, c):
        p.add_argument("--profile", default=os.environ.get("RAGNAROK_PROFILE", "auto"), help="auto | full | light | gguf | nollm (既定 auto)")
        p.add_argument("--judge-url", default=os.environ.get("RAGNAROK_JUDGE_URL")); p.add_argument("--judge-model", default=os.environ.get("RAGNAROK_JUDGE_MODEL"))
        p.add_argument("--fast-url", default=os.environ.get("RAGNAROK_FAST_URL")); p.add_argument("--fast-model", default=os.environ.get("RAGNAROK_FAST_MODEL"))
    args = ap.parse_args()
    if args.cmd in ("up", "down", "status"):
        from . import profiles
        if args.cmd == "up":
            profiles.up(args.tier, gpu=args.gpu, judge_only=args.judge_only, wait=not args.no_wait)
        elif args.cmd == "down":
            s = profiles.down(); print("stopped: " + (", ".join(s) if s else "(nothing)"))
        else:
            print(profiles.status())
        return
    from .engine import Engine
    from .corpus import Corpus
    over = {k: v for k, v in {"judge_url": args.judge_url, "judge_model": args.judge_model, "fast_url": args.fast_url, "fast_model": args.fast_model}.items() if v}
    eng = Engine.from_profile(args.profile, **over)
    print(f"[ragnarok] profile={eng.profile} judge={eng.judge.model if eng.judge else 'none'}", flush=True)
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
            if not eng.judge: print("(nollm: 回答役が無いので根拠ページだけ。`ragnarok up` で LLM を立てると回答もできる)"); return
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
