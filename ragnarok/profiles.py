"""構成 (profile) の選択を考えなくて済むようにする層。

  tier (段):  full  = 27B 判定+回答 (vLLM)           GPU 24GB 以上      FinanceBench 根拠 hit@5 0.91 / 正解 0.73
              light = Qwen3-4B 判定+回答 (vLLM)      GPU 10GB 以上      0.82 / 0.64
              gguf  = Qwen3.5-4B Q3 判定 + Qwen3-4B Q4 回答 (llama.cpp)  GPU 8GB (判定だけなら 4GB)  0.84 / 0.63
              nollm = 埋め込みだけ (CPU 可)。根拠ページを返すだけで判定・棄却・回答は無い        0.71 / —

  使い方:  ragnarok up            # 空き VRAM と入っているもの (vllm / llama-server) から段を選んでサーバーを立てる
           ragnarok status        # どの段のサーバーが生きているか、Engine.from_profile("auto") が何を選ぶか
           ragnarok down          # up で立てたものを止める
           Engine.from_profile("auto")   # 生きているサーバーから最良の段を選ぶ (無ければ nollm)

  環境変数:  RAGNAROK_PROFILE (auto | full | light | gguf | nollm | …)、RAGNAROK_GGUF_DIR、RAGNAROK_LLAMA_SERVER、RAGNAROK_VLLM、RAGNAROK_GPU
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

# ---- 各 profile の接続先。judge が必須、fast / answer は無ければ judge で代用する ------------------------------
PROFILES = {
    "full":  {"judge_model": "qwen27b", "judge_url": "http://127.0.0.1:8310/v1", "fast_model": "qwen3-4b", "fast_url": "http://127.0.0.1:8311/v1",
              "note": "27B 判定+回答 (vLLM、GPU 24GB 以上、4bit)。FinanceBench 根拠 hit@5 0.91 / 正解 0.73"},
    "light": {"judge_model": "qwen3-4b", "judge_url": "http://127.0.0.1:8311/v1", "fast_model": "qwen3-4b", "fast_url": "http://127.0.0.1:8311/v1",
              "note": "Qwen3-4B 判定+回答 (vLLM、GPU 10GB 級)。0.82 / 0.64"},
    "gguf":  {"kind": "llamacpp", "judge_model": "q354-q3", "judge_url": "http://127.0.0.1:8320/v1", "fast_model": "q354-q3", "fast_url": "http://127.0.0.1:8320/v1",
              "answer_model": "q34-q4", "answer_url": "http://127.0.0.1:8321/v1",
              "note": "llama.cpp: 判定 Qwen3.5-4B Q3_K_M (3.6GB) + 回答 Qwen3-4B Q4_K_M (4.2GB)。回答役が無ければ判定役で回答 (正解 0.63 → 0.49)。0.84 / 0.63"},
    "nollm": {"judge_model": None, "judge_url": None, "fast_model": None, "fast_url": None,
              "note": "LLM なし: 埋め込み (CPU 可)。判定・棄却・回答は無く、根拠ページ候補を返すだけ。0.71 / —"},
    # 以下は実験用 (auto は生きていれば light2 を light より優先する)
    "light2": {"judge_model": "qwen3.5-4b", "judge_url": "http://127.0.0.1:8313/v1", "fast_model": "qwen3.5-4b", "fast_url": "http://127.0.0.1:8313/v1",
               "answer_model": "qwen3-4b", "answer_url": "http://127.0.0.1:8311/v1", "note": "判定 Qwen3.5-4B + 回答 Qwen3-4B (vLLM 2 台、計 16GB bf16)。0.87 / 0.64"},
    "tiny":  {"judge_model": "qwen3.5-2b", "judge_url": "http://127.0.0.1:8313/v1", "fast_model": "qwen3.5-2b", "fast_url": "http://127.0.0.1:8313/v1",
              "note": "2B 判定+回答。判定が埋め込み順と変わらず回答も 0.18 — 勧めない"},
}
TIERS = ["full", "light", "gguf", "nollm"]           # 人に見せる 4 段
AUTO_ORDER = ["full", "light2", "light", "gguf", "nollm"]   # 生きているサーバーから選ぶ順

# ---- サーバーの生存確認 -------------------------------------------------------------------------------------
def alive(url: str | None, model: str | None, timeout: float = 1.5) -> bool:
    """OpenAI 互換の /models に model が載っているか。"""
    if not url:
        return False
    try:
        r = httpx.get(url.rstrip("/") + "/models", timeout=timeout)
        ids = [m.get("id") for m in (r.json().get("data") or [])]
        return (model in ids) if model else bool(ids)
    except Exception:
        return False


def probe(name: str) -> dict:
    """profile の各サーバーの生死。judge が生きていれば使える。"""
    p = PROFILES[name]
    out = {"judge": alive(p.get("judge_url"), p.get("judge_model")) if p.get("judge_url") else None}
    if p.get("fast_url") and p.get("fast_url") != p.get("judge_url"):
        out["fast"] = alive(p["fast_url"], p.get("fast_model"))
    if p.get("answer_url"):
        out["answer"] = alive(p["answer_url"], p.get("answer_model"))
    out["usable"] = (out["judge"] is None) or bool(out["judge"])
    return out


def detect() -> tuple[str, dict]:
    """生きているサーバーから最良の段を返す。無ければ nollm。"""
    for name in AUTO_ORDER:
        st = probe(name)
        if st["usable"]:
            return name, st
    return "nollm", probe("nollm")


def resolve(name: str = "auto") -> tuple[str, dict]:
    """from_profile に渡す引数を作る。auto は detect()。足りないサーバー (fast / answer) は judge に倒す。"""
    name = name or "auto"
    st = None
    if name == "auto":
        name, st = detect()
    p = {k: v for k, v in PROFILES[name].items() if k != "note"}
    st = st or probe(name)
    if st.get("fast") is False:
        p["fast_url"], p["fast_model"] = p["judge_url"], p["judge_model"]
    if st.get("answer") is False:
        p.pop("answer_url", None); p.pop("answer_model", None)
    return name, p

# ---- 環境からの推奨 (up auto) ---------------------------------------------------------------------------------
def gpus() -> list[dict]:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.total,memory.used", "--format=csv,noheader,nounits"], text=True, timeout=10)
        return [{"index": int(a), "total_gb": int(b) / 1024, "free_gb": (int(b) - int(c)) / 1024} for a, b, c in (l.split(", ") for l in out.strip().splitlines())]
    except Exception:
        return []


def have_vllm() -> str | None:
    return os.environ.get("RAGNAROK_VLLM") or shutil.which("vllm")


def have_llama() -> str | None:
    return os.environ.get("RAGNAROK_LLAMA_SERVER") or shutil.which("llama-server")


def recommend() -> tuple[str, str]:
    """(段, 理由)。空き VRAM と入っているサーバー実装で決める。"""
    g = gpus(); vllm, llama = have_vllm(), have_llama()
    if not g:
        return "nollm", "GPU が見つからない (nvidia-smi 無し)"
    best = max(g, key=lambda x: x["free_gb"]); free = best["free_gb"]; gi = best["index"]
    if free >= 24 and vllm:
        return "full", f"GPU{gi} 空き {free:.0f}GB、vllm あり"
    if free >= 8 and llama:
        return "gguf", f"GPU{gi} 空き {free:.0f}GB、llama-server あり" + ("" if free < 10 or not vllm else " (light と同等の質、起動が軽い)")
    if free >= 10 and vllm:
        return "light", f"GPU{gi} 空き {free:.0f}GB、vllm あり (llama-server 無し)"
    if free >= 4 and llama:
        return "gguf", f"GPU{gi} 空き {free:.0f}GB → 判定役だけ (回答は判定役で代用、正解 0.49)"
    why = "空き VRAM 不足" if (vllm or llama) else "vllm も llama-server も見つからない"
    return "nollm", f"GPU{gi} 空き {free:.0f}GB、{why}"

# ---- サーバーの起動・停止 -------------------------------------------------------------------------------------
HOME = Path(os.environ.get("RAGNAROK_HOME", Path.home() / ".cache" / "ragnarok"))
PIDS = HOME / "pids.json"
GGUF = {  # 段 gguf が使う重み (unsloth)
    "judge":  ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q3_K_M.gguf", 8320, "q354-q3", "16384", "2"),   # 8k × 2 slot ≈ 3.6GB
    "answer": ("unsloth/Qwen3-4B-GGUF",   "Qwen3-4B-Q4_K_M.gguf",   8321, "q34-q4",  "8192",  "1"),   # 8k × 1 slot ≈ 4.2GB
}
VLLM = {  # 段 full / light が使うモデル (vLLM)。(repo, served name, port, 必要 VRAM GB, 追加引数)
    "judge27b": ("cyankiwi/Qwen3.8-27B-AWQ-INT4", "qwen27b", 8310, 22, ["--max-num-seqs", "16", "--max-logprobs", "40", "--default-chat-template-kwargs", '{"enable_thinking": false}']),
    "fast4b":   ("Qwen/Qwen3-4B", "qwen3-4b", 8311, 8, ["--max-num-seqs", "64", "--max-logprobs", "30"]),
}


def _load_pids() -> dict:
    try:
        return json.loads(PIDS.read_text())
    except Exception:
        return {}


def _save_pids(d: dict):
    HOME.mkdir(parents=True, exist_ok=True); PIDS.write_text(json.dumps(d, indent=1))


def _spawn(key: str, cmd: list[str], gpu: int | None, env_extra: dict | None = None) -> int:
    (HOME / "logs").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ); env.update(env_extra or {})
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log = open(HOME / "logs" / f"{key}.log", "ab")
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    d = _load_pids(); d[key] = {"pid": p.pid, "cmd": cmd, "at": time.time()}; _save_pids(d)
    return p.pid


def _gguf_path(repo: str, fname: str) -> str:
    d = Path(os.environ.get("RAGNAROK_GGUF_DIR", HOME / "gguf")); d.mkdir(parents=True, exist_ok=True)
    f = d / fname
    if not f.exists():
        print(f"downloading {repo}/{fname} → {d}", file=sys.stderr, flush=True)
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo, fname, local_dir=str(d))
    return str(f)


def _wait(url: str, model: str, key: str, timeout: float = 900) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if alive(url, model):
            return True
        pid = (_load_pids().get(key) or {}).get("pid")
        if pid and not _pid_alive(pid):
            return False
        time.sleep(3)
    return False


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0); return True
    except OSError:
        return False


def up(tier: str = "auto", gpu: int | None = None, judge_only: bool = False, wait: bool = True) -> str:
    """段のサーバーを立てる。戻り値は立てた段の名前。すでに生きているものは立て直さない。"""
    if tier == "auto":
        tier, why = recommend(); print(f"→ {tier}: {why}", flush=True)
    if tier == "nollm":
        print("nollm: サーバーは不要 (埋め込みのみ)。LLM を使うなら `ragnarok up gguf` か `ragnarok up full`", flush=True); return tier
    if tier not in PROFILES:
        raise SystemExit(f"unknown tier {tier} (full | light | gguf | nollm)")
    g = gpus()
    if gpu is None:
        gpu = int(os.environ["RAGNAROK_GPU"]) if os.environ.get("RAGNAROK_GPU") else (max(g, key=lambda x: x["free_gb"])["index"] if g else None)
    free = next((x["free_gb"] for x in g if x["index"] == gpu), 0.0); total = next((x["total_gb"] for x in g if x["index"] == gpu), 0.0)
    started = []
    if tier == "gguf":
        ls = have_llama()
        if not ls:
            raise SystemExit("llama-server が見つからない (PATH か RAGNAROK_LLAMA_SERVER)。https://github.com/ggml-org/llama.cpp")
        roles = ["judge"] if (judge_only or free < 8) else ["judge", "answer"]
        if roles == ["judge"] and not judge_only:
            print(f"GPU{gpu} 空き {free:.1f}GB < 8GB → 判定役だけ立てる (回答は判定役で代用)", flush=True)
        for role in roles:
            repo, fname, port, alias, ctx, par = GGUF[role]
            if alive(f"http://127.0.0.1:{port}/v1", alias):
                print(f"{role}: :{port} はすでに生きている", flush=True); continue
            cmd = [ls, "-m", _gguf_path(repo, fname), "--port", str(port), "-ngl", "99", "-c", ctx, "--parallel", par, "-b", "2048", "--alias", alias]
            _spawn(f"gguf_{role}", cmd, gpu); started.append((role, f"http://127.0.0.1:{port}/v1", alias, f"gguf_{role}"))
    else:  # full / light / light2 → vLLM
        vl = have_vllm()
        if not vl:
            raise SystemExit("vllm が見つからない (PATH か RAGNAROK_VLLM)。GGUF で済ませるなら `ragnarok up gguf`")
        want = ["judge27b", "fast4b"] if tier == "full" else ["fast4b"]
        if tier == "full" and free < VLLM["judge27b"][3] + VLLM["fast4b"][3]:
            want = ["judge27b"]; print(f"GPU{gpu} 空き {free:.1f}GB → 27B だけ立てる (速い判定役は 27B で代用)", flush=True)
        for key in want:
            repo, name, port, need, extra = VLLM[key]
            if alive(f"http://127.0.0.1:{port}/v1", name):
                print(f"{key}: :{port} はすでに生きている", flush=True); continue
            util = f"{min(0.95, (need + 1) / total):.2f}" if total else "0.5"
            cmd = [vl, "serve", repo, "--served-model-name", name, "--port", str(port), "--max-model-len", "32768", "--gpu-memory-utilization", util, *extra]
            _spawn(f"vllm_{key}", cmd, gpu); started.append((key, f"http://127.0.0.1:{port}/v1", name, f"vllm_{key}"))
    for key, url, name, pkey in started:
        print(f"{key}: 起動中 ({url}, log: {HOME/'logs'/(pkey+'.log')})", flush=True)
    if wait:
        for key, url, name, pkey in started:
            ok = _wait(url, name, pkey)
            print(f"{key}: {'ready' if ok else 'FAILED — see ' + str(HOME/'logs'/(pkey+'.log'))}", flush=True)
    return tier


def down() -> list[str]:
    d = _load_pids(); stopped = []
    for key, v in list(d.items()):
        pid = v.get("pid")
        if pid and _pid_alive(pid):
            try:
                os.killpg(os.getpgid(pid), 15)
            except OSError:
                os.kill(pid, 15)
            stopped.append(key)
        d.pop(key, None)
    _save_pids(d)
    return stopped


def status() -> str:
    chosen, _ = detect(); rec, why = recommend(); lines = [f"auto が選ぶ段: {chosen}    (この環境への推奨: {rec} — {why})", ""]
    lines.append(f"{'段':<7} {'判定':<8} {'速い判定':<9} {'回答':<8} 説明")
    for name in TIERS + ["light2"]:
        st = probe(name); p = PROFILES[name]
        f = lambda k: ("—" if st.get(k) is None else ("up" if st[k] else "down"))
        lines.append(f"{name:<7} {f('judge'):<8} {f('fast') if 'fast' in st else '(=判定)':<9} {f('answer') if 'answer' in st else '(=判定)':<8} {p['note']}")
    d = _load_pids()
    if d:
        lines += ["", "ragnarok up が立てたもの: " + ", ".join(f"{k} (pid {v['pid']}{'' if _pid_alive(v['pid']) else ', 終了'})" for k, v in d.items())]
    return "\n".join(lines)
