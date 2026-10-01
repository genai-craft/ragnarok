"""script.json のナレーションを Irodori TTS (aunvox の Large サーバー :8095、caption で声質、seed 固定) で合成し narration.json を書く。"""
import json, os, subprocess, wave
V = os.environ.get("RAGNAROK_VIDEO", "/data/ragnarok/video"); TTS = os.environ.get("IRODORI_URL", "http://127.0.0.1:8095/tts")
CAP = "落ち着いた若い女性のナレーター。はっきりした発音で、少し楽しそうな明るい調子"
out = []
for i, l in enumerate(json.load(open(f"{V}/script.json"))):
    f = f"nar_{i:02d}_{l['scene']}.wav"
    subprocess.run(["curl", "-s", "-m", "180", "-X", "POST", TTS, "-H", "Content-Type: application/json", "-d", json.dumps({"text": l["text"], "caption": CAP, "seed": 7, "num_steps": 40}, ensure_ascii=False), "-o", f"{V}/{f}"], check=True)
    with wave.open(f"{V}/{f}") as w: dur = w.getnframes() / w.getframerate()
    out.append({**l, "file": f, "dur": round(dur, 2)}); print(l["scene"], round(dur, 1), "s")
json.dump(out, open(f"{V}/narration.json", "w"), ensure_ascii=False, indent=1)
