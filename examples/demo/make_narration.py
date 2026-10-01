"""script.json のナレーションを Irodori TTS (aunvox の Large サーバー :8095、caption で声質、seed 固定) で合成し narration.json を書く。"""
import json, os, subprocess, wave
V = os.environ.get("RAGNAROK_VIDEO", "/data/ragnarok/video"); TTS = os.environ.get("IRODORI_URL", "http://127.0.0.1:8095/tts")
BASE = "落ち着いた若い女性のナレーター。はっきりした発音。"
# 声 (話者) は参照音声で固定し、感情・調子だけ caption で場面ごとに変える (声が途中で変わらないように)
REF = os.environ.get("IRODORI_REF", "/app/assets/tmp_ragnarok/narrator.wav")   # TTS コンテナ内のパス (ホスト ~/dev/lychee-ja/deploy/app/assets/tmp_ragnarok/)
EMOTION = {"title": "力強く、わくわくした調子で宣言するように", "pin1": "穏やかに、案内するように", "pin2": "淡々と丁寧に説明する", "pin3": "少し声を強めて、いちばん大事なところを強調する",
           "pin4": "自信を持って、落ち着いて", "abst1": "いたずらっぽく、楽しそうに", "abst2": "誇らしげに、はっきりと", "sweep1": "挑戦するように、わくわくして",
           "sweep2": "テンポよく、明るく", "ja1": "柔らかく、親しみやすく", "ja2": "嬉しそうに", "outro": "堂々と、締めくくるように"}
out = []
for i, l in enumerate(json.load(open(f"{V}/script.json"))):
    f = f"nar_{i:02d}_{l['scene']}.wav"
    # 感情 caption を場面ごとに変えると声そのものが変わって聞こえる (話者類似度は 0.95 でも調子が別人) → 既定は参照音声 + 同じ caption で統一。
    # RAGNAROK_EMOTION=1 で場面ごとの感情 caption を使う
    cap = BASE + (EMOTION.get(l["scene"], "") if os.environ.get("RAGNAROK_EMOTION") == "1" else "少し楽しそうな明るい調子で、一定のテンポ")
    body = {"text": l["text"], "caption": cap, "ref_wav": REF, "seed": 7, "num_steps": 40}
    subprocess.run(["curl", "-s", "-m", "180", "-X", "POST", TTS, "-H", "Content-Type: application/json", "-d", json.dumps(body, ensure_ascii=False), "-o", f"{V}/{f}"], check=True)
    with wave.open(f"{V}/{f}") as w: dur = w.getnframes() / w.getframerate()
    out.append({**l, "file": f, "dur": round(dur, 2)}); print(l["scene"], round(dur, 1), "s")
json.dump(out, open(f"{V}/narration.json", "w"), ensure_ascii=False, indent=1)
