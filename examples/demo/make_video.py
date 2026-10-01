"""録画 (record_video.py の webm + timeline.json) とナレーション (narration.json: Irodori TTS の wav と尺) から、字幕付きの mp4 を合成する。
使い方: python examples/demo/make_video.py  → /data/ragnarok/video/ragnarok_demo.mp4"""
import json, os, subprocess
V = os.environ.get("RAGNAROK_VIDEO", "/data/ragnarok/video")
tl = {x["scene"]: x["t"] for x in json.load(open(f"{V}/timeline.json"))}; nar = json.load(open(f"{V}/narration.json"))
webm = [f for f in os.listdir(f"{V}/raw") if f.endswith(".webm")][0]
def ts(s): h = int(s // 3600); m = int(s % 3600 // 60); sec = s % 60; return f"{h:02d}:{m:02d}:{sec:06.3f}".replace(".", ",")
srt = []; k = 1
for n in nar:   # 句点で 2 行までに分け、尺を文字数で按分
    t0 = tl[n["scene"]]; parts = [p for p in n["text"].replace("。", "。\n").split("\n") if p.strip()]
    while len(parts) > 2: parts = [parts[0] + parts[1]] + parts[2:]
    tot = sum(len(p) for p in parts); cur = t0
    for p in parts:
        d = n["dur"] * len(p) / tot; srt.append(f"{k}\n{ts(cur)} --> {ts(cur + d - 0.05)}\n{p}\n"); cur += d; k += 1
open(f"{V}/subs.srt", "w").write("\n".join(srt))
inputs = []; filt = []
for i, n in enumerate(nar):
    inputs += ["-i", f"{V}/{n['file']}"]; ms = int(tl[n["scene"]] * 1000)
    filt.append(f"[{i+1}:a]adelay={ms}|{ms},aformat=sample_rates=48000:channel_layouts=stereo[a{i}]")
mix = "".join(f"[a{i}]" for i in range(len(nar))) + f"amix=inputs={len(nar)}:normalize=0,alimiter=limit=0.9,loudnorm=I=-16:TP=-1.5:LRA=11[aout]"
vf = (f"fps=30,pad=1440:1000:0:0:color=#0d1117,subtitles={V}/subs.srt:force_style='FontName=IPAPGothic,FontSize=15,PrimaryColour=&H00FFFFFF,"
      "OutlineColour=&H00000000,Outline=1.2,Shadow=0,MarginV=22,Alignment=2'")
cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", f"{V}/raw/{webm}"] + inputs + ["-filter_complex", ";".join(filt) + ";" + mix + f";[0:v]{vf}[vout]",
       "-map", "[vout]", "-map", "[aout]", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-t", str(tl["end"]), "-movflags", "+faststart", f"{V}/ragnarok_demo.mp4"]
subprocess.run(cmd, check=True); print("->", f"{V}/ragnarok_demo.mp4")
