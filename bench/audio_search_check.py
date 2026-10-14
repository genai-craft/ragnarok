"""音声索引の確認: FLEURS ja の読み上げ 20 本を 0.3 秒の無音を挟んでつないだ「録音」を 3 本作り、各文の転記を質問にして、
書き起こしなし (音声埋め込みだけ) でその文を含む区間が 1 位に来る割合を測る。区間幅 2 通り。"""
import glob, json, os, sys, time
import numpy as np, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ragnarok.engine import Engine
man = glob.glob("/data/lychee_ja/asr_bench/fleurs_ja/*.jsonl")[0]; items = [json.loads(l) for l in open(man)][:60]
eng = Engine(judge_url="", judge_model="", fast_url="", fast_model="")
out = "/data/lychee_ja/eg2_probe/audio_search"; os.makedirs(out, exist_ok=True)
for seg, hop in ((20.0, 15.0), (10.0, 5.0)):
    hit1 = hit3 = n = 0; t_ix = 0.0
    for r in range(3):
        clips = items[r * 20:(r + 1) * 20]; spans = []; wav = []; t = 0.0
        for it in clips:
            x, sr = sf.read(it["wav"], dtype="float32"); assert sr == 16000
            spans.append((t, t + len(x) / sr)); wav.append(x); wav.append(np.zeros(4800, np.float32)); t += len(x) / sr + 0.3
        path = f"{out}/rec{r}.wav"; sf.write(path, np.concatenate(wav), 16000)
        t0 = time.time(); ix = eng.index_audio(path, seg_sec=seg, hop_sec=hop); t_ix += time.time() - t0
        for it, (s, e) in zip(clips, spans):
            res = eng.search_audio(ix, it["text"], k=3)
            ok = [max(0.0, min(e, h["end"]) - max(s, h["start"])) > 0.5 * (e - s) for h in res]
            hit1 += ok[0]; hit3 += any(ok); n += 1
    print(f"seg {seg:.0f}s/hop {hop:.0f}s: hit@1 {hit1}/{n} = {hit1/n:.2f}  hit@3 {hit3}/{n} = {hit3/n:.2f}  (索引 {t_ix:.1f}s / 3 本 ≈ {sum(float(i['duration']) for i in items)/60:.0f} 分)")
ix = eng.index_audio(f"{out}/rec0.wav"); print("paraphrase queries:")
for q in ["世界的に有名な観光地の話", "インターネットで検索すると何が出てくるか", "料理の作り方"]:
    r = eng.search_audio(ix, q, k=1)[0]; print(f"  {q} -> {r['label']} score {r['score']:.3f}")
