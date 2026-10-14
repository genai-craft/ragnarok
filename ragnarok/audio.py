"""音声 (会議録音など) を索引にするための小物。EmbeddingGemma 2 は音声も文字と同じ 768 次元空間に埋め込むので、書き起こしなしでも
文字の質問から該当区間を引ける (手元の実測: FLEURS ja 300 択で音声→文字 R@1 0.96、ReazonSpeech の短い TV 発話 400 択で 0.66)。
書き起こし (任意、whisper_transcriber) を付ければ本文つきの通常の索引になり、retrieve / answer もそのまま使える。"""
from __future__ import annotations

import numpy as np

SR = 16000


def load_audio(path: str, sr: int = SR) -> np.ndarray:
    """WAV/FLAC/MP3 などを 16 kHz mono float32 で読む。"""
    import soundfile as sf
    x, r = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if r != sr:
        import librosa
        x = librosa.resample(x, orig_sr=r, target_sr=sr)
    return x


def segment_bounds(n: int, seg_sec: float, hop_sec: float, sr: int = SR) -> list[tuple[int, int]]:
    """[0, n) を seg_sec 秒の区間に hop_sec ずらしで切る (最後の区間は短くてよい)。"""
    seg, hop = int(seg_sec * sr), int(hop_sec * sr)
    out, s = [], 0
    while True:
        e = min(s + seg, n)
        out.append((s, e))
        if e >= n:
            return out
        s += hop


def fmt_ts(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}" if sec >= 3600 else f"{sec // 60:02d}:{sec % 60:02d}"


def whisper_transcriber(model: str = "kotoba-tech/kotoba-whisper-v2.0", device: str = "cuda", batch_size: int = 8, language: str = "ja"):
    """transformers の pipeline で区間ごとに書き起こす callable (list[np.ndarray] -> list[str]) を返す。"""
    import torch
    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", model=model, device=device, dtype=torch.float16 if device.startswith("cuda") else torch.float32)

    def run(segs: list[np.ndarray]) -> list[str]:
        out = asr([{"raw": s, "sampling_rate": SR} for s in segs], batch_size=batch_size, generate_kwargs={"language": language, "task": "transcribe"})
        return [o["text"].strip() for o in out]

    return run
