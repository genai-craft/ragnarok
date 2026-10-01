"""openvons の学習済み判定モデル (凍結 Qwen + head) を jev-rag の非同期 API で使う。"""
from __future__ import annotations

import asyncio

from openvons.core.primitives import Question
from openvons.lm.backends.base import Decision
from openvons.lm.backends.model_backend import ModelBackend
from openvons.lm.models.decision_model import DecisionModel


class HeadBackend:
    """DecisionModel.from_checkpoint(path) を包み、adecide(state, [q]) を提供する。GPU 1 枚で逐次 (asyncio の lock)。"""
    def __init__(self, path: str, device: str = "cuda", mode: str = "naive"):
        self.model = DecisionModel.from_checkpoint(path, device); self.model.eval()
        self.b = ModelBackend(self.model, mode=mode); self.lock = asyncio.Lock()

    async def adecide(self, state: str, qs: list[Question]) -> list[Decision]:
        async with self.lock:
            return await asyncio.to_thread(self.b.decide, state, qs)
