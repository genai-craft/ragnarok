"""判定バックエンドの追加実装。LlamaCppBackend: llama.cpp サーバー (OpenAI 互換) 向け — vLLM の structured_outputs の代わりに GBNF の grammar で選択肢を制約する。"""
from __future__ import annotations

from openvons.lm.backends.llm_backend import LLMBackend


class LlamaCppBackend(LLMBackend):
    """llama-server (/v1/chat/completions): `grammar` で選択肢の記号に制約し、top_logprobs から分布を読む。"""
    async def _chat(self, body: dict) -> dict:
        body = dict(body)
        so = body.pop("structured_outputs", None)
        if so and so.get("choice"):
            body["grammar"] = "root ::= " + " | ".join('"' + c.replace('"', '\\"') + '"' for c in so["choice"])
        body.pop("chat_template_kwargs", None) if not self.disable_thinking else None
        return await super()._chat(body)
