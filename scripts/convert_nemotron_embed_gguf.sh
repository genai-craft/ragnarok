#!/usr/bin/env bash
# NVIDIA Nemotron-3.5-Embed-1B (本文部分) を GGUF に変換する。言箱 (kotohako) の「ファイルに質問」が CPU (llama-cpp-python) で使うもの。
#   LLAMA_CPP=~/llama.cpp OUT=/data/ragnarok/gguf_embed scripts/convert_nemotron_embed_gguf.sh
# llama.cpp の convert_hf_to_gguf.py は Ministral3Model (この埋め込みモデルの本文部分) を知らないので、一時ディレクトリに写した変換部品に
#   (1) Ministral3Model / Mistral3Model を mistral3 として登録、(2) sentence-transformers の平均プーリングと双方向の注意を書く、
#   (3) BOS/EOS を付けない (検索用テンプレートは BOS/EOS を省く。付けると余弦 0.99 / 検索の 1 位一致 94% に落ちる)
# を足して変換する (llama.cpp 本体は変えない)。確認済み: 元のモデル (sentence-transformers、float32) と余弦 0.9998、検索の 1 位 100% 一致。
# 2026-10-27 の出力 (Q8_0) の sha256: 85a26840bd5095dcb245c6007782c9a844946b6d0ad79467b7e0340e576a22e1 (llama.cpp b23701f、gguf-py 同梱版)
# 注意: llama-server で双方向の埋め込みモデルを同時に流す (--parallel > 1) と別の文書が混ざって壊れる。1 件ずつ流す。
set -euo pipefail
L="${LLAMA_CPP:-/data/openvons/choice_spec/tools/llama.cpp}"; OUT="${OUT:-/data/ragnarok/gguf_embed}"; TYPE="${TYPE:-q8_0}"
PY="${PY:-python3}"; REPO="${REPO:-nvidia/Nemotron-3.5-Embed-1B-BF16}"
W="$(mktemp -d)"; trap 'rm -rf "$W"' EXIT
cp "$L/convert_hf_to_gguf.py" "$W/"; cp -r "$L/conversion" "$W/"
"$PY" - "$W" <<'PYEOF'
import pathlib, sys
W = pathlib.Path(sys.argv[1])
p = W / "conversion/mistral3.py"; s = p.read_text()
s = s.replace('''@ModelBase.register(
    "Mistral3ForConditionalGeneration",
    "Ministral3ForCausalLM",
)''', '''@ModelBase.register(
    "Mistral3ForConditionalGeneration",
    "Ministral3ForCausalLM",
    "Ministral3Model",
    "Mistral3Model",
)''')
s = s.replace('''    class Ministral3Model(LlamaModel):
        model_arch = gguf.MODEL_ARCH.MISTRAL3
''', '''    class Ministral3Model(LlamaModel):
        model_arch = gguf.MODEL_ARCH.MISTRAL3

        def set_vocab(self):
            super().set_vocab()
            if (self.dir_model / "modules.json").is_file():      # 埋め込みモデル: BOS/EOS を付けない
                self.gguf_writer.add_add_bos_token(False)
                self.gguf_writer.add_add_eos_token(False)
''', 1)
s = s.replace('''                self.gguf_writer.add_attn_temperature_scale(rope_params["llama_4_scaling_beta"])
''', '''                self.gguf_writer.add_attn_temperature_scale(rope_params["llama_4_scaling_beta"])
            if (self.dir_model / "modules.json").is_file():          # 埋め込みモデル: 平均プーリング・双方向
                self._try_set_pooling_type()
                self.gguf_writer.add_causal_attention(False)
''', 1)
assert "Ministral3Model\",\n    \"Mistral3Model\"" in s and "add_add_bos_token(False)" in s and "add_causal_attention(False)" in s, "patch failed (llama.cpp の版が違う?)"
p.write_text(s)
p = W / "conversion/__init__.py"; s = p.read_text()
s = s.replace('''    "Mistral3ForConditionalGeneration": "mistral3",''', '''    "Mistral3ForConditionalGeneration": "mistral3",
    "Ministral3Model": "mistral3",
    "Mistral3Model": "mistral3",''', 1)
p.write_text(s)
PYEOF
SRC="$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$REPO'))")"
mkdir -p "$OUT"; F="$OUT/nemotron-3.5-embed-1b-$TYPE.gguf"
PYTHONPATH="$L/gguf-py:$W" "$PY" "$W/convert_hf_to_gguf.py" "$SRC" --outfile "$F" --outtype "$TYPE"
sha256sum "$F"
