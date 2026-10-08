#!/usr/bin/env bash
# gguf profile 用の llama.cpp サーバーを手で立てる版 (普段は `ragnarok up gguf`)。 判定 Qwen3.5-4B Q3_K_M (:8320) + 回答 Qwen3-4B Q4_K_M (:8321)。
# 重み計 4.8GB、8k ctx ×1 slot で VRAM 計 7.4GB。判定だけ (nollm 以上 light 未満) なら 8320 だけで 3.2GB。
# 例: GGUF_DIR=/data/ragnarok/gguf GPU=0 scripts/serve_gguf.sh
#   GGUF: https://huggingface.co/unsloth/Qwen3.5-4B-GGUF (Qwen3.5-4B-Q3_K_M.gguf), https://huggingface.co/unsloth/Qwen3-4B-GGUF (Qwen3-4B-Q4_K_M.gguf)
LS=${LLAMA_SERVER:-llama-server}; D=${GGUF_DIR:-/data/ragnarok/gguf}; CTX=${CTX:-8192}; PAR=${PARALLEL:-1}   # -c は slot 数で割られる (8 並列なら -c 65536 以上)
CUDA_VISIBLE_DEVICES=${GPU:-0} nohup $LS -m $D/Qwen3.5-4B-Q3_K_M.gguf --port 8320 -ngl 99 -c $CTX --parallel $PAR -b 2048 --alias q354-q3 > /tmp/ragnarok_gguf_judge.log 2>&1 &
CUDA_VISIBLE_DEVICES=${GPU:-0} nohup $LS -m $D/Qwen3-4B-Q4_K_M.gguf --port 8321 -ngl 99 -c $CTX --parallel $PAR -b 2048 --alias q34-q4 > /tmp/ragnarok_gguf_answer.log 2>&1 &
echo "starting judge Qwen3.5-4B Q3 (:8320) and answerer Qwen3-4B Q4 (:8321)"
