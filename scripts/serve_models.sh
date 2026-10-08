#!/usr/bin/env bash
# ragnarok が使うモデルサーバー (vLLM) を手で立てる版。普段は `ragnarok up` (ragnarok/profiles.py) で足りる。
# 例: JUDGE_GPU=6 FAST_GPU=5 scripts/serve_models.sh
VENV=${VLLM_VENV:-/home/toriumi/dev/typesafe_clone/decision-model/.venv}
export PATH=$VENV/bin:$PATH HF_HUB_CACHE=${HF_HUB_CACHE:-/data/openvons/choice_spec/hf_cache}
CUDA_VISIBLE_DEVICES=${JUDGE_GPU:-6} nohup $VENV/bin/vllm serve ${JUDGE_MODEL:-cyankiwi/Qwen3.8-27B-AWQ-INT4} --served-model-name qwen27b --port 8310 --max-model-len 32768 --gpu-memory-utilization ${JUDGE_MEM:-0.5} --max-num-seqs 16 --max-logprobs 40 --default-chat-template-kwargs '{"enable_thinking": false}' > /tmp/ragnarok_27b.log 2>&1 &
CUDA_VISIBLE_DEVICES=${FAST_GPU:-5} nohup $VENV/bin/vllm serve ${FAST_MODEL:-Qwen/Qwen3-4B} --served-model-name qwen3-4b --port 8311 --max-model-len 32768 --gpu-memory-utilization ${FAST_MEM:-0.25} --max-num-seqs 64 --max-logprobs 30 > /tmp/ragnarok_4b.log 2>&1 &
echo "starting 27B (:8310) and 4B (:8311)"
