#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU_ID="${GPU_ID:-5}"
MODEL_DIR="${MODEL_DIR:-/home/u2026104455/Models/Qwen3-4B}"
PORT="${PORT:-8000}"
LOG_FILE="${LOG_FILE:-$ROOT_DIR/logs/qwen3-4b-vllm.log}"
PID_FILE="$ROOT_DIR/logs/qwen3-4b-vllm.pid"

mkdir -p "$ROOT_DIR/logs"
if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "vLLM is already serving on port ${PORT}."
  exit 0
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" setsid nohup "$ROOT_DIR/.venv/bin/python" -u -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name Qwen3-4B \
  --host 127.0.0.1 \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --enable-reasoning \
  --reasoning-parser deepseek_r1 \
  --trust-remote-code \
  >"$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "Started vLLM PID $(cat "$PID_FILE") on physical GPU ${GPU_ID}."
echo "Log: $LOG_FILE"
