#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <aime24|aime25|hmmt_feb_2025>" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASET="$1"
DATA_FILE="${DATA_FILE:-$ROOT_DIR/data/$DATASET.jsonl}"
PORT="${PORT:-8000}"
MODEL_NAME="${MODEL_NAME:-Qwen3-4B}"
MODEL_SLUG="${MODEL_NAME,,}"
BUDGET="${BUDGET:-80}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
TEMPERATURE="${TEMPERATURE:-0.6}"
SUMMARY_FILE="${SUMMARY_FILE:-$ROOT_DIR/results/$DATASET-$MODEL_SLUG.json}"
LOG_FILE="${LOG_FILE:-$ROOT_DIR/logs/$DATASET-$MODEL_SLUG-eval.log}"
PID_FILE="${PID_FILE:-$ROOT_DIR/logs/$DATASET-$MODEL_SLUG-eval.pid}"

if [[ ! -f "$DATA_FILE" ]]; then
  echo "Dataset not found: $DATA_FILE" >&2
  exit 1
fi
if [[ ! -x "$ROOT_DIR/.venv/bin/python" ]]; then
  echo "Python environment not found: $ROOT_DIR/.venv" >&2
  exit 1
fi

mkdir -p "$ROOT_DIR/results" "$ROOT_DIR/logs"
if ! curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "vLLM is not ready at http://127.0.0.1:${PORT}. Start it first with scripts/start_vllm_qwen3_4b.sh." >&2
  exit 1
fi

setsid nohup "$ROOT_DIR/.venv/bin/python" -m src.evaluation.evaluate \
  --server-url "http://127.0.0.1:${PORT}/v1" \
  --model "$MODEL_NAME" \
  --data "$DATA_FILE" \
  --budget "$BUDGET" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  >"$SUMMARY_FILE" 2>"$LOG_FILE" < /dev/null &

echo $! > "$PID_FILE"
TREE_FILE="$ROOT_DIR/results/$DATASET-$MODEL_NAME-tree.jsonl"
echo "Started $DATASET evaluation (PID $(<"$PID_FILE"))."
echo "Summary: $SUMMARY_FILE"
echo "Log: $LOG_FILE"
echo "Tree: $TREE_FILE"
