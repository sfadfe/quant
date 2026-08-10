#!/usr/bin/env bash
# Qwen3-8B W8A16-AWQ on one 3090 (CPU offload + sequential targets; W4 used 2-GPU).
set -euo pipefail

ROOT=/workspace/llm_inference
OUT="$ROOT/quantized_local/Qwen3-8B-W8A16-AWQ"
VENV="$ROOT/.venv-llmcompressor"
LOG_DIR="$ROOT/adaptive_weight/results/w8_awq_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$LOG_DIR" "$(dirname "$OUT")"

export PATH=/opt/conda/bin:$PATH
export HF_HOME="${HF_HOME:-$ROOT/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$HUGGINGFACE_HUB_CACHE"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[setup] creating $VENV"
  python -m venv "$VENV"
  "$VENV/bin/pip" install -U pip setuptools wheel
  "$VENV/bin/pip" install \
    "torch==2.10.0" --index-url https://download.pytorch.org/whl/cu128
  "$VENV/bin/pip" install \
    "llmcompressor==0.12.0" \
    "transformers>=5.9.0" \
    "datasets" \
    "accelerate" \
    "sentencepiece" \
    "protobuf"
fi

mkdir -p "$OUT"

echo "[run] log=$LOG_DIR"
echo $$ > "$LOG_DIR/run.pid"
"$VENV/bin/python" -u "$ROOT/adaptive_weight/quantize_w8_awq.py" \
  --model "$ROOT/quantized/Qwen3-8B-BF16" \
  --out-dir "$OUT" \
  --scheme W8A16 \
  --num-samples 128 \
  --max-seq-length 512 \
  --max-memory-gib 10 \
  --n-grid 10 \
  --sequential-targets Qwen3DecoderLayer \
  --awq-offload cpu \
  2>&1 | tee "$LOG_DIR/run.log"

echo "$LOG_DIR" > "$LOG_DIR/DONE"
echo "[done] $OUT"
