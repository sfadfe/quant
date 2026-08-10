#!/usr/bin/env bash
# Adaptive-Weight L-sweep → toks/vram/short_qa/swap scatters.
# L = 2048…24576 step 2048. run_id = UTC stamp only.
#
#   ./adaptive_weight/run_beat_bench.sh
#   ./adaptive_weight/run_beat_bench.sh --out-dir results/$(date -u +%Y%m%dT%H%M%SZ)
#   ./adaptive_weight/run_beat_bench.sh --plot-only --run-dir results/<stamp>
set -euo pipefail

PKG="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$PKG/.." && pwd)"
cd "$ROOT"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR=""
RUN_DIR=""
PLOT_ONLY=0
BUDGET_GIB=12
CHECKPOINTS=(2048 4096 6144 8192 10240 12288 14336 16384 18432 20480 22528 24576)
BASELINE="fixed_w4"
OURS="hf_mixed_adaptive"
TRANSFER_T=4096
LAYER_RANK="$PKG/results/layer_rank.json"
WAVES=2
EXTRA_BENCH_ARGS=()

run_id_from_dir() {
  local base
  base="$(basename "$1")"
  if [[ "$base" =~ ([0-9]{8}T[0-9]{6}Z)$ ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo "$base"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --plot-only) PLOT_ONLY=1; shift ;;
    --budget-gib) BUDGET_GIB="$2"; shift 2 ;;
    --baseline) BASELINE="$2"; shift 2 ;;
    --ours) OURS="$2"; shift 2 ;;
    --transfer-t) TRANSFER_T="$2"; shift 2 ;;
    --waves) WAVES="$2"; shift 2 ;;
    --layer-rank) LAYER_RANK="$2"; shift 2 ;;
    --hf-mixed) shift ;;
    --) shift; EXTRA_BENCH_ARGS+=("$@"); break ;;
    *) EXTRA_BENCH_ARGS+=("$1"); shift ;;
  esac
done

if [[ "$PLOT_ONLY" == "1" ]]; then
  if [[ -z "$RUN_DIR" ]]; then
    echo "need --run-dir with --plot-only" >&2
    exit 2
  fi
  if [[ -f "$RUN_DIR/meta.json" ]]; then
    META_OURS="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('ours') or '')" "$RUN_DIR/meta.json" 2>/dev/null || true)"
    META_BASE="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('baseline') or '')" "$RUN_DIR/meta.json" 2>/dev/null || true)"
    if [[ -n "$META_OURS" ]]; then OURS="$META_OURS"; fi
    if [[ -n "$META_BASE" ]]; then BASELINE="$META_BASE"; fi
  fi
else
  OUT_DIR="${OUT_DIR:-$ROOT/results/${STAMP}}"
  RUN_DIR="$OUT_DIR"
  mkdir -p "$OUT_DIR"
  echo "[beat] L-sweep checkpoints: ${CHECKPOINTS[*]}"
  echo "[beat] metrics: tok/s, peak VRAM, short_qa (+ swap_s)"
  echo "[beat] ours=$OURS out: $OUT_DIR"

  HF_ARGS=(
    --mode lsweep
    --checkpoints "${CHECKPOINTS[@]}"
    --budget-gib "$BUDGET_GIB"
    --out-dir "$OUT_DIR"
    --waves "$WAVES"
    --transfer-t "$TRANSFER_T"
    --layer-rank "$LAYER_RANK"
  )

  LOCK="$PKG/run_locked_clocks.sh"
  if [[ -x "$LOCK" ]]; then
    "$LOCK" -- python3 -u "$PKG/hf_mixed_demote.py" \
      "${HF_ARGS[@]}" \
      "${EXTRA_BENCH_ARGS[@]}"
  else
    python3 -u "$PKG/hf_mixed_demote.py" \
      "${HF_ARGS[@]}" \
      "${EXTRA_BENCH_ARGS[@]}"
  fi
fi

RUN_ID="$(run_id_from_dir "$RUN_DIR")"
FIG_DIR="${FIG_DIR:-$ROOT/figs/${RUN_ID}}"
mkdir -p "$FIG_DIR"

echo "[plot] scatters → $FIG_DIR (run_id=$RUN_ID) ours=$OURS baseline=$BASELINE"
python3 -u "$PKG/plot_beat_scatter.py" \
  --run-dir "$RUN_DIR" \
  --out-dir "$FIG_DIR" \
  --baseline "$BASELINE" \
  --ours "$OURS" \
  --transfer-t 0 \
  --baseline-label "AWQ 4bit" \
  --ours-label "Adaptive" \
  --w8 fixed_w8 \
  --w8-label "AWQ 8bit"

echo "[done] raw=$RUN_DIR figs=$FIG_DIR"
ls -la "$FIG_DIR"/*.png 2>/dev/null || true
