#!/usr/bin/env bash
# Pin GPU SM clocks for reproducible decode tok/s, then restore on EXIT.
# 3090 boost swings (~39 vs ~48 tok/s) swamp the effect under test.
#
#   ./run_locked_clocks.sh [--gpu N] [--sm-clock MHZ] -- <command...>
set -uo pipefail

GPU_ID=0
SM_CLOCK=""
LOCKED=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU_ID="$2"; shift 2 ;;
    --sm-clock) SM_CLOCK="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "usage: $0 [--gpu N] [--sm-clock MHZ] -- <command...>" >&2
  exit 2
fi

restore_clocks() {
  if [[ "$LOCKED" == "1" ]]; then
    echo "[clocks] resetting GPU $GPU_ID to default clocks"
    nvidia-smi -i "$GPU_ID" -rgc >/dev/null 2>&1 \
      || echo "[clocks] WARN: reset failed; run 'nvidia-smi -i $GPU_ID -rgc' manually" >&2
  fi
}
trap restore_clocks EXIT INT TERM

if [[ -z "$SM_CLOCK" ]]; then
  # Prefer sustained-load clock over boost ceiling (thermal drop mid-run).
  MAX_SM=$(nvidia-smi -i "$GPU_ID" --query-gpu=clocks.max.sm --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  if [[ "$MAX_SM" =~ ^[0-9]+$ ]]; then
    SM_CLOCK=$(( MAX_SM * 85 / 100 ))
  fi
fi

if [[ -n "$SM_CLOCK" ]]; then
  if nvidia-smi -i "$GPU_ID" -lgc "$SM_CLOCK" >/dev/null 2>&1; then
    LOCKED=1
    echo "[clocks] locked GPU $GPU_ID SM clock to ${SM_CLOCK} MHz"
  else
    echo "[clocks] WARN: could not lock clocks (needs root?); running unlocked" >&2
  fi
else
  echo "[clocks] WARN: could not read max SM clock; running unlocked" >&2
fi

nvidia-smi -i "$GPU_ID" --query-gpu=name,clocks.sm,clocks.max.sm,persistence_mode \
  --format=csv,noheader 2>/dev/null || true

"$@"
rc=$?
echo "[clocks] command exited rc=$rc"
exit "$rc"
