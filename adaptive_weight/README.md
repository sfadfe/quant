# adaptive_weight

HF **Adaptive-Weight** L-sweep (progressive packed W8→W4 demote + Marlin GEMM) vs **AWQ int4**.

Published plots + numbers: repo root [`README.md`](../README.md) (`docs/figs/20260806T052437Z/`).

## L-sweep

Checkpoints: **2048…24576** step 2048 (plot ticks: `2k`, `4k`, …)  
Metrics: **tok/s**, **peak VRAM**, **short_qa**, **swap_s**  
Orchestration: **`run_beat_bench.sh` only** (edit in place — no one-off wrappers).

```bash
./adaptive_weight/run_beat_bench.sh \
  --out-dir results/$(date -u +%Y%m%dT%H%M%SZ)

# plot + pull only
./adaptive_weight/run_beat_bench.sh --plot-only --run-dir results/<stamp>
```

`run_id` = UTC stamp only (e.g. `20260806T052437Z`) under `results/`, `gpuprofile/`, `figs/`.

Soft hold defaults (tunable flags on `hf_mixed_demote.py`):

- `hold_headroom_gib=0.50` → `target ≈ W8_alloc + 0.5`
- `kv_mib_per_tok=0.23` (projected pressure; set `0` if you pre-allocate KV)
- `occ_k_min/max=2/6`, KV quant off (`--kv-quant-t 0`)

## Key scripts

| Script | Role |
| --- | --- |
| `hf_mixed_demote.py` | Adaptive L-sweep / smoke / `--mode ifstruct` |
| `occupancy_ctrl.py` | Soft progressive demote planner |
| `inplace_w_replace.py` | Layerwise W8→W4 weight morph |
| `plot_beat_scatter.py` | `toks` / `vram` / `short_qa` / `swap` scatters |
| `build_awq_layer_rank.py` | Offline demote order → `results/layer_rank.json` |
| `quantize_w8_awq.py` | Build W8A16-AWQ checkpoint |
| `run_beat_bench.sh` | Locked clocks → bench → plot |

## Layer rank

```bash
python3 -u adaptive_weight/build_awq_layer_rank.py \
  --w4-dir quantized/Qwen3-8B-W4A16-AWQ \
  --w8-dir quantized_local/Qwen3-8B-W8A16-AWQ \
  --bf16-dir quantized/Qwen3-8B-BF16 \
  --out adaptive_weight/results/layer_rank.json
```

## IFStruct accuracy

```bash
python3 -u adaptive_weight/hf_mixed_demote.py --mode ifstruct \
  --out-dir ifstruct_results/<stamp> \
  --dataset quantized/ifstruct_sample_100.jsonl \
  --ifstruct-policies "Adaptive-Weight,AWQ int4"
```

Uses `qwen_ifstruct_eval` validators; metric = `pass_rate`.
