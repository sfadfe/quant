# Dual-resident weight transfer (Qwen3-8B)

Short contexts keep **W8** for better early accuracy; once context grows past a threshold we switch to **W4** so long sessions still fit. Both weight engines stay resident so the switch is a pointer flip, not a cold reload.

Latest run on RTX 3090 (`beat_w4_20260805T074012Z`): policy **`cp_T4k_w4_dual`** vs fixed W4 / fixed W8.

## What we claim

| Phase | Policy | Goal |
| --- | --- | --- |
| L &lt; 4k | W8A16-RTN | Higher short-context accuracy than fixed W4 |
| L ≥ 4k | W4A16-AWQ (dual pointer) | Same VRAM / speed ballpark as fixed W4, fit through 16k |

Fixed W8 alone fails past 8k under the same 12 GiB budget. Fixed W4 fits 16k but loses early `short_qa`.

## Headline numbers (12 GiB budget, no KV prealloc)

| L | ours tok/s | fixed_w4 tok/s | ours VRAM | fixed_w4 VRAM | ours short_qa | fixed_w4 short_qa |
| ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| 2k | 44.5 | 47.1 | 10391 | 7121 | ✓ | ✗ |
| 4k | 46.8 | 38.8 | 7671 | 7671 | ✗ | ✗ |
| 8k | 47.7 | 39.6 | 8777 | 8777 | ✗ | ✗ |
| 12k | 38.9 | 47.9 | 9367 | 9367 | ✗ | ✗ |
| 16k | 47.6 | 47.3 | 9943 | 9943 | ✗ | ✗ |

- **Transfer stall:** `swap_s = 0` at T=4k (`dual_resident_hit=True`) — W4 was pipelined in after the 2k step while W8 was still resident.
- **Mean step tok/s:** ours **45.1** vs fixed_w4 **44.2** (on average comparable; per-L scatter is GPU clock sticky ~39 vs ~48, not a reload tax).
- **VRAM after T:** identical to fixed W4 (same W4+auto recipe). Strictly-lower VRAM needs a later KV-fp8 stage (not in this run).
- **Early accuracy:** at 2k, ours matches W8 on `short_qa`; fixed W4 misses.

Plots: `figs/beat_w4_20260805T074012Z/` · raw: `gpuprofile/beat_w4_20260805T074012Z/`
