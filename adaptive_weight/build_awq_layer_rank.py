#!/usr/bin/env python3
"""Offline AWQ layer rank for demote order.

Low W4 reconstruction error (vs BF16) ⇒ demote first; also records per-layer
W8→W4 MiB savings for occupancy → K.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
PROJ_RE = re.compile(
    r"layers\.\d+\.(self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)\.weight_packed$"
)


def layer_id(name: str) -> int | None:
    m = LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def list_weight_map(model_dir: Path) -> dict[str, Path]:
    idx = model_dir / "model.safetensors.index.json"
    if idx.exists():
        wm = json.loads(idx.read_text())["weight_map"]
        return {k: model_dir / v for k, v in wm.items()}
    shards = sorted(model_dir.glob("*.safetensors"))
    if len(shards) != 1:
        raise FileNotFoundError(f"need index or single shard in {model_dir}")
    from safetensors import safe_open

    with safe_open(str(shards[0]), framework="pt", device="cpu") as f:
        return {k: shards[0] for k in f.keys()}


def load_tensor(path: Path, name: str):
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def iter_shard_groups(name_to_shard: dict[str, Path], names: list[str]) -> Iterator[tuple[Path, list[str]]]:
    by: dict[Path, list[str]] = {}
    for n in names:
        by.setdefault(name_to_shard[n], []).append(n)
    yield from by.items()


def nbytes_by_layer(name_to_shard: dict[str, Path]) -> dict[int, int]:
    """Sum safetensor nbytes for tensors belonging to each decoder layer."""
    from safetensors import safe_open

    out: dict[int, int] = defaultdict(int)
    by_shard: dict[Path, list[str]] = defaultdict(list)
    for name, shard in name_to_shard.items():
        lid = layer_id(name)
        if lid is None:
            continue
        by_shard[shard].append(name)
    for shard, names in by_shard.items():
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for n in names:
                t = f.get_tensor(n)
                out[layer_id(n)] += int(t.nbytes)  # type: ignore[index]
                del t
    return dict(out)


def dequant_w4_module(
    *,
    packed,
    scale,
    zp,
    shape_t,
    num_bits: int = 4,
):
    import torch
    from compressed_tensors.compressors.pack_quantized import unpack_from_int32
    from compressed_tensors.quantization import dequantize

    orig = torch.Size([int(x) for x in shape_t.tolist()])
    zp_shape = (*orig[:-1], scale.shape[-1])
    zpu = unpack_from_int32(zp, num_bits, zp_shape, packed_dim=0)
    q = unpack_from_int32(packed, num_bits, orig, packed_dim=1)
    return dequantize(x_q=q, scale=scale, zero_point=zpu)


def score_layers(
    w4_dir: Path,
    bf16_dir: Path | None,
    *,
    device: str = "cpu",
) -> tuple[list[dict[str, Any]], str]:
    """Return per-layer rows and primary metric name."""
    import torch

    w4_map = list_weight_map(w4_dir)
    packed_names = sorted(n for n in w4_map if PROJ_RE.search(n))
    if not packed_names:
        raise RuntimeError(f"no packed Linear weights in {w4_dir}")

    bf_map = list_weight_map(bf16_dir) if bf16_dir is not None else None
    use_recon = bf_map is not None

    sse: dict[int, float] = defaultdict(float)
    n_elem: dict[int, int] = defaultdict(int)
    scale_abs_sum: dict[int, float] = defaultdict(float)
    scale_n: dict[int, int] = defaultdict(int)
    modules: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for packed_name in packed_names:
        lid = layer_id(packed_name)
        assert lid is not None
        base = packed_name[: -len(".weight_packed")]
        scale_name = f"{base}.weight_scale"
        zp_name = f"{base}.weight_zero_point"
        shape_name = f"{base}.weight_shape"
        bf_name = f"{base}.weight"

        packed = load_tensor(w4_map[packed_name], packed_name)
        scale = load_tensor(w4_map[scale_name], scale_name)
        zp = load_tensor(w4_map[zp_name], zp_name)
        shape_t = load_tensor(w4_map[shape_name], shape_name)

        scale_f = scale.float()
        scale_abs_sum[lid] += float(scale_f.abs().sum().item())
        scale_n[lid] += int(scale_f.numel())

        mod_row: dict[str, Any] = {
            "name": base.split(".", 2)[-1] if base.startswith("model.") else base,
            "mean_abs_scale": float(scale_f.abs().mean().item()),
        }

        if use_recon:
            assert bf_map is not None
            if bf_name not in bf_map:
                raise KeyError(f"missing BF16 weight {bf_name}")
            deq = dequant_w4_module(
                packed=packed, scale=scale, zp=zp, shape_t=shape_t, num_bits=4
            )
            if device != "cpu":
                deq = deq.to(device)
            w = load_tensor(bf_map[bf_name], bf_name).float()
            if device != "cpu":
                w = w.to(device)
            err = deq.float() - w
            sse_i = float((err * err).sum().item())
            n_i = int(err.numel())
            mse_i = sse_i / max(n_i, 1)
            sse[lid] += sse_i
            n_elem[lid] += n_i
            mod_row["mse"] = mse_i
            mod_row["rmse"] = mse_i**0.5
            del deq, w, err

        modules[lid].append(mod_row)
        del packed, scale, zp, shape_t

    layers = sorted(set(scale_abs_sum) | set(sse) | set(modules))
    rows: list[dict[str, Any]] = []
    for lid in layers:
        mean_abs_scale = scale_abs_sum[lid] / max(scale_n[lid], 1)
        row: dict[str, Any] = {
            "layer": lid,
            "mean_abs_scale": mean_abs_scale,
            "modules": modules[lid],
        }
        if use_recon:
            mse = sse[lid] / max(n_elem[lid], 1)
            row["mse"] = mse
            row["rmse"] = mse**0.5
            row["n_elem"] = n_elem[lid]
            row["score"] = mse
        else:
            row["score"] = mean_abs_scale
        rows.append(row)

    metric = "w4_recon_mse_vs_bf16" if use_recon else "w4_mean_abs_scale"
    return rows, metric


def build_rank(
    w4_dir: Path,
    w8_dir: Path,
    bf16_dir: Path | None,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    t0 = time.time()
    rows, metric = score_layers(w4_dir, bf16_dir, device=device)
    rows_sorted = sorted(rows, key=lambda r: (float(r["score"]), int(r["layer"])))
    demote_order = [int(r["layer"]) for r in rows_sorted]

    w8_bytes = nbytes_by_layer(list_weight_map(w8_dir))
    w4_bytes = nbytes_by_layer(list_weight_map(w4_dir))
    all_layers = sorted(set(w8_bytes) | set(w4_bytes) | {r["layer"] for r in rows})
    save_mib: dict[str, float] = {}
    for lid in all_layers:
        delta = max(0, w8_bytes.get(lid, 0) - w4_bytes.get(lid, 0))
        save_mib[str(lid)] = delta / (1024**2)

    saves = [save_mib[str(lid)] for lid in all_layers if str(lid) in save_mib]
    mean_save = sum(saves) / len(saves) if saves else 0.0

    by_layer = {int(r["layer"]): r for r in rows}
    return {
        "model": {
            "w4_dir": str(w4_dir),
            "w8_dir": str(w8_dir),
            "bf16_dir": str(bf16_dir) if bf16_dir else None,
        },
        "metric": metric,
        "rank_rule": "ascending score = demote first (lowest recon err / scale)",
        "n_layers": len(demote_order),
        "demote_order": demote_order,
        "save_per_layer_mib": save_mib,
        "save_per_layer_mib_mean": mean_save,
        "layers": [by_layer[lid] for lid in sorted(by_layer)],
        "elapsed_s": time.time() - t0,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--w4-dir",
        type=Path,
        default=Path("/workspace/llm_inference/quantized/Qwen3-8B-W4A16-AWQ"),
    )
    p.add_argument(
        "--w8-dir",
        type=Path,
        default=Path("/workspace/llm_inference/quantized_local/Qwen3-8B-W8A16-AWQ"),
    )
    p.add_argument(
        "--bf16-dir",
        type=Path,
        default=Path("/workspace/llm_inference/quantized/Qwen3-8B-BF16"),
    )
    p.add_argument(
        "--no-bf16",
        action="store_true",
        help="Skip recon; rank by mean |weight_scale| only",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("adaptive_weight/results/layer_rank.json"),
    )
    args = p.parse_args()

    bf = None if args.no_bf16 else args.bf16_dir
    if bf is not None and not bf.exists():
        raise FileNotFoundError(f"BF16 dir missing: {bf} (pass --no-bf16 for scale-only)")

    print(
        f"[rank] w4={args.w4_dir} w8={args.w8_dir} bf16={bf} device={args.device}",
        flush=True,
    )
    payload = build_rank(args.w4_dir, args.w8_dir, bf, device=args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"[rank] metric={payload['metric']} n={payload['n_layers']} "
        f"mean_save={payload['save_per_layer_mib_mean']:.1f} MiB/layer "
        f"elapsed={payload['elapsed_s']:.1f}s",
        flush=True,
    )
    print(f"[rank] demote_order[:12]={payload['demote_order'][:12]}", flush=True)
    print(f"[wrote] {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
