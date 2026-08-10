#!/usr/bin/env python3
"""Layerwise W8→W4 weight replace (layout audit, GPU peak, xfer bench).

W8/W4 packed widths differ and W4 adds weight_zero_point, so same-slot copy_
is impossible. Peak path: free layer W8 → alloc/H2D W4 (not dual-resident).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


@dataclass(frozen=True)
class TensorMeta:
    name: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int


@dataclass
class LayoutReport:
    w8_dir: str
    w4_dir: str
    w8_bits: int | None
    w4_bits: int | None
    w8_symmetric: bool | None
    w4_symmetric: bool | None
    w8_has_zp: bool
    w4_has_zp: bool
    same_slot_copy_ok: bool
    blockers: list[str] = field(default_factory=list)
    examples: dict[str, Any] = field(default_factory=dict)
    w8_bytes: int = 0
    w4_bytes: int = 0
    n_layers: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _quant_weights(config_path: Path) -> dict[str, Any]:
    cfg = json.loads(config_path.read_text())
    qc = cfg.get("quantization_config") or {}
    groups = qc.get("config_groups") or {}
    g0 = groups.get("group_0") or next(iter(groups.values()), {})
    return dict(g0.get("weights") or {}), qc


def iter_safetensor_meta(model_dir: Path) -> Iterator[TensorMeta]:
    """Yield tensor metas. Loads each tensor once (CPU); OK for one-shot audit."""
    from safetensors import safe_open

    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors in {model_dir}")
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                yield TensorMeta(
                    name=key,
                    shape=tuple(t.shape),
                    dtype=str(t.dtype).replace("torch.", ""),
                    nbytes=int(t.nbytes),
                )
                del t


def checkpoint_nbytes(model_dir: Path) -> int:
    return sum(p.stat().st_size for p in model_dir.glob("*.safetensors"))


def load_named_shapes(model_dir: Path, names: list[str]) -> dict[str, TensorMeta]:
    from safetensors import safe_open

    idx_file = model_dir / "model.safetensors.index.json"
    name_to_shard: dict[str, Path] = {}
    if idx_file.exists():
        wm = json.loads(idx_file.read_text())["weight_map"]
        name_to_shard = {n: model_dir / s for n, s in wm.items()}
    else:
        shards = sorted(model_dir.glob("*.safetensors"))
        if len(shards) != 1:
            raise FileNotFoundError(f"need index or single shard in {model_dir}")
        with safe_open(str(shards[0]), framework="pt", device="cpu") as f:
            name_to_shard = {k: shards[0] for k in f.keys()}

    out: dict[str, TensorMeta] = {}
    by_shard: dict[Path, list[str]] = {}
    for n in names:
        if n not in name_to_shard:
            continue
        by_shard.setdefault(name_to_shard[n], []).append(n)
    for shard, ns in by_shard.items():
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for n in ns:
                t = f.get_tensor(n)
                out[n] = TensorMeta(n, tuple(t.shape), str(t.dtype).replace("torch.", ""), int(t.nbytes))
                del t
    return out


def list_tensor_names(model_dir: Path) -> list[str]:
    idx_file = model_dir / "model.safetensors.index.json"
    if idx_file.exists():
        return sorted(json.loads(idx_file.read_text())["weight_map"].keys())
    from safetensors import safe_open

    shards = sorted(model_dir.glob("*.safetensors"))
    if len(shards) != 1:
        raise FileNotFoundError(f"need index or single shard in {model_dir}")
    with safe_open(str(shards[0]), framework="pt", device="cpu") as f:
        return sorted(f.keys())


def layer_id(name: str) -> int | None:
    m = LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def group_by_layer(metas: list[TensorMeta]) -> dict[int | None, list[TensorMeta]]:
    out: dict[int | None, list[TensorMeta]] = {}
    for m in metas:
        out.setdefault(layer_id(m.name), []).append(m)
    return out


def audit_layouts(w8_dir: Path, w4_dir: Path) -> LayoutReport:
    """Prove same-slot copy is illegal; summarize sizes for peak math."""
    w8_w, _w8_qc = _quant_weights(w8_dir / "config.json")
    w4_w, _w4_qc = _quant_weights(w4_dir / "config.json")

    w8_names = list_tensor_names(w8_dir)
    w4_names = list_tensor_names(w4_dir)
    w8_set, w4_set = set(w8_names), set(w4_names)

    probes = [
        "model.layers.0.self_attn.q_proj.weight_packed",
        "model.layers.0.self_attn.q_proj.weight_scale",
        "model.layers.0.self_attn.q_proj.weight_zero_point",
        "model.layers.0.mlp.gate_proj.weight_packed",
    ]
    shapes = {
        "w8": load_named_shapes(w8_dir, probes),
        "w4": load_named_shapes(w4_dir, probes),
    }

    blockers: list[str] = []
    examples: dict[str, Any] = {}
    for name in probes:
        if name in shapes["w8"] and name in shapes["w4"]:
            a, b = shapes["w8"][name], shapes["w4"][name]
            examples[name] = {
                "w8": {"shape": a.shape, "dtype": a.dtype},
                "w4": {"shape": b.shape, "dtype": b.dtype},
            }
            if a.shape != b.shape:
                blockers.append(f"{name} shape {a.shape} != {b.shape}")

    w8_zp = any(k.endswith("weight_zero_point") for k in w8_set)
    w4_zp = any(k.endswith("weight_zero_point") for k in w4_set)
    if w4_zp and not w8_zp:
        blockers.append("W4 has weight_zero_point; W8 does not — scheme/metadata diverge")
    if w8_w.get("num_bits") != w4_w.get("num_bits"):
        blockers.append(f"num_bits W8={w8_w.get('num_bits')} W4={w4_w.get('num_bits')}")

    packed_shared = [n for n in w8_set & w4_set if n.endswith("weight_packed")]
    sample = [n for n in packed_shared if ".layers.15." in n or ".layers.0." in n][:6]
    sample_shapes_w8 = load_named_shapes(w8_dir, sample)
    sample_shapes_w4 = load_named_shapes(w4_dir, sample)
    mismatch = 0
    for n in sample:
        if n in sample_shapes_w8 and n in sample_shapes_w4:
            if sample_shapes_w8[n].shape != sample_shapes_w4[n].shape:
                mismatch += 1
    if mismatch:
        blockers.append(f"{mismatch}/{len(sample)} sampled packed tensors differ in shape")

    n_layers = len({layer_id(n) for n in w8_names if layer_id(n) is not None})

    return LayoutReport(
        w8_dir=str(w8_dir),
        w4_dir=str(w4_dir),
        w8_bits=w8_w.get("num_bits"),
        w4_bits=w4_w.get("num_bits"),
        w8_symmetric=w8_w.get("symmetric"),
        w4_symmetric=w4_w.get("symmetric"),
        w8_has_zp=w8_zp,
        w4_has_zp=w4_zp,
        same_slot_copy_ok=len(blockers) == 0,
        blockers=blockers,
        examples=examples,
        w8_bytes=checkpoint_nbytes(w8_dir),
        w4_bytes=checkpoint_nbytes(w4_dir),
        n_layers=n_layers,
    )


class LayerwiseWeightReplacer:
    """GPU-resident W8 → free/alloc W4 per layer; track peak (no forward)."""

    def __init__(self, w8_dir: Path, w4_dir: Path, device: str = "cuda:0"):
        self.w8_dir = Path(w8_dir)
        self.w4_dir = Path(w4_dir)
        self.device = device
        self._w8_index = self._build_index(self.w8_dir)
        self._w4_index = self._build_index(self.w4_dir)

    @staticmethod
    def _build_index(model_dir: Path) -> dict[str, tuple[Path, str]]:
        """name -> (shard_path, name)."""
        idx_file = model_dir / "model.safetensors.index.json"
        out: dict[str, tuple[Path, str]] = {}
        if idx_file.exists():
            weight_map = json.loads(idx_file.read_text())["weight_map"]
            for name, shard in weight_map.items():
                out[name] = (model_dir / shard, name)
            return out
        shards = sorted(model_dir.glob("*.safetensors"))
        if len(shards) != 1:
            raise FileNotFoundError(f"need index.json or one shard in {model_dir}")
        from safetensors import safe_open

        with safe_open(str(shards[0]), framework="pt", device="cpu") as f:
            for name in f.keys():
                out[name] = (shards[0], name)
        return out

    def _load_names(self, index: dict[str, tuple[Path, str]], names: list[str], device: str):
        import torch
        from safetensors import safe_open

        by_shard: dict[Path, list[str]] = {}
        for n in names:
            shard, _ = index[n]
            by_shard.setdefault(shard, []).append(n)
        tensors: dict[str, Any] = {}
        for shard, ns in by_shard.items():
            with safe_open(str(shard), framework="pt", device="cpu") as f:
                for n in ns:
                    tensors[n] = f.get_tensor(n).to(device, non_blocking=True)
        return tensors

    def run(self, *, empty_cache_each_layer: bool = True) -> dict[str, Any]:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for peak probe")

        torch.cuda.set_device(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        w8_names = sorted(self._w8_index.keys())
        w4_names = sorted(self._w4_index.keys())
        w8_layers = group_by_layer(
            [TensorMeta(n, (), "", 0) for n in w8_names]
        )
        layer_ids = sorted(i for i in w8_layers if i is not None)

        shared = [n for n in w8_names if layer_id(n) is None]

        resident: dict[str, Any] = {}
        resident.update(self._load_names(self._w8_index, w8_names, self.device))
        torch.cuda.synchronize()
        after_w8 = torch.cuda.max_memory_allocated()
        allocated_w8 = torch.cuda.memory_allocated()

        per_layer: list[dict[str, Any]] = []
        for i in layer_ids:
            drop = [n for n in list(resident) if layer_id(n) == i]
            for n in drop:
                del resident[n]
            if empty_cache_each_layer:
                torch.cuda.empty_cache()

            add = [n for n in w4_names if layer_id(n) == i]
            new_t = self._load_names(self._w4_index, add, self.device)
            resident.update(new_t)
            torch.cuda.synchronize()
            per_layer.append(
                {
                    "layer": i,
                    "dropped_w8": len(drop),
                    "added_w4": len(add),
                    "allocated": torch.cuda.memory_allocated(),
                    "peak_so_far": torch.cuda.max_memory_allocated(),
                }
            )

        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        final_alloc = torch.cuda.memory_allocated()

        dual_est = 0

        n_resident = len(resident)
        del resident
        torch.cuda.empty_cache()

        return {
            "device": self.device,
            "n_layers": len(layer_ids),
            "n_shared_kept_from_w8": len(shared),
            "n_tensors_after": n_resident,
            "bytes": {
                "after_full_w8_allocated": allocated_w8,
                "after_full_w8_peak": after_w8,
                "after_inplace_allocated": final_alloc,
                "after_inplace_peak": peak,
                "dual_resident_estimate": dual_est,
            },
            "gib": {
                "after_full_w8_allocated": allocated_w8 / 1024**3,
                "after_full_w8_peak": after_w8 / 1024**3,
                "after_inplace_allocated": final_alloc / 1024**3,
                "after_inplace_peak": peak / 1024**3,
                "dual_resident_estimate": dual_est / 1024**3,
            },
            "peak_vs_dual_ratio": None,
            "claim_ok_peak_below_dual": None,
            "per_layer_tail": per_layer[:3] + per_layer[-3:],
        }


def _names_by_layer(names: list[str]) -> dict[int | None, list[str]]:
    out: dict[int | None, list[str]] = {}
    for n in names:
        out.setdefault(layer_id(n), []).append(n)
    return out


class LayerwiseXferBench:
    """Layerwise W8→W4 xfer bench.

    window: stage W4[i..i+K) on host, free GPU W8[i], H2D W4[i].
    full_pin: pin entire W4 on host then H2D layerwise (discrete-GPU ref).
    """

    def __init__(self, w8_dir: Path, w4_dir: Path, device: str = "cuda:0"):
        self.w8_dir = Path(w8_dir)
        self.w4_dir = Path(w4_dir)
        self.device = device
        self._w8_index = LayerwiseWeightReplacer._build_index(self.w8_dir)
        self._w4_index = LayerwiseWeightReplacer._build_index(self.w4_dir)
        self._loader = LayerwiseWeightReplacer(self.w8_dir, self.w4_dir, self.device)

    def _load_host(
        self, names: list[str], *, pin: bool
    ) -> dict[str, Any]:
        import torch
        from safetensors import safe_open

        by_shard: dict[Path, list[str]] = {}
        for n in names:
            by_shard.setdefault(self._w4_index[n][0], []).append(n)
        out: dict[str, Any] = {}
        for shard, ns in by_shard.items():
            with safe_open(str(shard), framework="pt", device="cpu") as f:
                for n in ns:
                    t = f.get_tensor(n)
                    out[n] = t.pin_memory() if pin and t.device.type == "cpu" else t
                    del t
        return out

    def _h2d(self, host: dict[str, Any], names: list[str]) -> dict[str, Any]:
        return {n: host[n].to(self.device, non_blocking=True) for n in names}

    def run(
        self,
        *,
        mode: str = "window",
        prefetch_window: int = 2,
        pin_window: bool = True,
        warmup: int = 1,
        repeats: int = 3,
        demote_layers: list[int] | None = None,
    ) -> dict[str, Any]:
        import time

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        if mode not in ("window", "full_pin"):
            raise ValueError(f"unknown mode {mode}")
        if prefetch_window < 1:
            raise ValueError("prefetch_window must be >= 1")

        torch.cuda.set_device(self.device)
        w8_names = sorted(self._w8_index.keys())
        w4_names = sorted(self._w4_index.keys())
        w8_by_layer = _names_by_layer(w8_names)
        w4_by_layer = _names_by_layer(w4_names)
        all_layer_ids = sorted(i for i in w8_by_layer if i is not None)
        if demote_layers is None:
            layer_ids = all_layer_ids
        else:
            # Preserve caller order (rank / salience schedule).
            present = set(all_layer_ids)
            seen: set[int] = set()
            layer_ids = []
            for x in demote_layers:
                i = int(x)
                if i in present and i not in seen:
                    seen.add(i)
                    layer_ids.append(i)
            if not layer_ids:
                raise ValueError(f"demote_layers empty after filter: {demote_layers}")
        dual_est = checkpoint_nbytes(self.w8_dir) + checkpoint_nbytes(self.w4_dir)

        full_pin_host: dict[str, Any] | None = None
        preload_s = 0.0
        preload_gib = 0.0
        if mode == "full_pin":
            t0 = time.perf_counter()
            full_pin_host = self._load_host(w4_names, pin=True)
            preload_s = time.perf_counter() - t0
            preload_gib = sum(t.nbytes for t in full_pin_host.values()) / 1024**3

        def stage_bytes(host: dict[str, Any]) -> int:
            return sum(t.nbytes for t in host.values())

        def one_xfer() -> dict[str, Any]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            t_load = time.perf_counter()
            resident = self._loader._load_names(self._w8_index, w8_names, self.device)
            torch.cuda.synchronize()
            w8_load_s = time.perf_counter() - t_load
            w8_alloc = torch.cuda.memory_allocated()

            per_layer_ms: list[float] = []
            peak_unified = w8_alloc
            host_stage: dict[str, Any] = {}
            max_stage = 0

            torch.cuda.synchronize()
            t_stall = time.perf_counter()

            if mode == "full_pin":
                assert full_pin_host is not None
                for i in layer_ids:
                    t_l = time.perf_counter()
                    for n in w8_by_layer.get(i, []):
                        resident.pop(n, None)
                    add = w4_by_layer.get(i, [])
                    resident.update(self._h2d(full_pin_host, add))
                    torch.cuda.synchronize()
                    per_layer_ms.append((time.perf_counter() - t_l) * 1e3)
                    peak_unified = max(
                        peak_unified,
                        torch.cuda.memory_allocated() + int(preload_gib * 1024**3),
                    )
            else:
                for idx, i in enumerate(layer_ids):
                    t_l = time.perf_counter()
                    want = set()
                    for j in layer_ids[idx : idx + prefetch_window]:
                        want.update(w4_by_layer.get(j, []))
                    for n in list(host_stage):
                        if n not in want:
                            del host_stage[n]
                    missing = [n for n in want if n not in host_stage]
                    if missing:
                        host_stage.update(self._load_host(missing, pin=pin_window))
                    max_stage = max(max_stage, stage_bytes(host_stage))

                    for n in w8_by_layer.get(i, []):
                        resident.pop(n, None)
                    add = w4_by_layer.get(i, [])
                    resident.update(self._h2d(host_stage, add))
                    for n in add:
                        host_stage.pop(n, None)
                    torch.cuda.synchronize()
                    per_layer_ms.append((time.perf_counter() - t_l) * 1e3)
                    peak_unified = max(
                        peak_unified,
                        torch.cuda.memory_allocated() + stage_bytes(host_stage),
                    )

            stall_s = time.perf_counter() - t_stall
            peak_gpu = torch.cuda.max_memory_allocated()
            final_alloc = torch.cuda.memory_allocated()
            n_res = len(resident)
            del resident
            host_stage.clear()
            torch.cuda.empty_cache()
            return {
                "w8_load_s": w8_load_s,
                "stall_s": stall_s,
                "per_layer_ms_mean": sum(per_layer_ms) / len(per_layer_ms),
                "per_layer_ms_p50": sorted(per_layer_ms)[len(per_layer_ms) // 2],
                "per_layer_ms_max": max(per_layer_ms),
                "w8_alloc_gib": w8_alloc / 1024**3,
                "final_alloc_gib": final_alloc / 1024**3,
                "peak_gpu_gib": peak_gpu / 1024**3,
                "peak_unified_est_gib": peak_unified / 1024**3,
                "max_host_stage_gib": max_stage / 1024**3,
                "n_tensors_after": n_res,
            }

        for _ in range(max(0, warmup)):
            one_xfer()
        runs = [one_xfer() for _ in range(max(1, repeats))]

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        tmp = self._loader._load_names(self._w8_index, w8_names, self.device)
        torch.cuda.synchronize()
        del tmp
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        t_cold = time.perf_counter()
        cold = self._loader._load_names(self._w4_index, w4_names, self.device)
        torch.cuda.synchronize()
        cold_s = time.perf_counter() - t_cold
        cold_peak = torch.cuda.max_memory_allocated() / 1024**3
        del cold
        torch.cuda.empty_cache()

        def mean(key: str) -> float:
            return sum(r[key] for r in runs) / len(runs)

        stall = mean("stall_s")
        return {
            "device": self.device,
            "target": "orin_unified_32gib",
            "mode": mode,
            "prefetch_window": prefetch_window if mode == "window" else None,
            "pin_window": pin_window if mode == "window" else True,
            "n_layers": len(layer_ids),
            "n_layers_total": len(all_layer_ids),
            "demote_layers": list(layer_ids),
            "preload": {
                "s": preload_s,
                "pinned_gib": preload_gib,
                "note": (
                    "full W4 pin — discrete GPU ref only"
                    if mode == "full_pin"
                    else "no full preload; rolling window only"
                ),
            },
            "xfer": {
                "repeats": repeats,
                "warmup": warmup,
                "stall_s_mean": stall,
                "stall_s_min": min(r["stall_s"] for r in runs),
                "stall_s_max": max(r["stall_s"] for r in runs),
                "per_layer_ms_mean": mean("per_layer_ms_mean"),
                "per_layer_ms_p50": mean("per_layer_ms_p50"),
                "per_layer_ms_max_mean": mean("per_layer_ms_max"),
                "peak_gpu_gib_mean": mean("peak_gpu_gib"),
                "peak_unified_est_gib_mean": mean("peak_unified_est_gib"),
                "max_host_stage_gib_mean": mean("max_host_stage_gib"),
                "final_alloc_gib_mean": mean("final_alloc_gib"),
                "w8_alloc_gib_mean": mean("w8_alloc_gib"),
                "runs": runs,
            },
            "cold_disk_w4_load": {
                "s": cold_s,
                "peak_gpu_gib": cold_peak,
                "note": "full W4 from disk after W8 freed (tensor-only; no vLLM)",
            },
            "dual_resident_estimate_gib": dual_est / 1024**3,
            "speedup_vs_cold": (cold_s / stall) if stall > 0 else None,
            "claim_ok_unified_below_dual": mean("peak_unified_est_gib")
            < (dual_est / 1024**3) * 0.85,
            "claim_ok_faster_than_cold": stall < cold_s,
        }


PinnedLayerwiseXferBench = LayerwiseXferBench

