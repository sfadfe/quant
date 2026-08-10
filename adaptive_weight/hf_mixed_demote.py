#!/usr/bin/env python3
"""Progressive packed W8→W4 demote under HF (no BF16 expand) + L-sweep bench.

Stock HF compressed-tensors decompresses the model to BF16 on first generate.
This path keeps packed W8/W4 via Marlin (or unpack+dequant) and demotes layers
in place. Modes: smoke | lsweep | ifstruct.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from inplace_w_replace import LayerwiseWeightReplacer, layer_id  # noqa: E402
from occupancy_ctrl import LayerRank, OccupancyCtrl  # noqa: E402

NEEDLE = "The secret code is ALPHA-7742."
NEEDLE_Q_SHORT = "What is the secret code? Reply with ONLY: ALPHA-7742"
NEEDLE_EXPECT = "ALPHA-7742"
SHORT_CTX_TOKENS = 512
FILLER = (
    "On-device inference shares a fixed VRAM budget between weights and the KV cache. "
    "As the conversation grows, the cache expands and competes with weight precision. "
    "Cross-pool control reallocates bits from the weight pool into KV capacity. "
)
DEFAULT_CHECKPOINTS = (
    2048,
    4096,
    6144,
    8192,
    10240,
    12288,
    14336,
    16384,
    18432,
    20480,
    22528,
    24576,
)
OURS_POLICY = "hf_mixed_adaptive"
BASELINE_POLICY = "fixed_w4"
FIXED_W8_POLICY = "fixed_w8"


def _mem() -> dict[str, float]:
    import torch

    torch.cuda.synchronize()
    return {
        "alloc_mib": round(torch.cuda.memory_allocated() / 1024**2, 1),
        "rsvd_mib": round(torch.cuda.memory_reserved() / 1024**2, 1),
    }


def _gpu_used_mib() -> float | None:
    try:
        import subprocess

        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        return float(out.splitlines()[0])
    except Exception:
        return None


def _remove_decompress_hook(model: Any) -> bool:
    hook = getattr(model, "ct_decompress_hook", None)
    if hook is None:
        return False
    hook.remove()
    delattr(model, "ct_decompress_hook")
    return True


def _make_scheme(num_bits: int, *, symmetric: bool, group_size: int = 128):
    from compressed_tensors.quantization.quant_args import (
        QuantizationArgs,
        QuantizationStrategy,
        QuantizationType,
    )
    from compressed_tensors.quantization.quant_scheme import QuantizationScheme

    w = QuantizationArgs(
        num_bits=num_bits,
        type=QuantizationType.INT,
        symmetric=symmetric,
        group_size=group_size,
        strategy=QuantizationStrategy.GROUP,
        observer="minmax",
    )
    return QuantizationScheme(
        targets=["Linear"],
        weights=w,
        input_activations=None,
        output_activations=None,
    )


_MARLIN_WORKSPACE: Any = None


def _marlin_workspace(device: Any):
    """One shared Marlin workspace per process (size independent of layer)."""
    global _MARLIN_WORKSPACE
    import torch
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
    )

    if _MARLIN_WORKSPACE is None or _MARLIN_WORKSPACE.device != torch.device(device):
        _MARLIN_WORKSPACE = marlin_make_workspace_new(device)
    return _MARLIN_WORKSPACE


def _clear_marlin_state(module: Any) -> int:
    """Drop Marlin buffers; return freed bytes."""
    st = getattr(module, "_marlin", None)
    if not st:
        return 0
    freed = 0
    import torch

    for v in st.values():
        if torch.is_tensor(v):
            freed += v.numel() * v.element_size()
            del v
    module._marlin = None
    return freed


def _ct_to_marlin_state(module: Any) -> dict[str, Any]:
    """Repack compressed-tensors weight_* into Marlin GPTQ layout (W4/W8)."""
    import torch
    import vllm._custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_gptq_marlin_linear,
        marlin_make_empty_g_idx,
        marlin_permute_scales,
        marlin_zero_points,
        unpack_cols,
    )
    from vllm.scalar_type import scalar_types

    scheme = module.quantization_scheme.weights
    bits = int(scheme.num_bits)
    gs = int(scheme.group_size or 128)
    sym = bool(scheme.symmetric)
    out_f, in_f = [int(x) for x in module.weight_shape.tolist()]
    device = module.weight_packed.device
    has_zp = (not sym) and getattr(module, "weight_zero_point", None) is not None

    if bits == 4 and has_zp:
        wtype = scalar_types.uint4
    elif bits == 4 and not has_zp:
        wtype = scalar_types.uint4b8
    elif bits == 8 and not has_zp:
        wtype = scalar_types.uint8b128
    elif bits == 8 and has_zp:
        # Marlin zp path is uint4-only; keep symmetric W8 empty-zp.
        raise RuntimeError("Marlin: W8 with zero-points not supported")
    else:
        raise RuntimeError(f"Marlin: unsupported bits={bits} has_zp={has_zp}")

    empty = marlin_make_empty_g_idx(device)
    w = module.weight_packed.data.T.contiguous()
    w_m = ops.gptq_marlin_repack(
        w, perm=empty, size_k=in_f, size_n=out_f, num_bits=bits
    )
    del w
    s = module.weight_scale.data.T.contiguous()
    s_m = marlin_permute_scales(s, size_k=in_f, size_n=out_f, group_size=gs)
    del s
    if has_zp:
        grouped_k = in_f // gs if gs > 0 else 1
        zp_unp = unpack_cols(
            module.weight_zero_point.data.T.contiguous(), bits, grouped_k, out_f
        )
        zp_m = marlin_zero_points(
            zp_unp, size_k=grouped_k, size_n=out_f, num_bits=bits
        )
        del zp_unp
    else:
        zp_m = empty

    return {
        "weight": w_m,
        "scale": s_m,
        "zp": zp_m,
        "g_idx": empty,
        "g_idx_sort": empty,
        "wtype": wtype,
        "in_f": in_f,
        "out_f": out_f,
        "bits": bits,
        "apply": apply_gptq_marlin_linear,
    }


def _install_packed_forward(module: Any) -> None:
    """Replace Linear.forward with on-the-fly unpack+dequant (no permanent BF16)."""
    import torch
    import torch.nn.functional as F
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
    from compressed_tensors.quantization import dequantize

    if not hasattr(module, "weight_packed"):
        return

    def packed_forward(self, x: torch.Tensor) -> torch.Tensor:
        scheme = getattr(self, "quantization_scheme", None)
        if scheme is None or scheme.weights is None:
            raise RuntimeError("packed_forward: missing quantization_scheme.weights")
        bits = int(scheme.weights.num_bits)
        packed = self.weight_packed
        scale = self.weight_scale
        shape_t = self.weight_shape
        orig = torch.Size([int(v) for v in shape_t.tolist()])
        zp = getattr(self, "weight_zero_point", None)
        zpu = None
        if zp is not None and not scheme.weights.symmetric:
            zp_shape = (*orig[:-1], scale.shape[-1])
            zpu = unpack_from_int32(zp, bits, zp_shape, packed_dim=0)
        q = unpack_from_int32(packed, bits, orig, packed_dim=1)
        weight = dequantize(x_q=q, scale=scale, zero_point=zpu)
        bias = getattr(self, "bias", None)
        return F.linear(x, weight.to(dtype=x.dtype), bias)

    module.forward = packed_forward.__get__(module, type(module))


def _install_marlin_forward(module: Any) -> None:
    """Repack CT weights to Marlin and replace forward with fused GEMM."""
    import torch

    if not hasattr(module, "weight_packed") or not hasattr(module, "quantization_scheme"):
        return

    _clear_marlin_state(module)
    st = _ct_to_marlin_state(module)
    device = module.weight_packed.device
    workspace = _marlin_workspace(device)

    # Free compressed-tensors packed params; keep weight_shape + scheme.
    for leaf in ("weight_packed", "weight_scale", "weight_zero_point", "weight"):
        if leaf in module._parameters:
            del module._parameters[leaf]
        if leaf in getattr(module, "_buffers", {}):
            del module._buffers[leaf]

    module._marlin = st

    def marlin_forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self._marlin
        return m["apply"](
            input=x,
            weight=m["weight"],
            weight_scale=m["scale"],
            weight_zp=m["zp"],
            g_idx=m["g_idx"],
            g_idx_sort_indices=m["g_idx_sort"],
            workspace=workspace,
            wtype=m["wtype"],
            output_size_per_partition=m["out_f"],
            input_size_per_partition=m["in_f"],
            is_k_full=True,
            bias=getattr(self, "bias", None),
        )

    module.forward = marlin_forward.__get__(module, type(module))


def _install_linear_forward(module: Any, *, use_marlin: bool) -> str:
    if use_marlin:
        try:
            _install_marlin_forward(module)
            return "marlin"
        except Exception as e:
            print(f"  [warn] Marlin install failed ({e}); fallback unpack", flush=True)
            _install_packed_forward(module)
            return "unpack"
    _install_packed_forward(module)
    return "unpack"


def _is_quant_linear(mod: Any) -> bool:
    if not hasattr(mod, "quantization_scheme"):
        return False
    return hasattr(mod, "weight_packed") or getattr(mod, "_marlin", None) is not None


def _iter_quant_linears(model: Any):
    for name, mod in model.named_modules():
        if _is_quant_linear(mod):
            yield name, mod


def _layer_modules(model: Any, lid: int) -> list[tuple[str, Any]]:
    prefix = f"model.layers.{lid}."
    out = []
    for name, mod in _iter_quant_linears(model):
        if name.startswith(prefix) or name.startswith(f"model.model.layers.{lid}."):
            out.append((name, mod))
    if not out:
        for name, mod in _iter_quant_linears(model):
            if f".layers.{lid}." in name:
                out.append((name, mod))
    return out


def count_bits(model: Any) -> dict[str, int]:
    n8 = n4 = other = 0
    for _, mod in _iter_quant_linears(model):
        bits = int(mod.quantization_scheme.weights.num_bits)
        if bits == 8:
            n8 += 1
        elif bits == 4:
            n4 += 1
        else:
            other += 1
    return {"n_w8_linears": n8, "n_w4_linears": n4, "n_other": other}


def demote_layer(
    model: Any,
    lid: int,
    w4_loader: LayerwiseWeightReplacer,
    device: str,
    *,
    use_marlin: bool = True,
) -> dict[str, Any]:
    """Replace one decoder layer's packed W8 params with W4 (+ZP), update scheme."""
    import torch

    mods = _layer_modules(model, lid)
    if not mods:
        raise RuntimeError(f"no quant Linears for layer {lid}")

    w4_names = [n for n in w4_loader._w4_index if layer_id(n) == lid]
    w4_t = w4_loader._load_names(w4_loader._w4_index, w4_names, device)

    freed = 0
    installed = 0
    backends: list[str] = []
    for full_name, mod in mods:
        # HF module path ↔ checkpoint key (model.layers… vs model.model.layers…).
        ckpt_prefix = None
        for cand in (
            full_name,
            full_name.replace("model.model.", "model.", 1),
        ):
            if f"{cand}.weight_packed" in w4_t or any(
                k.startswith(cand + ".") for k in w4_t
            ):
                ckpt_prefix = cand
                break
        if ckpt_prefix is None:
            alt = full_name[len("model.") :] if full_name.startswith("model.") else full_name
            if any(k.startswith("model." + alt + ".") or k.startswith(alt + ".") for k in w4_t):
                ckpt_prefix = (
                    "model." + alt
                    if any(k.startswith("model." + alt + ".") for k in w4_t)
                    else alt
                )
        if ckpt_prefix is None:
            raise RuntimeError(f"cannot map module {full_name} to W4 tensors")

        def _pop_param(m, leaf: str):
            nonlocal freed
            if leaf in m._parameters:
                t = m._parameters.pop(leaf)
                freed += t.numel() * t.element_size()
                del t
            if leaf in m._buffers:
                t = m._buffers.pop(leaf)
                freed += t.numel() * t.element_size()
                del t

        freed += _clear_marlin_state(mod)
        for leaf in ("weight_packed", "weight_scale", "weight_shape", "weight_zero_point", "weight"):
            _pop_param(mod, leaf)

        for suffix in ("weight_packed", "weight_scale", "weight_shape", "weight_zero_point"):
            key = f"{ckpt_prefix}.{suffix}"
            if key not in w4_t:
                if suffix == "weight_zero_point":
                    continue
                raise KeyError(key)
            ten = w4_t[key]
            mod.register_parameter(suffix, torch.nn.Parameter(ten, requires_grad=False))
            installed += 1

        mod.quantization_scheme = _make_scheme(4, symmetric=False, group_size=128)
        from compressed_tensors.quantization import QuantizationStatus

        mod.quantization_status = QuantizationStatus.COMPRESSED
        backends.append(_install_linear_forward(mod, use_marlin=use_marlin))

    del w4_t
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "layer": lid,
        "n_modules": len(mods),
        "freed_bytes": freed,
        "n_params_installed": installed,
        "backends": backends,
    }


def load_model(model_dir: Path, tok_dir: Path, device: str, *, use_marlin: bool = True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, CompressedTensorsConfig

    tok = AutoTokenizer.from_pretrained(str(tok_dir), trust_remote_code=True)
    qcfg = CompressedTensorsConfig(run_compressed=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        quantization_config=qcfg,
    )
    removed = _remove_decompress_hook(model)
    n_installed = 0
    backends: dict[str, int] = {}
    for _, mod in _iter_quant_linears(model):
        if "weight" in mod._parameters and "weight_packed" in getattr(mod, "_parameters", {}):
            del mod._parameters["weight"]
        b = _install_linear_forward(mod, use_marlin=use_marlin)
        backends[b] = backends.get(b, 0) + 1
        n_installed += 1
    return tok, model, {
        "decompress_hook_removed": removed,
        "packed_forwards": n_installed,
        "backends": backends,
        "use_marlin": use_marlin,
    }


def _model_device(model: Any):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return getattr(model, "device", "cuda:0")


def generate_ok(model, tok, prompt: str, max_new: int = 8) -> dict[str, Any]:
    import torch

    ids = tok(prompt, return_tensors="pt")
    dev = _model_device(model)
    ids = {k: v.to(dev) for k, v in ids.items()}
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False)
    dt = time.perf_counter() - t0
    text = tok.decode(out[0], skip_special_tokens=True)
    q = model.model.layers[0].self_attn.q_proj
    keys = list(q.state_dict().keys())
    has_marlin = getattr(q, "_marlin", None) is not None
    return {
        "ok": True,
        "text": text,
        "s": round(dt, 3),
        "l0_keys": keys,
        "l0_bits": int(q.quantization_scheme.weights.num_bits),
        "still_packed": (has_marlin or "weight_packed" in keys) and "weight" not in keys,
        "backend": "marlin" if has_marlin else ("packed" if "weight_packed" in keys else "unknown"),
    }


def _resolve_layer_rank(path: Path | None) -> LayerRank | None:
    cands: list[Path] = []
    if path is not None:
        cands.append(path)
    cands.extend(
        [
            ROOT / "results" / "layer_rank.json",
            Path("/workspace/llm_inference/adaptive_weight/results/layer_rank.json"),
        ]
    )
    for cand in cands:
        if cand.exists():
            return LayerRank.load(cand)
    return None


def score_needle(text: str) -> dict[str, Any]:
    t = (text or "").strip()
    norm = re.sub(r"[^A-Za-z0-9]", "", t.upper())
    exact = NEEDLE_EXPECT.replace("-", "").upper() in norm
    loose = bool(re.search(r"ALPHA[\s\-]?7742", t, re.I))
    return {"needle_exact": exact or loose, "needle_raw": t[:200]}


def build_short_qa(tok: Any) -> str:
    needle_ids = tok.encode(NEEDLE, add_special_tokens=False)
    filler_ids = tok.encode(FILLER, add_special_tokens=False)
    body: list[int] = []
    while len(body) < SHORT_CTX_TOKENS - len(needle_ids):
        body.extend(filler_ids)
    body = body[: max(0, SHORT_CTX_TOKENS - len(needle_ids))]
    body.extend(needle_ids)
    return tok.decode(body) + "\n\n" + NEEDLE_Q_SHORT


def build_haystack(tok: Any, target_len: int, needle_depth: float = 0.25) -> tuple[str, int]:
    needle_ids = tok.encode(NEEDLE, add_special_tokens=False)
    filler_ids = tok.encode(FILLER, add_special_tokens=False)
    if len(filler_ids) < 8:
        filler_ids = tok.encode(" padding text ", add_special_tokens=False)
    prefix_n = max(0, int(target_len * needle_depth) - len(needle_ids) // 2)
    body: list[int] = []
    while len(body) < prefix_n:
        body.extend(filler_ids)
    body = body[:prefix_n]
    body.extend(needle_ids)
    while len(body) < target_len:
        body.extend(filler_ids)
    body = body[:target_len]
    return tok.decode(body), prefix_n


def _peak_mib() -> float:
    import torch

    torch.cuda.synchronize()
    alloc = torch.cuda.max_memory_allocated() / 1024**2
    smi = _gpu_used_mib()
    return max(alloc, smi or 0.0)


def _generate_kwargs(*, use_kv_quant: bool) -> dict[str, Any]:
    """HF generate extras: QuantizedCache (quanto int4) when KV quant is on."""
    if not use_kv_quant:
        return {}
    return {
        "cache_implementation": "quantized",
        "cache_config": {"backend": "quanto", "nbits": 4},
    }


def measure_decode(
    model: Any,
    tok: Any,
    prompt: str,
    *,
    decode_tokens: int,
    warmup_tokens: int,
    speed_repeats: int,
    use_kv_quant: bool = False,
) -> dict[str, Any]:
    """Median tok/s over timed generate runs (Marlin / packed path)."""
    import torch

    dev = _model_device(model)
    ids = tok(prompt, return_tensors="pt")
    ids = {k: v.to(dev) for k, v in ids.items()}
    in_len = int(ids["input_ids"].shape[1])
    gen_extra = _generate_kwargs(use_kv_quant=use_kv_quant)

    if warmup_tokens > 0:
        with torch.no_grad():
            model.generate(
                **ids, max_new_tokens=warmup_tokens, do_sample=False, **gen_extra
            )
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    rates: list[float] = []
    last_dt = 0.0
    last_n = 0
    for _ in range(max(1, speed_repeats)):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **ids, max_new_tokens=decode_tokens, do_sample=False, **gen_extra
            )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        n = int(out.shape[1]) - in_len
        last_dt, last_n = dt, n
        if dt > 0 and n > 0:
            rates.append(n / dt)
        del out
        gc.collect()
        torch.cuda.empty_cache()

    rates_sorted = sorted(rates)
    mid = rates_sorted[len(rates_sorted) // 2] if rates_sorted else float("nan")
    return {
        "tok_per_s": mid,
        "tok_per_s_reps": rates_sorted,
        "decode_s": last_dt,
        "decode_tokens": last_n,
        "peak_mib": _peak_mib(),
        "kv_cache_dtype": "quantized_int4" if use_kv_quant else "auto",
    }


def demote_layers_live(
    model: Any,
    layers: list[int],
    w4_loader: LayerwiseWeightReplacer,
    device: str,
    *,
    use_marlin: bool = True,
) -> dict[str, Any]:
    """Progressive live demote (no engine teardown). Returns stall + bit counts."""
    import torch

    t0 = time.perf_counter()
    infos = []
    for lid in layers:
        infos.append(demote_layer(model, lid, w4_loader, device, use_marlin=use_marlin))
    torch.cuda.synchronize()
    stall = time.perf_counter() - t0
    bits = count_bits(model)
    return {
        "stall_s": round(stall, 3),
        "layers": list(layers),
        "n_layers": len(layers),
        "layer_infos": infos,
        **bits,
        "alloc_mib": _mem()["alloc_mib"],
        "nvidia_used_mib": _gpu_used_mib(),
    }


def run_hf_session(
    *,
    policy: str,
    tok: Any,
    model: Any,
    checkpoints: list[int],
    budget_gib: float,
    decode_tokens: int,
    warmup_tokens: int,
    speed_repeats: int,
    needle_depth: float,
    device: str,
    w8_dir: Path | None,
    w4_dir: Path | None,
    rank: LayerRank | None,
    waves: int,
    use_marlin: bool = True,
    transfer_t: int = 4096,
    kv_quant_t: int = 0,
    target_gib: float | None = None,
    occ_k_min: int = 2,
    occ_k_max: int = 6,
    kv_mib_per_tok: float = 0.22,
) -> dict[str, Any]:
    """One policy across L checkpoints. Soft progressive demote + optional KV quant."""
    import torch

    occ: OccupancyCtrl | None = None
    w4_loader: LayerwiseWeightReplacer | None = None
    if policy == OURS_POLICY:
        hold = float(target_gib) if target_gib is not None else None
        if hold is None:
            alloc_gib = float(torch.cuda.memory_allocated()) / 1024**3
            hold = alloc_gib + 0.75  # ≈ W8 + short-activation headroom (~9.6)
        occ = OccupancyCtrl(
            budget_gib=budget_gib,
            k_min=occ_k_min,
            k_max=occ_k_max,
            waves=waves,
            rank=rank,
            soft=True,
            target_gib=hold,
            min_fire_ctx=0,
            complete_wave2=False,
        )
        assert w8_dir is not None and w4_dir is not None
        w4_loader = LayerwiseWeightReplacer(w8_dir, w4_dir, device=device)
        print(
            f"  [{policy}] soft hold target={hold:.3f}GiB "
            f"k=[{occ_k_min},{occ_k_max}] kv_quant_t={kv_quant_t}",
            flush=True,
        )

    steps: list[dict[str, Any]] = []
    swap_events: list[dict[str, Any]] = []
    demoted: list[int] = []

    for ctx in checkpoints:
        use_kv_quant = (
            policy == OURS_POLICY and kv_quant_t > 0 and ctx >= kv_quant_t
        )
        kv_label = "quantized_int4" if use_kv_quant else "auto"
        print(f"  [{policy} L={ctx}] measure (kv={kv_label})…", flush=True)
        occ_event = None
        swap_s = 0.0
        swapped = False

        # Fire before measure on projected usage; soft may demote in small batches.
        if occ is not None and w4_loader is not None:
            while True:
                alloc_mib = float(torch.cuda.memory_allocated()) / 1024**2
                projected_gib = (alloc_mib + ctx * kv_mib_per_tok) / 1024.0
                fired = occ.maybe_fire(projected_gib, ctx_len=ctx)
                if fired is None:
                    break
                layers = [int(x) for x in fired.get("layers") or []]
                layers = [lid for lid in layers if lid not in demoted]
                if not layers:
                    break
                print(
                    f"  [{policy} L={ctx}] OCC fire {fired.get('lever')} "
                    f"projected={projected_gib:.2f}GiB "
                    f"target={occ.hold_gib:.2f}GiB k={len(layers)}…",
                    flush=True,
                )
                rec = demote_layers_live(
                    model, layers, w4_loader, device, use_marlin=use_marlin
                )
                demoted.extend(layers)
                fired["projected_gib"] = round(projected_gib, 3)
                fired["alloc_mib_before"] = round(alloc_mib, 1)
                fired["real_demote"] = rec
                fired["bits_after"] = count_bits(model)
                occ_event = fired
                swap_s += float(rec["stall_s"])
                swapped = True
                swap_events.append(
                    {
                        "at_ctx": ctx,
                        "swap_s": round(float(rec["stall_s"]), 3),
                        "occupancy_event": fired,
                    }
                )
                print(
                    f"  [{policy} L={ctx}] demote stall={rec['stall_s']:.3f}s "
                    f"w8={rec['n_w8_linears']} w4={rec['n_w4_linears']} "
                    f"alloc={rec['alloc_mib']}",
                    flush=True,
                )
                gc.collect()
                torch.cuda.empty_cache()
                if not occ.soft:
                    break

        torch.cuda.reset_peak_memory_stats()
        hay, needle_pos = build_haystack(tok, ctx, needle_depth)
        gen_extra = _generate_kwargs(use_kv_quant=use_kv_quant)

        try:
            speed = measure_decode(
                model,
                tok,
                hay,
                decode_tokens=decode_tokens,
                warmup_tokens=warmup_tokens,
                speed_repeats=speed_repeats,
                use_kv_quant=use_kv_quant,
            )
            tok_s = float(speed["tok_per_s"])
            peak = float(speed["peak_mib"])
            short_prompt = build_short_qa(tok)
            ids = tok(short_prompt, return_tensors="pt")
            ids = {k: v.to(_model_device(model)) for k, v in ids.items()}
            with torch.no_grad():
                out = model.generate(
                    **ids, max_new_tokens=12, do_sample=False, **gen_extra
                )
            ans = tok.decode(out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True)
            short_score = score_needle(ans)
            short_qa_hit = short_score["needle_exact"]
            short_qa_answer = short_score["needle_raw"]
            del out
            ok = True
            err = None
        except Exception as e:
            ok = False
            err = f"{type(e).__name__}: {e}"
            tok_s = float("nan")
            peak = _peak_mib()
            short_qa_hit = False
            short_qa_answer = f"ERR:{type(e).__name__}"
            speed = {"decode_s": 0.0, "decode_tokens": 0, "tok_per_s_reps": []}
            print(f"  [{policy} L={ctx}] FAIL: {err}", flush=True)
            if occ is not None and w4_loader is not None and count_bits(model)["n_w8_linears"] > 0:
                used_gib = float(peak) / 1024.0
                k = occ.plan_k(max(used_gib, occ.hold_gib + 0.01))
                order = rank.demote_order if rank else list(range(36))
                todo = [lid for lid in order if lid not in demoted][:k]
                if todo:
                    print(f"  [{policy} L={ctx}] emergency demote k={len(todo)}…", flush=True)
                    rec = demote_layers_live(
                        model, todo, w4_loader, device, use_marlin=use_marlin
                    )
                    demoted.extend(todo)
                    occ.demoted_layers = list(dict.fromkeys(occ.demoted_layers + todo))
                    swap_s = float(rec["stall_s"])
                    swapped = True
                    occ_event = {
                        "lever": "emergency_demote",
                        "at_ctx": ctx,
                        "k": len(todo),
                        "layers": todo,
                        "real_demote": rec,
                    }
                    swap_events.append(
                        {"at_ctx": ctx, "swap_s": swap_s, "occupancy_event": occ_event}
                    )

        bits = count_bits(model)
        is_short = bool(occ.short_phase) if occ is not None else False
        step = {
            "ctx_len": ctx,
            "ok": ok,
            "policy": policy,
            "weight_model": (
                "hf_packed_mixed"
                if policy == OURS_POLICY
                else ("hf_packed_w8" if policy == FIXED_W8_POLICY else "hf_packed_w4")
            ),
            "kv_cache_dtype": kv_label,
            "swapped": swapped,
            "swap_s": round(swap_s, 3) if swapped else 0.0,
            "peak_vram_mib": round(peak, 1),
            "tok_per_s": round(tok_s, 3) if math.isfinite(tok_s) else None,
            "tok_per_s_reps": [round(x, 3) for x in (speed.get("tok_per_s_reps") or [])],
            "decode_s": round(float(speed.get("decode_s") or 0.0), 4),
            "decode_tokens": int(speed.get("decode_tokens") or 0),
            "short_qa_hit": short_qa_hit,
            "short_qa_answer": short_qa_answer,
            "needle_pos": needle_pos,
            "is_short_phase": is_short,
            "occupancy_trigger": occ is not None,
            "occupancy_event": occ_event,
            "used_gib": round(float(peak) / 1024.0, 3),
            "n_w8_linears": bits["n_w8_linears"],
            "n_w4_linears": bits["n_w4_linears"],
            "demoted_layers": list(demoted),
            "error": err,
            "backend": "hf_mixed",
        }
        steps.append(step)
        print(
            f"  [{policy} L={ctx}] ok={ok} vram={peak:.0f} "
            f"tok/s={tok_s if math.isfinite(tok_s) else float('nan'):.1f} "
            f"qa={short_qa_hit} w8={bits['n_w8_linears']} w4={bits['n_w4_linears']} "
            f"kv={kv_label} swap={swap_s if swapped else 0:.2f}s",
            flush=True,
        )
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "ok": all(s.get("ok") for s in steps) and len(steps) == len(checkpoints),
        "policy": policy,
        "steps": steps,
        "swap_events": swap_events,
        "occupancy": occ.to_dict() if occ is not None else None,
        "demoted_layers": demoted,
        "final_bits": count_bits(model),
        "target_gib": occ.hold_gib if occ is not None else None,
        "kv_quant_t": kv_quant_t if policy == OURS_POLICY else None,
    }


def beat_w4_gates_simple(
    ours_name: str,
    flat: list[dict[str, Any]],
    transfer_t: int = 4096,
    kv_quant_t: int = 0,
    target_gib: float | None = None,
    budget_gib: float = 12.0,
) -> dict[str, Any]:
    """Gates vs fixed_w4: hold W8 for accuracy within budget; soft demote for VRAM."""
    w4 = [s for s in flat if s.get("policy") == BASELINE_POLICY and s.get("ok")]
    ours = [s for s in flat if s.get("policy") == ours_name and s.get("ok")]
    by_w4 = {int(s["ctx_len"]): s for s in w4}
    by_o = {int(s["ctx_len"]): s for s in ours}
    shared = sorted(set(by_w4) & set(by_o))
    budget_mib = budget_gib * 1024.0

    tok_ok = True
    tok_detail = []
    for L in shared:
        a, b = by_o[L].get("tok_per_s"), by_w4[L].get("tok_per_s")
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            tol = 0.90 if L < transfer_t else 0.97
            good = a >= b * tol
            tok_ok = tok_ok and good
            tok_detail.append({"L": L, "ours": a, "w4": b, "tol": tol, "ok": good})

    early = [L for L in shared if L < transfer_t]
    later = [L for L in shared if L >= transfer_t]
    early_qa_ok = True
    for L in early:
        oa, wa = by_o[L].get("short_qa_hit"), by_w4[L].get("short_qa_hit")
        if oa is None and wa is None:
            continue
        early_qa_ok = early_qa_ok and bool(oa)

    later_acc_ok = True
    for L in later:
        oa, wa = by_o[L].get("short_qa_hit"), by_w4[L].get("short_qa_hit")
        if oa is None or wa is None:
            continue
        later_acc_ok = later_acc_ok and (bool(oa) >= bool(wa))

    vram_ok = True
    vram_detail = []
    for L in shared:
        ov, wv = by_o[L].get("peak_vram_mib"), by_w4[L].get("peak_vram_mib")
        if not isinstance(ov, (int, float)):
            continue
        n_w8 = int(by_o[L].get("n_w8_linears") or 0)
        kv = str(by_o[L].get("kv_cache_dtype") or "auto")
        good = ov <= budget_mib
        band = "within_budget"
        if kv_quant_t > 0 and L >= kv_quant_t and isinstance(wv, (int, float)):
            good = good and ov < wv and kv.startswith("quantized")
            band = "late_lt_w4"
        vram_ok = vram_ok and good
        vram_detail.append(
            {
                "L": L,
                "ours": ov,
                "w4": wv,
                "n_w8": n_w8,
                "kv": kv,
                "band": band,
                "budget_mib": round(budget_mib, 1),
                "ok": good,
            }
        )

    passed = bool(shared) and tok_ok and early_qa_ok and later_acc_ok and vram_ok
    return {
        "passed": passed,
        "shared_L": shared,
        "tok_s_ge_w4": tok_ok,
        "tok_detail": tok_detail,
        "early_qa_ok": early_qa_ok,
        "later_acc_ok": later_acc_ok,
        "vram_ok": vram_ok,
        "vram_detail": vram_detail,
        "transfer_t": transfer_t,
        "kv_quant_t": kv_quant_t,
        "target_gib": target_gib,
        "budget_gib": budget_gib,
        "backend": "hf_mixed",
        "note": "HF soft demote (KV quant off by default) vs fixed_w4",
    }


def _parse_lsweep_policies(raw: str) -> list[str]:
    allowed = {BASELINE_POLICY, FIXED_W8_POLICY, OURS_POLICY}
    pols = [p.strip() for p in str(raw).split(",") if p.strip()]
    bad = [p for p in pols if p not in allowed]
    if bad:
        raise SystemExit(f"unknown --lsweep-policies {bad}; allowed={sorted(allowed)}")
    if not pols:
        raise SystemExit("empty --lsweep-policies")
    return pols


def run_lsweep(args: argparse.Namespace) -> int:
    import torch

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = [int(x) for x in args.checkpoints]
    rank = _resolve_layer_rank(args.layer_rank)
    policies = _parse_lsweep_policies(
        getattr(args, "lsweep_policies", None)
        or f"{BASELINE_POLICY},{FIXED_W8_POLICY},{OURS_POLICY}"
    )
    merge_into = getattr(args, "merge_into", None)
    flat: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    if merge_into is not None:
        merge_path = Path(merge_into)
        prev_steps = merge_path / "xfer_steps.json"
        prev_pols = merge_path / "xfer_policies.json"
        if prev_steps.exists():
            flat = json.loads(prev_steps.read_text())
            flat = [s for s in flat if s.get("policy") not in set(policies)]
        if prev_pols.exists():
            policy_rows = [
                r
                for r in json.loads(prev_pols.read_text())
                if r.get("policy") not in set(policies)
            ]
        out_dir = merge_path
        out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "stamp": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "backend": "hf_mixed",
        "checkpoints": checkpoints,
        "budget_gib": args.budget_gib,
        "decode_tokens": args.decode_tokens,
        "speed_repeats": args.speed_repeats,
        "warmup_tokens": args.warmup_tokens,
        "baseline": BASELINE_POLICY,
        "fixed_w8": FIXED_W8_POLICY,
        "ours": OURS_POLICY,
        "lsweep_policies": policies,
        "layer_rank": rank.path if rank else None,
        "waves": args.waves,
        "use_marlin": bool(args.use_marlin),
        "transfer_t": args.transfer_t,
        "kv_quant_t": args.kv_quant_t,
        "occ_k_min": args.occ_k_min,
        "occ_k_max": args.occ_k_max,
        "kv_mib_per_tok": args.kv_mib_per_tok,
        "hold_headroom_gib": args.hold_headroom_gib,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    res_w4: dict[str, Any] | None = None
    res_w8: dict[str, Any] | None = None
    res_m: dict[str, Any] | None = None

    if BASELINE_POLICY in policies:
        print(f"[hf_mixed lsweep] load {BASELINE_POLICY}…", flush=True)
        tok, model_w4, load_w4 = load_model(
            args.w4_dir, args.tok_dir, args.device, use_marlin=args.use_marlin
        )
        print(f"  load {load_w4} bits={count_bits(model_w4)} mem={_mem()}", flush=True)
        res_w4 = run_hf_session(
            policy=BASELINE_POLICY,
            tok=tok,
            model=model_w4,
            checkpoints=checkpoints,
            budget_gib=args.budget_gib,
            decode_tokens=args.decode_tokens,
            warmup_tokens=args.warmup_tokens,
            speed_repeats=args.speed_repeats,
            needle_depth=args.needle_depth,
            device=args.device,
            w8_dir=None,
            w4_dir=None,
            rank=None,
            waves=args.waves,
            use_marlin=args.use_marlin,
            transfer_t=args.transfer_t,
            kv_quant_t=0,
            kv_mib_per_tok=args.kv_mib_per_tok,
        )
        (out_dir / f"raw_{BASELINE_POLICY}.json").write_text(
            json.dumps(res_w4, indent=2) + "\n"
        )
        flat.extend(res_w4["steps"])
        policy_rows.append({"policy": BASELINE_POLICY, "ok": res_w4["ok"]})
        del model_w4
        gc.collect()
        torch.cuda.empty_cache()

    if FIXED_W8_POLICY in policies:
        print(f"[hf_mixed lsweep] load {FIXED_W8_POLICY}…", flush=True)
        tok, model_w8, load_w8 = load_model(
            args.w8_dir, args.tok_dir, args.device, use_marlin=args.use_marlin
        )
        print(f"  load {load_w8} bits={count_bits(model_w8)} mem={_mem()}", flush=True)
        res_w8 = run_hf_session(
            policy=FIXED_W8_POLICY,
            tok=tok,
            model=model_w8,
            checkpoints=checkpoints,
            budget_gib=args.budget_gib,
            decode_tokens=args.decode_tokens,
            warmup_tokens=args.warmup_tokens,
            speed_repeats=args.speed_repeats,
            needle_depth=args.needle_depth,
            device=args.device,
            w8_dir=None,
            w4_dir=None,
            rank=None,
            waves=args.waves,
            use_marlin=args.use_marlin,
            transfer_t=args.transfer_t,
            kv_quant_t=0,
            kv_mib_per_tok=args.kv_mib_per_tok,
        )
        (out_dir / f"raw_{FIXED_W8_POLICY}.json").write_text(
            json.dumps(res_w8, indent=2) + "\n"
        )
        flat.extend(res_w8["steps"])
        policy_rows.append({"policy": FIXED_W8_POLICY, "ok": res_w8["ok"]})
        del model_w8
        gc.collect()
        torch.cuda.empty_cache()

    if OURS_POLICY in policies:
        print(f"[hf_mixed lsweep] load {OURS_POLICY} (W8 packed)…", flush=True)
        tok, model_m, load_m = load_model(
            args.w8_dir, args.tok_dir, args.device, use_marlin=args.use_marlin
        )
        mem0 = _mem()
        print(f"  load {load_m} bits={count_bits(model_m)} mem={mem0}", flush=True)
        alloc_gib = float(mem0["alloc_mib"]) / 1024.0
        hold = alloc_gib + float(args.hold_headroom_gib)
        res_m = run_hf_session(
            policy=OURS_POLICY,
            tok=tok,
            model=model_m,
            checkpoints=checkpoints,
            budget_gib=args.budget_gib,
            decode_tokens=args.decode_tokens,
            warmup_tokens=args.warmup_tokens,
            speed_repeats=args.speed_repeats,
            needle_depth=args.needle_depth,
            device=args.device,
            w8_dir=args.w8_dir,
            w4_dir=args.w4_dir,
            rank=rank,
            waves=args.waves,
            use_marlin=args.use_marlin,
            transfer_t=args.transfer_t,
            kv_quant_t=args.kv_quant_t,
            target_gib=hold,
            occ_k_min=args.occ_k_min,
            occ_k_max=args.occ_k_max,
            kv_mib_per_tok=args.kv_mib_per_tok,
        )
        (out_dir / f"raw_{OURS_POLICY}.json").write_text(json.dumps(res_m, indent=2) + "\n")
        flat.extend(res_m["steps"])
        policy_rows.append(
            {
                "policy": OURS_POLICY,
                "ok": res_m["ok"],
                "occupancy": res_m.get("occupancy"),
                "final_bits": res_m.get("final_bits"),
                "demoted_layers": res_m.get("demoted_layers"),
                "target_gib": res_m.get("target_gib"),
                "kv_quant_t": res_m.get("kv_quant_t"),
            }
        )
        del model_m
        gc.collect()
        torch.cuda.empty_cache()

    (out_dir / "xfer_steps.json").write_text(json.dumps(flat, indent=2) + "\n")
    (out_dir / "xfer_policies.json").write_text(json.dumps(policy_rows, indent=2) + "\n")
    target = None if res_m is None else res_m.get("target_gib")
    verdict = beat_w4_gates_simple(
        OURS_POLICY,
        flat,
        transfer_t=args.transfer_t,
        kv_quant_t=args.kv_quant_t,
        target_gib=target,
        budget_gib=args.budget_gib,
    )
    verdict["policies_ok"] = {r["policy"]: r["ok"] for r in policy_rows}
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps({"verdict_passed": verdict["passed"], **verdict["policies_ok"]}, indent=2), flush=True)
    print(f"[hf_mixed lsweep] wrote {out_dir}", flush=True)
    any_ok = any(bool(r.get("ok")) for r in policy_rows) or any(
        bool(s.get("ok")) for s in flat
    )
    return 0 if any_ok else 1


def run_smoke(args: argparse.Namespace) -> int:
    import torch

    report: dict[str, Any] = {
        "w8_dir": str(args.w8_dir),
        "w4_dir": str(args.w4_dir),
        "k": args.k,
        "steps": [],
    }

    print("[hf_mixed] load W8 packed…", flush=True)
    tok, model, load_meta = load_model(
        args.w8_dir, args.tok_dir, args.device, use_marlin=args.use_marlin
    )
    report["load"] = load_meta
    report["after_load"] = {**_mem(), "nvidia_used_mib": _gpu_used_mib(), **count_bits(model)}
    print(f"  {report['after_load']}", flush=True)

    rank = _resolve_layer_rank(args.layer_rank)
    order = rank.demote_order if rank else list(range(36))
    if rank:
        report["layer_rank"] = rank.path
    report["demote_order_head"] = order[: args.k]

    print("[hf_mixed] generate before demote (must stay packed)…", flush=True)
    try:
        g0 = generate_ok(model, tok, "The capital of France is")
        report["gen_before"] = g0
        print(
            f"  gen_before ok={g0['ok']} packed={g0['still_packed']} "
            f"bits={g0['l0_bits']} text={g0['text']!r}",
            flush=True,
        )
    except Exception as e:
        report["gen_before"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        print(f"  FAIL gen_before: {e}", flush=True)
        _dump(args.out, report)
        return 1

    if not g0.get("still_packed"):
        report["verdict"] = "FAIL_DECOMPRESSED_TO_BF16"
        print("  FAIL: weights expanded to BF16", flush=True)
        _dump(args.out, report)
        return 1

    mem_before = report["after_load"]["alloc_mib"]
    w4_loader = LayerwiseWeightReplacer(args.w8_dir, args.w4_dir, device=args.device)
    demoted: list[int] = []
    print(f"[hf_mixed] demote K={args.k} layers…", flush=True)
    for lid in order[: args.k]:
        info = demote_layer(
            model, lid, w4_loader, args.device, use_marlin=args.use_marlin
        )
        demoted.append(lid)
        step = {
            **info,
            **_mem(),
            "nvidia_used_mib": _gpu_used_mib(),
            **count_bits(model),
            "demoted_layers": list(demoted),
        }
        report["steps"].append(step)
        print(
            f"  L{lid} freed_mib={info['freed_bytes']/1024**2:.1f} "
            f"alloc={step['alloc_mib']} w8={step['n_w8_linears']} w4={step['n_w4_linears']}",
            flush=True,
        )

    report["after_demote"] = {
        **_mem(),
        "nvidia_used_mib": _gpu_used_mib(),
        **count_bits(model),
        "demoted_layers": demoted,
    }
    print(f"  after_demote {report['after_demote']}", flush=True)

    print("[hf_mixed] generate after mixed demote…", flush=True)
    try:
        g1 = generate_ok(model, tok, "The capital of France is")
        report["gen_after"] = g1
        print(
            f"  gen_after ok={g1['ok']} packed={g1['still_packed']} "
            f"L0_bits={g1['l0_bits']} text={g1['text']!r}",
            flush=True,
        )
    except Exception as e:
        report["gen_after"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        print(f"  FAIL gen_after: {e}", flush=True)
        report["verdict"] = "FAIL_MIXED_GENERATE"
        _dump(args.out, report)
        return 1

    mem_after = report["after_demote"]["alloc_mib"]
    drop = mem_before - mem_after
    report["alloc_drop_mib"] = round(drop, 1)
    bits = count_bits(model)
    ok = (
        bool(g0.get("ok"))
        and bool(g1.get("ok"))
        and bool(g0.get("still_packed"))
        and bool(g1.get("still_packed"))
        and bits["n_w4_linears"] > 0
        and bits["n_w8_linears"] > 0
        and drop > 50.0
    )
    report["verdict"] = "PASS" if ok else "FAIL"
    report["ok_criteria"] = {
        "gen_before": g0.get("ok"),
        "gen_after": g1.get("ok"),
        "stayed_packed": g0.get("still_packed") and g1.get("still_packed"),
        "mixed_bits": bits["n_w4_linears"] > 0 and bits["n_w8_linears"] > 0,
        "vram_dropped": drop > 50.0,
        "alloc_drop_mib": round(drop, 1),
    }
    print(json.dumps({"verdict": report["verdict"], **report["ok_criteria"]}, indent=2), flush=True)
    _dump(args.out, report)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return 0 if ok else 1


def _dump(path: Path | None, report: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[hf_mixed] wrote {path}", flush=True)


def _import_ifstruct():
    """Reuse IFStruct grading from repo-root qwen_ifstruct_eval.py."""
    root = ROOT.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import qwen_ifstruct_eval as ifs  # noqa: WPS001

    return ifs


def _apply_chat(tok: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tok.apply_chat_template(
            messages, enable_thinking=False, **kwargs
        )
    except TypeError:
        return tok.apply_chat_template(messages, **kwargs)


def _eval_ifstruct_one(
    *,
    label: str,
    model_dir: Path,
    tok_dir: Path,
    device: str,
    use_marlin: bool,
    examples: list[dict[str, Any]],
    ifs: Any,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch

    print(f"[ifstruct] load {label} from {model_dir}", flush=True)
    t_load = time.perf_counter()
    tok, model, meta = load_model(
        model_dir, tok_dir, device, use_marlin=use_marlin
    )
    load_s = time.perf_counter() - t_load
    bits = count_bits(model)
    print(f"  load {meta} bits={bits} mem={_mem()}", flush=True)

    samples: list[dict[str, Any]] = []
    t_gen0 = time.perf_counter()
    for i, example in enumerate(examples):
        text = _apply_chat(tok, example["prompt"])
        ids = tok(text, return_tensors="pt")
        ids = {k: v.to(_model_device(model)) for k, v in ids.items()}
        prompt_n = int(ids["input_ids"].shape[1])
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        dt = time.perf_counter() - t0
        gen_ids = out[0][prompt_n:]
        response = tok.decode(gen_ids, skip_special_tokens=True)
        validation = ifs.validate_response(response, example)
        finish = "length" if len(gen_ids) >= max_new_tokens else "stop"
        samples.append(
            {
                "seed": example["seed"],
                "entity_type": example["entity_type"],
                "output_format": example["output_format"],
                "require_wrapper_key": example["require_wrapper_key"],
                "require_code_block": example["require_code_block"],
                "require_no_commentary": example["require_no_commentary"],
                "passed": validation.passed,
                "errors": validation.errors,
                "validation_details": validation.details,
                "prompt": example["prompt"],
                "response": response,
                "prompt_tokens": prompt_n,
                "output_tokens": int(len(gen_ids)),
                "finish_reason": finish,
                "latency_seconds": round(dt, 6),
            }
        )
        if (i + 1) % 10 == 0 or i == 0:
            acc = ifs.build_accuracy_summary(samples)
            print(
                f"  [{label} {i+1}/{len(examples)}] "
                f"pass={acc['passed']}/{acc['total']} "
                f"({100*acc['pass_rate']:.1f}%)",
                flush=True,
            )

    gen_s = time.perf_counter() - t_gen0
    accuracy = ifs.build_accuracy_summary(samples)
    result = {
        "label": label,
        "model": model_dir.name,
        "model_path": str(model_dir),
        "status": "SUCCESS",
        "load_seconds": round(load_s, 3),
        "generation_seconds": round(gen_s, 3),
        "bits": bits,
        "backends": meta.get("backends"),
        "accuracy": accuracy,
        "samples": samples,
        "mem": _mem(),
    }
    print(
        f"[ifstruct] {label} done pass_rate={accuracy['pass_rate']:.4f} "
        f"({accuracy['passed']}/{accuracy['total']})",
        flush=True,
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_ifstruct(args: argparse.Namespace) -> int:
    """IFStruct pass_rate: Adaptive-Weight (W8 start) and/or AWQ int4."""
    ifs = _import_ifstruct()
    dataset = ifs.ensure_dataset(args.dataset)
    examples = ifs.load_jsonl(dataset)
    if not examples:
        print(f"FAIL: empty dataset {dataset}", flush=True)
        return 2
    if args.n_samples and args.n_samples > 0:
        n = min(args.n_samples, len(examples))
        if args.sample_seed is not None:
            import random

            rng = random.Random(args.sample_seed)
            examples = rng.sample(examples, n)
        else:
            examples = examples[:n]

    policies = [p.strip() for p in args.ifstruct_policies.split(",") if p.strip()]
    known = {"Adaptive-Weight", "AWQ int4"}
    bad = [p for p in policies if p not in known]
    if bad or not policies:
        print(f"FAIL: --ifstruct-policies must be subset of {sorted(known)}", flush=True)
        return 2

    out_dir = args.out_dir
    assert out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}
    if "Adaptive-Weight" in policies:
        ours = _eval_ifstruct_one(
            label="Adaptive-Weight",
            model_dir=args.w8_dir,
            tok_dir=args.tok_dir,
            device=args.device,
            use_marlin=args.use_marlin,
            examples=examples,
            ifs=ifs,
            max_new_tokens=args.max_tokens,
        )
        (out_dir / "Adaptive-Weight.json").write_text(
            json.dumps(ours, indent=2) + "\n", encoding="utf-8"
        )
        results["Adaptive-Weight"] = ours

    if "AWQ int4" in policies:
        awq = _eval_ifstruct_one(
            label="AWQ int4",
            model_dir=args.w4_dir,
            tok_dir=args.tok_dir,
            device=args.device,
            use_marlin=args.use_marlin,
            examples=examples,
            ifs=ifs,
            max_new_tokens=args.max_tokens,
        )
        (out_dir / "AWQ_int4.json").write_text(
            json.dumps(awq, indent=2) + "\n", encoding="utf-8"
        )
        results["AWQ int4"] = awq

    policy_summary = {
        name: {
            "pass_rate": r["accuracy"]["pass_rate"],
            "passed": r["accuracy"]["passed"],
            "total": r["accuracy"]["total"],
            "by_format": r["accuracy"]["by_format"],
            "model": r["model"],
        }
        for name, r in results.items()
    }
    summary: dict[str, Any] = {
        "dataset": str(dataset),
        "dataset_sha256": ifs.sha256_file(dataset),
        "ifstruct_commit": ifs.IFSTRUCT_COMMIT,
        "n_samples": len(examples),
        "sample_seed": args.sample_seed,
        "max_tokens": args.max_tokens,
        "policies": policy_summary,
    }
    if "Adaptive-Weight" in results and "AWQ int4" in results:
        summary["delta_points"] = round(
            100.0
            * (
                results["Adaptive-Weight"]["accuracy"]["pass_rate"]
                - results["AWQ int4"]["accuracy"]["pass_rate"]
            ),
            3,
        )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode", choices=("smoke", "lsweep", "ifstruct"), default="smoke"
    )
    ap.add_argument(
        "--w8-dir",
        type=Path,
        default=Path("/workspace/llm_inference/quantized_local/Qwen3-8B-W8A16-AWQ"),
    )
    ap.add_argument(
        "--w4-dir",
        type=Path,
        default=Path("/workspace/llm_inference/quantized/Qwen3-8B-W4A16-AWQ"),
    )
    ap.add_argument(
        "--tok-dir",
        type=Path,
        default=Path("/workspace/llm_inference/quantized/Qwen3-8B-BF16"),
    )
    ap.add_argument("--layer-rank", type=Path, default=None)
    ap.add_argument("--k", type=int, default=8, help="smoke: layers to demote")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=None, help="smoke JSON path")
    ap.add_argument("--out-dir", type=Path, default=None, help="lsweep/ifstruct output dir")
    ap.add_argument("--checkpoints", type=int, nargs="+", default=list(DEFAULT_CHECKPOINTS))
    ap.add_argument("--budget-gib", type=float, default=12.0)
    ap.add_argument("--decode-tokens", type=int, default=64)
    ap.add_argument("--warmup-tokens", type=int, default=8)
    ap.add_argument("--speed-repeats", type=int, default=3)
    ap.add_argument("--needle-depth", type=float, default=0.25)
    ap.add_argument("--waves", type=int, default=2)
    ap.add_argument("--transfer-t", type=int, default=4096)
    ap.add_argument(
        "--kv-quant-t",
        type=int,
        default=0,
        help="If >0, enable HF QuantizedCache (quanto int4) for ours at L>=this; 0=off",
    )
    ap.add_argument(
        "--hold-headroom-gib",
        type=float,
        default=0.50,
        help="soft target = W8_alloc + headroom (~9.4 GiB; between 0.35 early / 0.75 late)",
    )
    ap.add_argument("--occ-k-min", type=int, default=2)
    ap.add_argument(
        "--occ-k-max",
        type=int,
        default=6,
        help="max layers demoted per soft fire (small steps)",
    )
    ap.add_argument(
        "--kv-mib-per-tok",
        type=float,
        default=0.23,
        help="KV+activation MiB/token for projected-usage fire check",
    )
    ap.add_argument(
        "--no-marlin",
        action="store_true",
        help="Use slow unpack+dequant forward instead of Marlin GEMM",
    )
    ap.add_argument(
        "--dataset",
        type=Path,
        default=Path("/workspace/llm_inference/quantized/ifstruct_sample_100.jsonl"),
        help="ifstruct: pre-sampled IFStruct JSONL",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="ifstruct: max new tokens per sample",
    )
    ap.add_argument(
        "--n-samples",
        type=int,
        default=0,
        help="ifstruct: use N dataset rows (0=all)",
    )
    ap.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="ifstruct: if set with --n-samples, random.sample N rows with this seed",
    )
    ap.add_argument(
        "--ifstruct-policies",
        default="Adaptive-Weight,AWQ int4",
        help="ifstruct: comma list of Adaptive-Weight and/or AWQ int4",
    )
    ap.add_argument(
        "--lsweep-policies",
        default=f"{BASELINE_POLICY},{FIXED_W8_POLICY},{OURS_POLICY}",
        help="lsweep: comma list of fixed_w4,fixed_w8,hf_mixed_adaptive",
    )
    ap.add_argument(
        "--merge-into",
        type=Path,
        default=None,
        help="lsweep: write/merge into this existing run dir (keeps other policies)",
    )
    args = ap.parse_args()
    args.use_marlin = not args.no_marlin

    import torch

    if not torch.cuda.is_available():
        print("FAIL: CUDA required", flush=True)
        return 2

    if args.mode == "lsweep":
        if args.out_dir is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            args.out_dir = Path("/workspace/llm_inference/results") / stamp
        return run_lsweep(args)
    if args.mode == "ifstruct":
        if args.out_dir is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            args.out_dir = Path("/workspace/llm_inference/ifstruct_results") / stamp
        return run_ifstruct(args)
    return run_smoke(args)


if __name__ == "__main__":
    raise SystemExit(main())
