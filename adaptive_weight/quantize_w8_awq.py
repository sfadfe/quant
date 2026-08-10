#!/usr/bin/env python3
"""Create Qwen3-8B W8A16-AWQ (compressed-tensors), matching the W4A16-AWQ recipe.

Same AWQ mappings/calib as the W4 checkpoint; scheme W8A16
(W8A16_ASYM is not a compressed-tensors preset).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
import traceback
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier


# Same mappings as Qwen3-8B-W4A16-AWQ/recipe.yaml
AWQ_MAPPINGS = [
    {
        "smooth_layer": r"re:.*input_layernorm$",
        "balance_layers": [r"re:.*q_proj$", r"re:.*k_proj$", r"re:.*v_proj$"],
    },
    {
        "smooth_layer": r"re:.*v_proj$",
        "balance_layers": [r"re:.*o_proj$"],
    },
    {
        "smooth_layer": r"re:.*post_attention_layernorm$",
        "balance_layers": [r"re:.*gate_proj$", r"re:.*up_proj$"],
    },
    {
        "smooth_layer": r"re:.*up_proj$",
        "balance_layers": [r"re:.*down_proj$"],
    },
]


def pick_scheme(prefer: str) -> str:
    """Use a compressed-tensors preset; W8A16_ASYM does not exist → W8A16."""
    try:
        from compressed_tensors.quantization.quant_scheme import PRESET_SCHEMES

        if prefer in PRESET_SCHEMES:
            return prefer
        if "W8A16" in PRESET_SCHEMES:
            print(f"[warn] scheme {prefer} missing; using W8A16", flush=True)
            return "W8A16"
    except Exception as e:
        print(f"[warn] scheme probe failed ({e}); using {prefer}", flush=True)
    return prefer


def build_calib(tokenizer, dataset_id: str, split: str, n: int, max_len: int, seed: int):
    ds = load_dataset(dataset_id, split=f"{split}[:{max(n * 4, n)}]")
    ds = ds.shuffle(seed=seed)

    def to_text(ex):
        msgs = ex.get("messages")
        if msgs:
            try:
                return {
                    "text": tokenizer.apply_chat_template(
                        msgs,
                        tokenize=False,
                        add_generation_prompt=False,
                        enable_thinking=False,
                    )
                }
            except TypeError:
                return {
                    "text": tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=False
                    )
                }
        return {"text": str(ex.get("text") or ex.get("content") or "")}

    out = []
    for ex in ds:
        t = to_text(ex)["text"]
        if not t or not str(t).strip():
            continue
        out.append({"text": t})
        if len(out) >= n:
            break
    if len(out) < n:
        raise RuntimeError(f"only got {len(out)}/{n} calibration samples")
    from datasets import Dataset

    return Dataset.from_list(out)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="/workspace/llm_inference/quantized/Qwen3-8B-BF16",
        help="BF16 (or fp16) source checkpoint",
    )
    p.add_argument(
        "--out-dir",
        default="/workspace/llm_inference/quantized_local/Qwen3-8B-W8A16-AWQ",
    )
    p.add_argument("--scheme", default="W8A16")
    p.add_argument("--num-samples", type=int, default=128)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--dataset-id", default="HuggingFaceH4/ultrachat_200k")
    p.add_argument("--dataset-split", default="train_sft")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-memory-gib",
        type=float,
        default=10.0,
        help="Cap GPU footprint via device_map max_memory (GiB). "
        "Leave headroom for AWQ activation caches on a single 24GB card "
        "(W4-AWQ was calibrated on world_size=2).",
    )
    p.add_argument(
        "--n-grid",
        type=int,
        default=10,
        help="AWQ smoothing grid size (W4 used 20; lower = less mem/compute)",
    )
    p.add_argument(
        "--sequential-targets",
        default="Qwen3DecoderLayer",
        help="llmcompressor sequential_targets (comma-separated). "
        "DecoderLayer (not Linear): Linear breaks residual FX subgraphs "
        "(Tensor+None). DecoderLayer still sequentializes AWQ for 24GB.",
    )
    p.add_argument(
        "--awq-offload",
        default="cpu",
        help="AWQModifier.offload_device (cpu recommended on 24GB)",
    )
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scheme = pick_scheme(args.scheme)
    seq_targets = [t.strip() for t in args.sequential_targets.split(",") if t.strip()]

    # Avoid allocator fragmentation OOM (prior runs left free≪reserved).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print(f"[info] source={args.model}", flush=True)
    print(f"[info] out={out} scheme={scheme}", flush=True)
    print(
        f"[info] cuda={torch.cuda.is_available()} n={torch.cuda.device_count()} "
        f"max_mem={args.max_memory_gib}GiB sequential_targets={seq_targets} "
        f"n_grid={args.n_grid} awq_offload={args.awq_offload}",
        flush=True,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_mem = {0: f"{args.max_memory_gib}GiB", "cpu": "128GiB"}
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_mem,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    print(f"[info] building calib n={args.num_samples} L={args.max_seq_length}", flush=True)
    ds = build_calib(
        tokenizer,
        args.dataset_id,
        args.dataset_split,
        args.num_samples,
        args.max_seq_length,
        args.seed,
    )

    awq_offload = None if args.awq_offload in ("", "none", "None") else torch.device(args.awq_offload)

    recipe = [
        AWQModifier(
            mappings=AWQ_MAPPINGS,
            duo_scaling="both",
            n_grid=args.n_grid,
            **({"offload_device": awq_offload} if awq_offload is not None else {}),
        ),
        QuantizationModifier(
            targets=["Linear"],
            ignore=["lm_head"],
            scheme=scheme,
        ),
    ]

    print("[info] oneshot AWQ+Quant starting...", flush=True)
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=args.num_samples,
        batch_size=1,
        sequential_targets=seq_targets,
        sequential_offload_device="cpu",
    )

    print(f"[info] saving to {out}", flush=True)
    model.save_pretrained(out, save_compressed=True)
    tokenizer.save_pretrained(out)

    peak = 0.0
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated(0) / (1024**3)

    import compressed_tensors
    import llmcompressor
    import transformers

    meta = {
        "model_id": str(args.model),
        "model_revision": "local-bf16",
        "method": "w8a16_awq",
        "algorithm": "AWQ",
        "scheme": scheme,
        "format": "compressed-tensors",
        "calibration": {
            "used": True,
            "dataset_id": args.dataset_id,
            "dataset_split": args.dataset_split,
            "num_samples": args.num_samples,
            "max_seq_length": args.max_seq_length,
            "seed": args.seed,
            "enable_thinking": False,
            "n_grid": args.n_grid,
            "sequential_targets": seq_targets,
            "max_memory_gib": args.max_memory_gib,
            "awq_offload": args.awq_offload,
            "note": "Single-GPU 24GB path; lighter than W4-AWQ (512x2048, n_grid=20, ws=2)",
        },
        "world_size": torch.cuda.device_count(),
        "elapsed_seconds": time.time() - t0,
        "peak_gpu_memory_gib_rank0": peak,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": getattr(torch.version, "cuda", None),
            "llmcompressor": getattr(llmcompressor, "__version__", "?"),
            "compressed_tensors": getattr(compressed_tensors, "__version__", "?"),
            "transformers": transformers.__version__,
        },
    }
    (out / "quantization_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    recipe_yaml = f"""default_stage:
  default_modifiers:
    AWQModifier:
      mappings:
      - smooth_layer: re:.*input_layernorm$
        balance_layers: ['re:.*q_proj$', 're:.*k_proj$', 're:.*v_proj$']
        activation_hook_target: null
      - smooth_layer: re:.*v_proj$
        balance_layers: ['re:.*o_proj$']
        activation_hook_target: null
      - smooth_layer: re:.*post_attention_layernorm$
        balance_layers: ['re:.*gate_proj$', 're:.*up_proj$']
        activation_hook_target: null
      - smooth_layer: re:.*up_proj$
        balance_layers: ['re:.*down_proj$']
        activation_hook_target: null
      duo_scaling: both
      n_grid: {args.n_grid}
    QuantizationModifier:
      targets: [Linear]
      ignore: [lm_head]
      scheme: {scheme}
      bypass_divisibility_checks: false
"""
    (out / "recipe.yaml").write_text(recipe_yaml)

    print(json.dumps(meta, indent=2), flush=True)
    print("[done] W8A16-AWQ written", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
