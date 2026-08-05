"""Measure memory and latency separately on Qwen3-8B checkpoints.

Each checkpoint runs in a fresh subprocess for each measurement kind:

* ``memory`` uses a small, fixed KV cache so weight quantization is visible in
  peak memory instead of vLLM filling the freed memory with KV-cache blocks.
* ``latency`` uses the requested production-like KV-cache policy and measures
  exactly one four-request batch after warmup.

Both Jetson unified system memory and vLLM worker CUDA/model memory are saved.
Worker-side PyTorch traces are optional and apply only to the latency run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = ROOT / "quantized"
DEFAULT_RESULTS_DIR = ROOT / "orin_batch4_profiles"
BATCH_SIZE = 4
MEMORY_PROFILE = "memory"
LATENCY_PROFILE = "latency"
PROFILE_KINDS = (MEMORY_PROFILE, LATENCY_PROFILE)
PROMPTS = [
    "Explain in two concise paragraphs why continuous batching improves LLM inference throughput.",
    (
        "Write a Python function named chunked that splits a list into fixed-size chunks. "
        "Include type hints, input validation, and one usage example."
    ),
    (
        "Return only a JSON object with these keys: severity, root_cause, next_action. "
        "A web API's p99 latency doubled immediately after a database index was removed."
    ),
    (
        "A team must deploy an 8B language model on an edge device with shared CPU/GPU "
        "memory. Compare BF16, weight-only INT4, and W8A8 quantization, then recommend "
        "what to measure before choosing. Keep the answer under 150 words."
    ),
]


def read_ram_mib() -> tuple[float, float]:
    values: dict[str, int] = {}
    with Path("/proc/meminfo").open(encoding="utf-8") as file:
        for line in file:
            key, value = line.split(":", 1)
            values[key] = int(value.strip().split()[0])
    total_kib = values["MemTotal"]
    used_kib = total_kib - values["MemAvailable"]
    return used_kib / 1024, total_kib / 1024


class UnifiedMemoryMonitor:
    """Sample system RAM, which is CPU/GPU unified memory on Jetson."""

    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = interval_seconds
        self.start_mib, self.total_mib = read_ram_mib()
        self.peak_mib = self.start_mib
        self.end_mib = self.start_mib
        self.samples = 1
        self.checkpoints: dict[str, dict[str, float]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._sample()
        self._stop.set()
        self._thread.join(timeout=5)
        self._sample()

    def _sample(self) -> None:
        used_mib, _ = read_ram_mib()
        with self._lock:
            self.end_mib = used_mib
            self.peak_mib = max(self.peak_mib, used_mib)
            self.samples += 1

    def checkpoint(self, name: str) -> None:
        """Record current and peak-so-far system unified memory."""
        self._sample()
        with self._lock:
            self.checkpoints[name] = {
                "used_mib": round(self.end_mib, 3),
                "delta_from_start_mib": round(self.end_mib - self.start_mib, 3),
                "peak_so_far_mib": round(self.peak_mib, 3),
                "peak_delta_so_far_mib": round(
                    max(0.0, self.peak_mib - self.start_mib), 3
                ),
            }

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def result(self) -> dict[str, Any]:
        with self._lock:
            return {
                "measurement": (
                    "system RAM used = MemTotal - MemAvailable "
                    "(CPU/GPU unified memory on Jetson)"
                ),
                "scope": "whole_system",
                "start_mib": round(self.start_mib, 3),
                "peak_mib": round(self.peak_mib, 3),
                "peak_delta_mib": round(
                    max(0.0, self.peak_mib - self.start_mib), 3
                ),
                "end_mib": round(self.end_mib, 3),
                "total_mib": round(self.total_mib, 3),
                "samples": self.samples,
                "interval_seconds": self.interval_seconds,
                "checkpoints": dict(self.checkpoints),
            }


class MemoryStatsWorkerExtension:
    """vLLM worker RPCs for memory stats without insecure callable RPCs."""

    def batch4_memory_stats(self) -> dict[str, int]:
        import torch

        model_runner = getattr(self, "model_runner", None)
        return {
            "rank": int(getattr(self, "rank", 0)),
            "model_memory_usage_bytes": int(
                getattr(model_runner, "model_memory_usage", 0)
            ),
            "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated()),
            "cuda_memory_reserved_bytes": int(torch.cuda.memory_reserved()),
            "cuda_max_memory_allocated_bytes": int(
                torch.cuda.max_memory_allocated()
            ),
            "cuda_max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }

    def batch4_reset_peak_memory_stats(self) -> None:
        import torch

        torch.cuda.reset_peak_memory_stats()


def atomic_write_json(path: Path, value: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def model_metadata(model_path: Path) -> dict[str, Any]:
    path = model_path / "quantization_metadata.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def runtime_hf_override(model_path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    """Clear runtime-irrelevant actorder rejected by dense W8A8 vLLM paths."""
    config_path = model_path / "config.json"
    if not config_path.is_file():
        return None, []
    config = json.loads(config_path.read_text(encoding="utf-8"))
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict) or quantization.get("format") != "int-quantized":
        return None, []

    changes: list[str] = []
    groups = quantization.get("config_groups", {})
    for name, group in groups.items() if isinstance(groups, dict) else ():
        if not isinstance(group, dict):
            continue
        weights = group.get("weights")
        activations = group.get("input_activations")
        if not isinstance(weights, dict) or not isinstance(activations, dict):
            continue
        is_w8a8 = (
            weights.get("num_bits") == 8
            and weights.get("strategy") == "channel"
            and activations.get("num_bits") == 8
            and activations.get("strategy") == "token"
            and activations.get("dynamic") is True
        )
        if is_w8a8 and weights.get("actorder") in ("static", "weight"):
            old = weights["actorder"]
            weights["actorder"] = None
            changes.append(f"config_groups.{name}.weights.actorder: {old} -> null")
    return ({"quantization_config": quantization}, changes) if changes else (None, [])

MEMORY_SUMMARY_FIELDS = [
    "profile_kind",
    "model",
    "method",
    "status",
    "error",
    "batch_size",
    "load_seconds",
    "warmup_seconds",
    "memory_workload_seconds",
    "model_memory_usage_mib",
    "cuda_after_load_allocated_mib",
    "cuda_after_load_reserved_mib",
    "cuda_after_load_peak_allocated_mib",
    "cuda_workload_peak_allocated_mib",
    "ram_peak_mib",
    "ram_peak_delta_mib",
    "kv_cache_memory_mib",
    "max_model_len",
    "max_tokens",
    "vllm_version",
    "result_file",
    "log_file",
]

LATENCY_SUMMARY_FIELDS = [
    "profile_kind",
    "model",
    "method",
    "status",
    "error",
    "batch_size",
    "batch_latency_seconds",
    "prompt_tokens",
    "output_tokens",
    "output_tokens_per_second",
    "load_seconds",
    "warmup_seconds",
    "gpu_memory_utilization",
    "max_model_len",
    "max_tokens",
    "torch_profiler_enabled",
    "vllm_version",
    "trace_dir",
    "result_file",
    "log_file",
]

MEMORY_COMPARISON_FIELDS = [
    "model",
    "method",
    "status",
    "model_memory_usage_mib",
    "bf16_model_memory_reduction_mib",
    "bf16_model_memory_reduction_percent",
    "cuda_after_load_allocated_mib",
    "bf16_cuda_after_load_reduction_mib",
    "cuda_workload_peak_allocated_mib",
    "ram_peak_delta_mib",
    "bf16_ram_peak_delta_reduction_mib",
]

LATENCY_COMPARISON_FIELDS = [
    "model",
    "method",
    "status",
    "batch_latency_seconds",
    "bf16_latency_speedup",
    "output_tokens_per_second",
    "bf16_throughput_ratio",
    "output_tokens",
    "trace_dir",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument(
        "--model",
        type=Path,
        action="append",
        help="Model directory; repeat to run selected models (default: all including BF16)",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--profile-kind",
        choices=("all", *PROFILE_KINDS),
        default="all",
        help="Run both measurements (default), or only memory/latency",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--memory-kv-cache-mib",
        type=int,
        default=128,
        help=(
            "Fixed KV cache for the memory run (default: 128 MiB). It must "
            "hold the configured four-request workload."
        ),
    )
    parser.add_argument(
        "--memory-max-model-len",
        type=int,
        default=256,
        help="Short context limit used only by the memory run (default: 256)",
    )
    parser.add_argument("--memory-sample-ms", type=int, default=50)
    parser.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable a worker-side PyTorch trace during the latency run. "
            "Disabled by default for clean wall-clock latency."
        ),
    )
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--_worker-kind", choices=PROFILE_KINDS, help=argparse.SUPPRESS
    )
    parser.add_argument("--_result-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_profile-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.batch_size != BATCH_SIZE:
        parser.error("This profiling workload is fixed to --batch-size 4")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if args.max_model_len < 2:
        parser.error("--max-model-len must be at least 2")
    if not 0 < args.max_tokens < args.max_model_len:
        parser.error("--max-tokens must be positive and smaller than --max-model-len")
    if args.memory_kv_cache_mib <= 0:
        parser.error("--memory-kv-cache-mib must be positive")
    if not 0 < args.max_tokens < args.memory_max_model_len:
        parser.error(
            "--max-tokens must be smaller than --memory-max-model-len so the "
            "memory workload fits"
        )
    if args.memory_sample_ms < 10:
        parser.error("--memory-sample-ms must be at least 10")
    if args._worker and (
        not args.model
        or len(args.model) != 1
        or not args._worker_kind
        or not args._result_file
        or not args._profile_dir
    ):
        parser.error("Internal worker requires one model, result file, and profile dir")
    return args


def discover_models(args: argparse.Namespace) -> list[Path]:
    if args.model:
        models = [path.expanduser().resolve() for path in args.model]
    else:
        root = args.models_dir.expanduser().resolve()
        models = [path for path in root.iterdir() if (path / "config.json").is_file()]
    models = sorted(models, key=lambda path: ("BF16" not in path.name.upper(), path.name))
    missing = [path for path in models if not (path / "config.json").is_file()]
    if missing:
        raise FileNotFoundError(f"Invalid model directories: {missing}")
    return models


def worker(args: argparse.Namespace) -> int:
    model_path = args.model[0].expanduser().resolve()
    result_file = args._result_file.expanduser().resolve()
    profile_dir = args._profile_dir.expanduser().resolve()
    profile_kind = args._worker_kind
    is_memory_run = profile_kind == MEMORY_PROFILE
    max_model_len = (
        args.memory_max_model_len if is_memory_run else args.max_model_len
    )
    kv_cache_memory_bytes = (
        args.memory_kv_cache_mib * 1024 * 1024 if is_memory_run else None
    )
    torch_profiler_enabled = args.profile and not is_memory_run
    metadata = model_metadata(model_path)
    monitor = UnifiedMemoryMonitor(args.memory_sample_ms / 1000)
    monitor.start()
    result: dict[str, Any] = {
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "profile_kind": profile_kind,
        "model": model_path.name,
        "model_path": str(model_path),
        "method": metadata.get("method", "baseline"),
        "status": "FAILED",
        "error": "",
        "settings": {
            "batch_size": BATCH_SIZE,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": max_model_len,
            "max_tokens": args.max_tokens,
            "kv_cache_memory_bytes": kv_cache_memory_bytes,
            "kv_cache_policy": (
                "fixed_minimal" if is_memory_run else "automatic_utilization"
            ),
            "temperature": 0.0,
            "enable_thinking": False,
            "enforce_eager": True,
            "torch_profiler_enabled": torch_profiler_enabled,
            "prompts": PROMPTS,
        },
        "trace_dir": str(profile_dir) if torch_profiler_enabled else "",
        "runtime_overrides": [],
        "worker_memory": {},
        "responses": [],
    }
    profile_started = False

    try:
        import torch
        import vllm
        from vllm import LLM, SamplingParams

        result["runtime"] = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "vllm": getattr(vllm, "__version__", "unknown"),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        }
        monitor.checkpoint("imports_ready")
        profiler_config = None
        if torch_profiler_enabled:
            profile_dir.mkdir(parents=True, exist_ok=True)
            profiler_config = {
                "profiler": "torch",
                "torch_profiler_dir": str(profile_dir),
                "torch_profiler_with_stack": False,
                "torch_profiler_record_shapes": True,
                # This source build otherwise constructs/prints a large CUDA table
                # while stop_profile is servicing a synchronous engine RPC.  The
                # Chrome trace already contains the same CUDA timing information.
                "torch_profiler_dump_cuda_time_total": False,
            }
        hf_overrides, changes = runtime_hf_override(model_path)
        result["runtime_overrides"] = changes
        llm_kwargs: dict[str, Any] = {
            "model": str(model_path),
            "runner": "generate",
            "dtype": "auto",
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": max_model_len,
            "max_num_seqs": BATCH_SIZE,
            "max_num_batched_tokens": max_model_len * 2,
            "enforce_eager": True,
            "language_model_only": True,
            "mm_processor_cache_gb": 0,
            "enable_prefix_caching": False,
            "generation_config": "vllm",
            "disable_log_stats": True,
            "worker_extension_cls": (
                "qwen_batch4_profile.MemoryStatsWorkerExtension"
            ),
        }
        if kv_cache_memory_bytes is not None:
            llm_kwargs["kv_cache_memory_bytes"] = kv_cache_memory_bytes
        if profiler_config is not None:
            llm_kwargs["profiler_config"] = profiler_config
        if hf_overrides is not None:
            llm_kwargs["hf_overrides"] = hf_overrides

        print(f"[LOAD] {model_path}", flush=True)
        load_started = time.perf_counter()
        llm = LLM(**llm_kwargs)
        result["load_seconds"] = round(time.perf_counter() - load_started, 6)
        monitor.checkpoint("after_load")
        result["worker_memory"]["after_load"] = llm.collective_rpc(
            "batch4_memory_stats"
        )
        llm.collective_rpc("batch4_reset_peak_memory_stats")

        messages = [[{"role": "user", "content": prompt}] for prompt in PROMPTS]
        warmup_started = time.perf_counter()
        llm.chat(
            messages,
            sampling_params=SamplingParams(temperature=0.0, max_tokens=1, seed=42),
            use_tqdm=False,
            chat_template_kwargs={"enable_thinking": False},
        )
        result["warmup_seconds"] = round(time.perf_counter() - warmup_started, 6)
        monitor.checkpoint("after_warmup")
        result["worker_memory"]["after_warmup"] = llm.collective_rpc(
            "batch4_memory_stats"
        )
        llm.collective_rpc("batch4_reset_peak_memory_stats")

        if torch_profiler_enabled:
            llm.start_profile(f"batch4_{model_path.name}")
            profile_started = True
        batch_started = time.perf_counter()
        outputs = llm.chat(
            messages,
            sampling_params=SamplingParams(
                temperature=0.0,
                max_tokens=args.max_tokens,
                seed=42,
            ),
            use_tqdm=False,
            chat_template_kwargs={"enable_thinking": False},
        )
        batch_latency = time.perf_counter() - batch_started
        monitor.checkpoint("after_workload")
        result["worker_memory"]["after_workload"] = llm.collective_rpc(
            "batch4_memory_stats"
        )
        if torch_profiler_enabled:
            llm.stop_profile()
            profile_started = False
            time.sleep(5)

        prompt_tokens = output_tokens = 0
        for index, (prompt, output) in enumerate(zip(PROMPTS, outputs, strict=True)):
            completion = output.outputs[0]
            prompt_count = len(output.prompt_token_ids)
            output_count = len(completion.token_ids)
            prompt_tokens += prompt_count
            output_tokens += output_count
            result["responses"].append(
                {
                    "index": index,
                    "prompt": prompt,
                    "response": completion.text,
                    "prompt_tokens": prompt_count,
                    "output_tokens": output_count,
                    "finish_reason": completion.finish_reason,
                }
            )
        result["performance"] = {
            "batch_latency_seconds": round(batch_latency, 6),
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "output_tokens_per_second": round(
                output_tokens / batch_latency if batch_latency else 0.0, 6
            ),
        }
        result["trace_files"] = (
            sorted(
                str(path.relative_to(profile_dir))
                for path in profile_dir.rglob("*")
                if path.is_file()
            )
            if torch_profiler_enabled
            else []
        )
        result["status"] = "SUCCESS"
        print(
            f"[{profile_kind.upper()}] batch=4, "
            f"{batch_latency:.3f}s, {output_tokens} tokens, "
            f"{result['performance']['output_tokens_per_second']:.2f} tok/s",
            flush=True,
        )
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
        print(result["traceback"], file=sys.stderr, flush=True)
    finally:
        if profile_started:
            try:
                llm.stop_profile()
                time.sleep(5)
            except BaseException:
                pass
        monitor.stop()
        result["unified_memory"] = monitor.result()
        result["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        atomic_write_json(result_file, result)
        print(f"[RESULT] {result_file}", flush=True)
    return 0 if result["status"] == "SUCCESS" else 1


def worker_command(
    args: argparse.Namespace,
    profile_kind: str,
    model: Path,
    result_file: Path,
    profile_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker",
        "--_worker-kind",
        profile_kind,
        "--model",
        str(model),
        "--_result-file",
        str(result_file),
        "--_profile-dir",
        str(profile_dir),
        "--batch-size",
        str(BATCH_SIZE),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--max-tokens",
        str(args.max_tokens),
        "--memory-kv-cache-mib",
        str(args.memory_kv_cache_mib),
        "--memory-max-model-len",
        str(args.memory_max_model_len),
        "--memory-sample-ms",
        str(args.memory_sample_ms),
        "--profile" if args.profile else "--no-profile",
    ]


def stop_worker_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def worker_stat_mib(
    result: dict[str, Any], checkpoint: str, field: str
) -> float | None:
    stats = result.get("worker_memory", {}).get(checkpoint, [])
    if not isinstance(stats, list) or not stats:
        return None
    values = [stat.get(field) for stat in stats if isinstance(stat, dict)]
    if not values or any(not isinstance(value, int) for value in values):
        return None
    return round(sum(values) / (1024 * 1024), 3)


def memory_summary_row(
    result: dict[str, Any], result_file: Path, log_file: Path
) -> dict[str, Any]:
    settings = result.get("settings", {})
    performance = result.get("performance", {})
    memory = result.get("unified_memory", {})
    runtime = result.get("runtime", {})
    return {
        "profile_kind": result.get("profile_kind", MEMORY_PROFILE),
        "model": result.get("model", result_file.stem),
        "method": result.get("method", ""),
        "status": result.get("status", "FAILED"),
        "error": result.get("error", ""),
        "batch_size": settings.get("batch_size", ""),
        "load_seconds": result.get("load_seconds", ""),
        "warmup_seconds": result.get("warmup_seconds", ""),
        "memory_workload_seconds": performance.get("batch_latency_seconds", ""),
        "model_memory_usage_mib": worker_stat_mib(
            result, "after_load", "model_memory_usage_bytes"
        )
        or "",
        "cuda_after_load_allocated_mib": worker_stat_mib(
            result, "after_load", "cuda_memory_allocated_bytes"
        )
        or "",
        "cuda_after_load_reserved_mib": worker_stat_mib(
            result, "after_load", "cuda_memory_reserved_bytes"
        )
        or "",
        "cuda_after_load_peak_allocated_mib": worker_stat_mib(
            result, "after_load", "cuda_max_memory_allocated_bytes"
        )
        or "",
        "cuda_workload_peak_allocated_mib": worker_stat_mib(
            result, "after_workload", "cuda_max_memory_allocated_bytes"
        )
        or "",
        "ram_peak_mib": memory.get("peak_mib", ""),
        "ram_peak_delta_mib": memory.get("peak_delta_mib", ""),
        "kv_cache_memory_mib": round(
            settings.get("kv_cache_memory_bytes", 0) / (1024 * 1024), 3
        )
        if settings.get("kv_cache_memory_bytes")
        else "",
        "max_model_len": settings.get("max_model_len", ""),
        "max_tokens": settings.get("max_tokens", ""),
        "vllm_version": runtime.get("vllm", ""),
        "result_file": str(result_file),
        "log_file": str(log_file),
    }


def latency_summary_row(
    result: dict[str, Any], result_file: Path, log_file: Path
) -> dict[str, Any]:
    settings = result.get("settings", {})
    performance = result.get("performance", {})
    runtime = result.get("runtime", {})
    return {
        "profile_kind": result.get("profile_kind", LATENCY_PROFILE),
        "model": result.get("model", result_file.stem),
        "method": result.get("method", ""),
        "status": result.get("status", "FAILED"),
        "error": result.get("error", ""),
        "batch_size": settings.get("batch_size", ""),
        "batch_latency_seconds": performance.get("batch_latency_seconds", ""),
        "prompt_tokens": performance.get("prompt_tokens", ""),
        "output_tokens": performance.get("output_tokens", ""),
        "output_tokens_per_second": performance.get("output_tokens_per_second", ""),
        "load_seconds": result.get("load_seconds", ""),
        "warmup_seconds": result.get("warmup_seconds", ""),
        "gpu_memory_utilization": settings.get("gpu_memory_utilization", ""),
        "max_model_len": settings.get("max_model_len", ""),
        "max_tokens": settings.get("max_tokens", ""),
        "torch_profiler_enabled": settings.get("torch_profiler_enabled", ""),
        "vllm_version": runtime.get("vllm", ""),
        "trace_dir": result.get("trace_dir", ""),
        "result_file": str(result_file),
        "log_file": str(log_file),
    }


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def baseline_result(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next(
        (
            result
            for result in results
            if result.get("status") == "SUCCESS"
            and (
                result.get("method") == "baseline"
                or "BF16" in str(result.get("model", "")).upper()
            )
        ),
        None,
    )


def memory_comparison_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = baseline_result(results)
    if baseline is None:
        return []
    base_model = worker_stat_mib(
        baseline, "after_load", "model_memory_usage_bytes"
    )
    base_cuda = worker_stat_mib(
        baseline, "after_load", "cuda_memory_allocated_bytes"
    )
    base_ram = baseline.get("unified_memory", {}).get("peak_delta_mib")
    rows: list[dict[str, Any]] = []
    for result in results:
        model_memory = worker_stat_mib(
            result, "after_load", "model_memory_usage_bytes"
        )
        cuda_memory = worker_stat_mib(
            result, "after_load", "cuda_memory_allocated_bytes"
        )
        workload_peak = worker_stat_mib(
            result, "after_workload", "cuda_max_memory_allocated_bytes"
        )
        ram_delta = result.get("unified_memory", {}).get("peak_delta_mib")
        model_reduction = (
            round(base_model - model_memory, 3)
            if base_model is not None and model_memory is not None
            else None
        )
        rows.append(
            {
                "model": result.get("model", ""),
                "method": result.get("method", ""),
                "status": result.get("status", "FAILED"),
                "model_memory_usage_mib": model_memory or "",
                "bf16_model_memory_reduction_mib": model_reduction
                if model_reduction is not None
                else "",
                "bf16_model_memory_reduction_percent": round(
                    100 * model_reduction / base_model, 3
                )
                if model_reduction is not None and base_model
                else "",
                "cuda_after_load_allocated_mib": cuda_memory or "",
                "bf16_cuda_after_load_reduction_mib": round(
                    base_cuda - cuda_memory, 3
                )
                if base_cuda is not None and cuda_memory is not None
                else "",
                "cuda_workload_peak_allocated_mib": workload_peak or "",
                "ram_peak_delta_mib": ram_delta if ram_delta is not None else "",
                "bf16_ram_peak_delta_reduction_mib": round(
                    base_ram - ram_delta, 3
                )
                if isinstance(base_ram, (int, float))
                and isinstance(ram_delta, (int, float))
                else "",
            }
        )
    return rows


def latency_comparison_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = baseline_result(results)
    if baseline is None:
        return []
    base_perf = baseline["performance"]
    rows: list[dict[str, Any]] = []
    for result in results:
        perf = result.get("performance", {})
        latency = perf.get("batch_latency_seconds")
        throughput = perf.get("output_tokens_per_second")
        rows.append(
            {
                "model": result.get("model", ""),
                "method": result.get("method", ""),
                "status": result.get("status", "FAILED"),
                "batch_latency_seconds": latency or "",
                "bf16_latency_speedup": round(
                    base_perf["batch_latency_seconds"] / latency, 4
                )
                if latency
                else "",
                "output_tokens_per_second": throughput or "",
                "bf16_throughput_ratio": round(
                    throughput / base_perf["output_tokens_per_second"], 4
                )
                if throughput
                else "",
                "output_tokens": perf.get("output_tokens", ""),
                "trace_dir": result.get("trace_dir", ""),
            }
        )
    return rows


def driver(args: argparse.Namespace) -> int:
    models = discover_models(args)
    if not models:
        print("No models found", file=sys.stderr)
        return 2
    run_dir = args.results_dir.expanduser().resolve() / datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    profile_kinds = (
        list(PROFILE_KINDS) if args.profile_kind == "all" else [args.profile_kind]
    )
    print(f"Results: {run_dir}")
    print("Models: " + ", ".join(model.name for model in models))
    print("Measurements: " + ", ".join(profile_kinds))
    print(
        f"Memory run: fixed KV={args.memory_kv_cache_mib} MiB, "
        f"max_model_len={args.memory_max_model_len}"
    )
    print(
        "Latency run: "
        + ("with PyTorch trace" if args.profile else "clean wall-clock timing")
    )

    all_results: dict[str, list[dict[str, Any]]] = {}
    completed_results: list[dict[str, Any]] = []
    interrupted = False
    for profile_kind in profile_kinds:
        phase_dir = run_dir / profile_kind
        phase_dir.mkdir(parents=True, exist_ok=False)
        results: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        all_results[profile_kind] = results
        print(f"\n=== {profile_kind.upper()} ===", flush=True)

        for index, model in enumerate(models, 1):
            result_file = phase_dir / f"{model.name}.json"
            log_file = phase_dir / f"{model.name}.log"
            profile_dir = phase_dir / "traces" / model.name
            print(f"[{index}/{len(models)}] {model.name}", flush=True)
            environment = os.environ.copy()
            environment.setdefault("HF_HUB_OFFLINE", "1")
            environment.setdefault("TRANSFORMERS_OFFLINE", "1")
            environment.setdefault("TOKENIZERS_PARALLELISM", "false")
            with log_file.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    worker_command(
                        args, profile_kind, model, result_file, profile_dir
                    ),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    start_new_session=True,
                )
                try:
                    return_code = process.wait()
                except KeyboardInterrupt:
                    interrupted = True
                    stop_worker_group(process)
                    return_code = (
                        process.returncode
                        if process.returncode is not None
                        else 130
                    )
            if result_file.is_file():
                result = json.loads(result_file.read_text(encoding="utf-8"))
            else:
                result = {
                    "profile_kind": profile_kind,
                    "model": model.name,
                    "method": model_metadata(model).get("method", "baseline"),
                    "status": "FAILED",
                    "error": f"worker exited with {return_code}; inspect log",
                    "trace_dir": str(profile_dir)
                    if profile_kind == LATENCY_PROFILE and args.profile
                    else "",
                }
                atomic_write_json(result_file, result)
            results.append(result)
            completed_results.append(result)

            if profile_kind == MEMORY_PROFILE:
                rows.append(memory_summary_row(result, result_file, log_file))
                write_csv(
                    phase_dir / "summary.csv", MEMORY_SUMMARY_FIELDS, rows
                )
                comparisons = memory_comparison_rows(results)
                if comparisons:
                    write_csv(
                        phase_dir / "comparison_with_bf16.csv",
                        MEMORY_COMPARISON_FIELDS,
                        comparisons,
                    )
                model_mib = worker_stat_mib(
                    result, "after_load", "model_memory_usage_bytes"
                )
                print(
                    f"{result['status']}: model_memory="
                    f"{model_mib if model_mib is not None else '-'} MiB, "
                    f"RAM peak delta="
                    f"{result.get('unified_memory', {}).get('peak_delta_mib', '-')} MiB; "
                    f"log={log_file.name}",
                    flush=True,
                )
            else:
                rows.append(latency_summary_row(result, result_file, log_file))
                write_csv(
                    phase_dir / "summary.csv", LATENCY_SUMMARY_FIELDS, rows
                )
                comparisons = latency_comparison_rows(results)
                if comparisons:
                    write_csv(
                        phase_dir / "comparison_with_bf16.csv",
                        LATENCY_COMPARISON_FIELDS,
                        comparisons,
                    )
                performance = result.get("performance", {})
                print(
                    f"{result['status']}: latency="
                    f"{performance.get('batch_latency_seconds', '-')}s, "
                    f"tok/s={performance.get('output_tokens_per_second', '-')}; "
                    f"log={log_file.name}",
                    flush=True,
                )

            atomic_write_json(phase_dir / "summary.json", results)
            atomic_write_json(run_dir / "summary.json", all_results)
            if interrupted:
                break

        if interrupted:
            break

    successes = sum(
        result.get("status") == "SUCCESS" for result in completed_results
    )
    expected = len(models) * len(profile_kinds)
    print(f"\nCompleted: {successes}/{expected} successful")
    for profile_kind in all_results:
        print(f"{profile_kind.capitalize()} summary: {run_dir / profile_kind / 'summary.csv'}")
    return 0 if successes == expected and not interrupted else 1


def main() -> None:
    args = parse_args()
    raise SystemExit(worker(args) if args._worker else driver(args))


if __name__ == "__main__":
    main()
