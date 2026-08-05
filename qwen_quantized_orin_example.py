"""Jetson Orin에서 Qwen3-8B 양자화 모델들의 vLLM smoke/latency를 측정한다.

기본 실행은 ``quantized/`` 아래의 양자화 모델만 하나씩 별도 프로세스에서
실행한다. 결과는 ``orin_quant_results/<실행시각>/``에 JSON, CSV, log로 남긴다.
Orin의 GPU/CPU unified memory는 /proc/meminfo의 MemAvailable을 샘플링하여
모델 로드부터 추론 종료까지의 시스템 RAM peak와 시작 대비 증가량을 기록한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_MODELS_DIR = Path(__file__).resolve().parent / "quantized"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "orin_quant_results"
DEFAULT_PROMPT = "양자화된 모델이 정상 동작하는지 확인 중입니다. 한 문장으로 답해주세요."

CSV_FIELDS = [
    "model",
    "method",
    "status",
    "error",
    "load_seconds",
    "warmup_seconds",
    "latency_mean_seconds",
    "latency_p50_seconds",
    "latency_min_seconds",
    "latency_max_seconds",
    "output_tokens_mean",
    "output_tokens_per_second_mean",
    "ram_start_mib",
    "ram_peak_mib",
    "ram_peak_delta_mib",
    "ram_total_mib",
    "gpu_memory_utilization",
    "max_model_len",
    "max_tokens",
    "repeats",
    "vllm_version",
    "torch_version",
    "runtime_overrides",
    "result_file",
    "log_file",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="양자화 모델들이 있는 디렉터리 (기본값: %(default)s)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        action="append",
        help="실행할 모델 디렉터리. 여러 번 지정 가능; 생략하면 양자화 모델 전체 실행",
    )
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help="BF16/baseline 디렉터리도 자동 탐색 대상에 포함",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="결과 root 디렉터리 (기본값: %(default)s)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.70,
        help="vLLM GPU memory utilization (기본값: 0.70)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=1024,
        help="최대 context 길이 (기본값: 1024)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="측정 요청당 생성 토큰 수 (기본값: 32)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="warm-up 뒤 latency 측정 횟수 (기본값: 3)",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="테스트 프롬프트")
    parser.add_argument(
        "--memory-sample-ms",
        type=int,
        default=100,
        help="unified RAM 샘플링 간격 ms (기본값: 100)",
    )
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_result-file", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization은 0보다 크고 1 이하여야 합니다")
    if args.max_model_len < 2:
        parser.error("--max-model-len은 2 이상이어야 합니다")
    if not 0 < args.max_tokens < args.max_model_len:
        parser.error("--max-tokens는 양수이며 --max-model-len보다 작아야 합니다")
    if args.repeats < 1:
        parser.error("--repeats는 1 이상이어야 합니다")
    if args.memory_sample_ms < 10:
        parser.error("--memory-sample-ms는 10 이상이어야 합니다")
    if args._worker and (not args.model or len(args.model) != 1 or not args._result_file):
        parser.error("내부 worker에는 모델 하나와 결과 파일이 필요합니다")
    return args


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
    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = interval_seconds
        self.start_mib, self.total_mib = read_ram_mib()
        self.peak_mib = self.start_mib
        self.end_mib = self.start_mib
        self.samples = 1
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
        self.end_mib = used_mib
        self.peak_mib = max(self.peak_mib, used_mib)
        self.samples += 1

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def result(self) -> dict[str, float | int]:
        return {
            "measurement": "system RAM used = MemTotal - MemAvailable (unified memory)",
            "start_mib": round(self.start_mib, 3),
            "peak_mib": round(self.peak_mib, 3),
            "peak_delta_mib": round(max(0.0, self.peak_mib - self.start_mib), 3),
            "end_mib": round(self.end_mib, 3),
            "total_mib": round(self.total_mib, 3),
            "samples": self.samples,
            "interval_seconds": self.interval_seconds,
        }


def atomic_write_json(path: Path, value: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def model_metadata(model_path: Path) -> dict[str, Any]:
    path = model_path / "quantization_metadata.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def runtime_hf_override(model_path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    """W8A8의 runtime-irrelevant actorder만 메모리 안에서 제거한다."""
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


def percentile_50(values: list[float]) -> float:
    return statistics.median(values)


def worker(args: argparse.Namespace) -> int:
    model_path = args.model[0].expanduser().resolve()
    result_file = args._result_file.expanduser().resolve()
    metadata = model_metadata(model_path)
    monitor = UnifiedMemoryMonitor(args.memory_sample_ms / 1000)
    monitor.start()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    result: dict[str, Any] = {
        "started_at": started_at,
        "model": model_path.name,
        "model_path": str(model_path),
        "method": metadata.get("method", "baseline"),
        "status": "FAILED",
        "error": "",
        "settings": {
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
            "repeats": args.repeats,
            "prompt": args.prompt,
            "temperature": 0.0,
            "ignore_eos": True,
            "enforce_eager": True,
        },
        "requests": [],
    }

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
        hf_overrides, changes = runtime_hf_override(model_path)
        result["runtime_overrides"] = changes
        llm_kwargs: dict[str, Any] = {
            "model": str(model_path),
            "runner": "generate",
            "dtype": "auto",
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_num_seqs": 1,
            "max_num_batched_tokens": args.max_model_len,
            "enforce_eager": True,
            "language_model_only": True,
            "mm_processor_cache_gb": 0,
            "enable_prefix_caching": False,
            "generation_config": "vllm",
            "disable_log_stats": True,
        }
        if hf_overrides is not None:
            llm_kwargs["hf_overrides"] = hf_overrides

        print(f"[LOAD] {model_path}", flush=True)
        load_started = time.perf_counter()
        llm = LLM(**llm_kwargs)
        result["load_seconds"] = round(time.perf_counter() - load_started, 6)

        messages = [{"role": "user", "content": args.prompt}]
        chat_kwargs = {
            "messages": messages,
            "use_tqdm": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        warmup_params = SamplingParams(temperature=0.0, max_tokens=8, seed=42)
        warmup_started = time.perf_counter()
        llm.chat(sampling_params=warmup_params, **chat_kwargs)
        result["warmup_seconds"] = round(time.perf_counter() - warmup_started, 6)

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            ignore_eos=True,
            seed=42,
        )
        latencies: list[float] = []
        throughputs: list[float] = []
        output_token_counts: list[int] = []
        for index in range(1, args.repeats + 1):
            request_started = time.perf_counter()
            output = llm.chat(sampling_params=sampling_params, **chat_kwargs)[0]
            latency = time.perf_counter() - request_started
            completion = output.outputs[0]
            output_tokens = len(completion.token_ids)
            throughput = output_tokens / latency if latency else 0.0
            request = {
                "index": index,
                "latency_seconds": round(latency, 6),
                "prompt_tokens": len(output.prompt_token_ids),
                "output_tokens": output_tokens,
                "output_tokens_per_second": round(throughput, 6),
                "text": completion.text.strip(),
            }
            result["requests"].append(request)
            latencies.append(latency)
            throughputs.append(throughput)
            output_token_counts.append(output_tokens)
            print(
                f"[RUN {index}/{args.repeats}] {latency:.3f}s, "
                f"{output_tokens} tokens, {throughput:.2f} tok/s",
                flush=True,
            )

        result["latency"] = {
            "mean_seconds": round(statistics.mean(latencies), 6),
            "p50_seconds": round(percentile_50(latencies), 6),
            "min_seconds": round(min(latencies), 6),
            "max_seconds": round(max(latencies), 6),
        }
        result["output_tokens_mean"] = round(statistics.mean(output_token_counts), 3)
        result["output_tokens_per_second_mean"] = round(statistics.mean(throughputs), 6)
        result["status"] = "SUCCESS"
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
        print(result["traceback"], file=sys.stderr, flush=True)
    finally:
        monitor.stop()
        result["unified_memory"] = monitor.result()
        result["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        atomic_write_json(result_file, result)
        print(f"[RESULT] {result_file}", flush=True)
    return 0 if result["status"] == "SUCCESS" else 1


def discover_models(args: argparse.Namespace) -> list[Path]:
    if args.model:
        candidates = [path.expanduser().resolve() for path in args.model]
    else:
        root = args.models_dir.expanduser().resolve()
        candidates = sorted(path for path in root.iterdir() if path.is_dir()) if root.is_dir() else []

    models: list[Path] = []
    for path in candidates:
        config_path = path / "config.json"
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        is_quantized = isinstance(config.get("quantization_config"), dict)
        if is_quantized or args.include_baseline or args.model:
            models.append(path)
    return models


def worker_command(args: argparse.Namespace, model: Path, result_file: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker",
        "--model",
        str(model),
        "--_result-file",
        str(result_file),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--max-tokens",
        str(args.max_tokens),
        "--repeats",
        str(args.repeats),
        "--prompt",
        args.prompt,
        "--memory-sample-ms",
        str(args.memory_sample_ms),
    ]


def stop_worker_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def csv_row(result: dict[str, Any], result_file: Path, log_file: Path) -> dict[str, Any]:
    settings = result.get("settings", {})
    memory = result.get("unified_memory", {})
    latency = result.get("latency", {})
    runtime = result.get("runtime", {})
    return {
        "model": result.get("model", result_file.stem),
        "method": result.get("method", ""),
        "status": result.get("status", "FAILED"),
        "error": result.get("error", ""),
        "load_seconds": result.get("load_seconds", ""),
        "warmup_seconds": result.get("warmup_seconds", ""),
        "latency_mean_seconds": latency.get("mean_seconds", ""),
        "latency_p50_seconds": latency.get("p50_seconds", ""),
        "latency_min_seconds": latency.get("min_seconds", ""),
        "latency_max_seconds": latency.get("max_seconds", ""),
        "output_tokens_mean": result.get("output_tokens_mean", ""),
        "output_tokens_per_second_mean": result.get("output_tokens_per_second_mean", ""),
        "ram_start_mib": memory.get("start_mib", ""),
        "ram_peak_mib": memory.get("peak_mib", ""),
        "ram_peak_delta_mib": memory.get("peak_delta_mib", ""),
        "ram_total_mib": memory.get("total_mib", ""),
        "gpu_memory_utilization": settings.get("gpu_memory_utilization", ""),
        "max_model_len": settings.get("max_model_len", ""),
        "max_tokens": settings.get("max_tokens", ""),
        "repeats": settings.get("repeats", ""),
        "vllm_version": runtime.get("vllm", ""),
        "torch_version": runtime.get("torch", ""),
        "runtime_overrides": " | ".join(result.get("runtime_overrides", [])),
        "result_file": str(result_file),
        "log_file": str(log_file),
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def driver(args: argparse.Namespace) -> int:
    models = discover_models(args)
    if not models:
        print("실행할 모델을 찾지 못했습니다.", file=sys.stderr)
        return 2

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.results_dir.expanduser().resolve() / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"결과 디렉터리: {run_dir}")
    print("실행 모델: " + ", ".join(path.name for path in models))

    rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    interrupted = False
    for index, model in enumerate(models, 1):
        result_file = run_dir / f"{model.name}.json"
        log_file = run_dir / f"{model.name}.log"
        print(f"\n[{index}/{len(models)}] {model.name}", flush=True)
        environment = os.environ.copy()
        environment.setdefault("HF_HUB_OFFLINE", "1")
        environment.setdefault("TRANSFORMERS_OFFLINE", "1")
        environment.setdefault("TOKENIZERS_PARALLELISM", "false")
        with log_file.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                worker_command(args, model, result_file),
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
                return_code = process.returncode if process.returncode is not None else 130

        if result_file.is_file():
            result = json.loads(result_file.read_text(encoding="utf-8"))
        else:
            result = {
                "model": model.name,
                "model_path": str(model),
                "method": model_metadata(model).get("method", ""),
                "status": "FAILED",
                "error": f"worker exited with code {return_code}; inspect log (OOM kill 가능)",
                "settings": {
                    "gpu_memory_utilization": args.gpu_memory_utilization,
                    "max_model_len": args.max_model_len,
                    "max_tokens": args.max_tokens,
                    "repeats": args.repeats,
                },
            }
            atomic_write_json(result_file, result)
        results.append(result)
        rows.append(csv_row(result, result_file, log_file))
        write_summary_csv(run_dir / "summary.csv", rows)
        atomic_write_json(run_dir / "summary.json", results)
        print(
            f"{result['status']}: {result.get('error') or 'latency/peak RAM 저장 완료'} "
            f"(log: {log_file.name})",
            flush=True,
        )
        if interrupted:
            break

    successes = sum(result.get("status") == "SUCCESS" for result in results)
    print(f"\n완료: SUCCESS {successes}/{len(results)}")
    print(f"요약 CSV: {run_dir / 'summary.csv'}")
    print(f"상세 JSON: {run_dir / 'summary.json'}")
    return 0 if successes == len(models) and not interrupted else 1


def main() -> None:
    args = parse_args()
    raise SystemExit(worker(args) if args._worker else driver(args))


if __name__ == "__main__":
    main()
