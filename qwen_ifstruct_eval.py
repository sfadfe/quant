"""Evaluate Qwen3-8B BF16/quantized checkpoints on pre-sampled IFStruct JSONL.

The driver evaluates every row in the JSONL supplied with ``--dataset`` and
launches one model per subprocess so device memory is returned between
checkpoints.  The default batch size is four on Jetson and eight elsewhere.
Results
include strict structured-output accuracy, per-sample responses/errors,
latency, token counts, and peak unified or host memory, followed by paired
comparisons against BF16.

RTX 3090 example (PyYAML and vLLM must already be installed)::

    CUDA_VISIBLE_DEVICES=0 python3 qwen_ifstruct_eval.py \
      --models-dir /workspace/quantized \
      --baseline-model /path/to/Qwen3-8B-snapshot \
      --dataset /workspace/quantized/ifstruct_sample_100.jsonl

The automatic defaults are gpu_memory_utilization=0.60 plus eager execution on
Jetson, and gpu_memory_utilization=0.90 plus CUDA graphs on other CUDA systems.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = ROOT / "quantized"
DEFAULT_RESULTS_DIR = ROOT / "ifstruct_results"
IFSTRUCT_COMMIT = "81dbaf26eddf20a0e36038a0a5b139bad4765eda"
IFSTRUCT_URL = (
    "https://raw.githubusercontent.com/Liquid4All/ifstruct/"
    f"{IFSTRUCT_COMMIT}/data/test.jsonl"
)
IFSTRUCT_SHA256 = "2a62e39293040aed7db6690bfa31c764cb761f71daf0ae0bd4abc4249f5f574e"
FROZEN_SOURCE_DATASET = (
    ROOT / ".cache" / "ifstruct" / f"test-{IFSTRUCT_COMMIT}.jsonl"
)
DEFAULT_DATASET = DEFAULT_MODELS_DIR / "ifstruct_sample_100.jsonl"
SAMPLE_500_DATASET = DEFAULT_MODELS_DIR / "ifstruct_sample_500.jsonl"
SAMPLE_SEED = 42

SUMMARY_FIELDS = [
    "model",
    "method",
    "status",
    "error",
    "passed",
    "total",
    "pass_rate",
    "json_pass_rate",
    "yaml_pass_rate",
    "truncated",
    "generation_seconds",
    "batch_latency_mean_seconds",
    "prompt_tokens",
    "output_tokens",
    "output_tokens_per_second",
    "load_seconds",
    "warmup_seconds",
    "ram_peak_mib",
    "ram_peak_delta_mib",
    "gpu_memory_utilization",
    "batch_size",
    "tokenizer",
    "max_model_len",
    "max_tokens",
    "vllm_version",
    "result_file",
    "log_file",
]

COMPARISON_FIELDS = [
    "model",
    "method",
    "status",
    "passed",
    "total",
    "pass_rate",
    "bf16_delta_points",
    "paired_regressions",
    "paired_recoveries",
    "paired_agreement",
    "batch_latency_mean_seconds",
    "bf16_latency_speedup",
    "output_tokens_per_second",
    "bf16_throughput_ratio",
    "ram_peak_mib",
    "ram_peak_delta_mib",
]


class _SafeLoaderNoDate(yaml.SafeLoader):
    pass


_SafeLoaderNoDate.yaml_implicit_resolvers = {
    key: [
        (tag, regexp)
        for tag, regexp in values
        if tag != "tag:yaml.org,2002:timestamp"
    ]
    for key, values in yaml.SafeLoader.yaml_implicit_resolvers.copy().items()
}


@dataclass
class ValidationResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument(
        "--model",
        type=Path,
        action="append",
        help="Model directory; repeat to run selected models (default: all including BF16)",
    )
    parser.add_argument(
        "--baseline-model",
        type=Path,
        help="Optional BF16 model directory outside --models-dir (for example an HF snapshot)",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help=(
            "Shared local tokenizer directory. By default the driver uses the BF16 "
            "checkpoint tokenizer when it can find one."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Pre-sampled IFStruct JSONL; every row is evaluated (default: %(default)s)",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--prepare-datasets",
        action="store_true",
        help="Create the fixed 100/500-row IFStruct JSONL files under quantized/ and exit",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Generation batch size (default: Orin 4, other systems 8)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        help="vLLM memory fraction (default: Orin 0.60, other CUDA systems 0.90)",
    )
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Disable CUDA graphs/compile (default: enabled on Orin, disabled elsewhere)",
    )
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--memory-sample-ms", type=int, default=100)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show model/batch progress in the driver terminal (default: enabled)",
    )
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_result-file", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.gpu_memory_utilization is not None and not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.max_model_len < 2:
        parser.error("--max-model-len must be at least 2")
    if not 0 < args.max_tokens < args.max_model_len:
        parser.error("--max-tokens must be positive and smaller than --max-model-len")
    if args.memory_sample_ms < 10:
        parser.error("--memory-sample-ms must be at least 10")
    if args._worker and (not args.model or len(args.model) != 1 or not args._result_file):
        parser.error("Internal worker requires exactly one --model and --_result-file")
    jetson = is_jetson()
    if args.batch_size is None:
        args.batch_size = 4 if jetson else 8
    if args.gpu_memory_utilization is None:
        args.gpu_memory_utilization = 0.60 if jetson else 0.90
    if args.enforce_eager is None:
        args.enforce_eager = jetson
    return args


def is_jetson() -> bool:
    if Path("/etc/nv_tegra_release").is_file():
        return True
    model_file = Path("/proc/device-tree/model")
    if not model_file.is_file():
        return False
    try:
        model = model_file.read_bytes().replace(b"\x00", b"").decode(errors="ignore")
    except OSError:
        return False
    return "jetson" in model.lower() or "orin" in model.lower()


def read_system_ram_mib() -> tuple[float, float]:
    values: dict[str, int] = {}
    with Path("/proc/meminfo").open(encoding="utf-8") as file:
        for line in file:
            key, value = line.split(":", 1)
            values[key] = int(value.strip().split()[0])
    total_kib = values["MemTotal"]
    used_kib = total_kib - values["MemAvailable"]
    return used_kib / 1024, total_kib / 1024


class SystemMemoryMonitor:
    """Track host RAM; on Jetson this is also CPU/GPU unified memory."""

    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = interval_seconds
        self.start_mib, self.total_mib = read_system_ram_mib()
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
        used_mib, _ = read_system_ram_mib()
        self.end_mib = used_mib
        self.peak_mib = max(self.peak_mib, used_mib)
        self.samples += 1

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def result(self) -> dict[str, Any]:
        return {
            "measurement": (
                "Jetson unified RAM used = MemTotal - MemAvailable"
                if is_jetson()
                else "host RAM used = MemTotal - MemAvailable (not discrete GPU VRAM)"
            ),
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
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def model_metadata(model_path: Path) -> dict[str, Any]:
    metadata_path = model_path / "quantization_metadata.json"
    if not metadata_path.is_file():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


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
    for group_name, group in groups.items() if isinstance(groups, dict) else ():
        if not isinstance(group, dict):
            continue
        weights = group.get("weights")
        activations = group.get("input_activations")
        if not isinstance(weights, dict) or not isinstance(activations, dict):
            continue
        is_dense_w8a8 = (
            weights.get("num_bits") == 8
            and weights.get("strategy") == "channel"
            and activations.get("num_bits") == 8
            and activations.get("strategy") == "token"
            and activations.get("dynamic") is True
        )
        if is_dense_w8a8 and weights.get("actorder") in ("static", "weight"):
            old = weights["actorder"]
            weights["actorder"] = None
            changes.append(
                f"quantization_config.config_groups.{group_name}.weights.actorder: "
                f"{old} -> null"
            )
    return ({"quantization_config": quantization}, changes) if changes else (None, [])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_frozen_source_dataset() -> Path:
    path = FROZEN_SOURCE_DATASET.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        temporary = path.with_suffix(path.suffix + ".download")
        print(f"Downloading frozen IFStruct test set: {IFSTRUCT_URL}", flush=True)
        with urllib.request.urlopen(IFSTRUCT_URL, timeout=60) as response:
            temporary.write_bytes(response.read())
        if sha256_file(temporary) != IFSTRUCT_SHA256:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("Downloaded IFStruct checksum mismatch")
        os.replace(temporary, path)
    if sha256_file(path) != IFSTRUCT_SHA256:
        raise RuntimeError(f"Frozen IFStruct checksum mismatch: {path}")
    return path


def ensure_dataset(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        hint = " (run --prepare-datasets first)" if path == DEFAULT_DATASET.resolve() else ""
        raise FileNotFoundError(f"Dataset not found: {path}{hint}")
    return path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    required = {
        "seed",
        "entity_type",
        "prompt",
        "json_schema",
        "top_level_count",
        "top_level_key",
        "require_wrapper_key",
        "require_code_block",
        "require_no_commentary",
        "output_format",
    }
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Dataset row {index} is missing: {sorted(missing)}")
    return rows


def stratified_sample(
    rows: list[dict[str, Any]], number: int, sample_seed: int
) -> list[dict[str, Any]]:
    """Balance JSON/YAML, then round-robin over entity types within each."""
    rng = random.Random(sample_seed)
    targets = {"json": number // 2 + number % 2, "yaml": number // 2}
    selected: list[dict[str, Any]] = []
    for output_format in ("json", "yaml"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["output_format"] == output_format:
                groups[row["entity_type"]].append(row)
        for group in groups.values():
            rng.shuffle(group)
        entity_types = sorted(groups)
        rng.shuffle(entity_types)
        format_selected: list[dict[str, Any]] = []
        round_index = 0
        while len(format_selected) < targets[output_format]:
            added = False
            for entity_type in entity_types:
                group = groups[entity_type]
                if round_index < len(group):
                    format_selected.append(group[round_index])
                    added = True
                    if len(format_selected) == targets[output_format]:
                        break
            if not added:
                break
            round_index += 1
        selected.extend(format_selected)
    if len(selected) != number:
        raise ValueError(f"Could only select {len(selected)} of {number} requested rows")
    rng.shuffle(selected)
    return selected


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def prepare_sample_datasets() -> int:
    source = ensure_frozen_source_dataset()
    rows = load_jsonl(source)
    outputs = ((100, DEFAULT_DATASET), (500, SAMPLE_500_DATASET))
    for number, output in outputs:
        sampled = stratified_sample(rows, number, SAMPLE_SEED)
        atomic_write_jsonl(output, sampled)
        print(
            f"Wrote {len(sampled)} rows: {output.resolve()} "
            f"(sha256={sha256_file(output)})"
        )
    return 0


def sample_manifest(
    dataset: Path, examples: list[dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    def counts(key: str) -> dict[str, int]:
        return dict(sorted(Counter(str(row[key]) for row in examples).items()))

    return {
        "dataset": str(dataset),
        "dataset_sha256": sha256_file(dataset),
        "ifstruct_commit": IFSTRUCT_COMMIT,
        "selection": "all rows from the supplied pre-sampled JSONL, in file order",
        "bundled_sample_provenance": (
            "seed 42; JSON/YAML balanced; entity-type round-robin; deterministic shuffle"
        ),
        "num_samples": len(examples),
        "batch_size": args.batch_size,
        "tokenizer": str(args.tokenizer) if args.tokenizer else "model-local",
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "sample_seeds": [row["seed"] for row in examples],
        "counts": {
            "output_format": counts("output_format"),
            "entity_type": counts("entity_type"),
            "require_wrapper_key": counts("require_wrapper_key"),
            "require_code_block": counts("require_code_block"),
            "require_no_commentary": counts("require_no_commentary"),
        },
    }


def remove_thinking_tags(text: str) -> str:
    text = re.sub(
        r"<think>.*?</think>|\[THINK\].*?\[/THINK\]",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(
        r"^.*?</think>|^.*?\[/THINK\]",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(
        r"<think>.*$|\[THINK\].*$",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return text.strip()


def extract_outer_fenced_block(response: str) -> tuple[str | None, str | None, str]:
    stripped = response.strip()
    lines = stripped.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("```")), None)
    if start is None:
        return None, None, ""
    opening = lines[start]
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("```")),
        None,
    )
    if end is None:
        return None, opening, "unclosed"
    outside = "\n".join(lines[:start] + lines[end + 1 :]).strip()
    return "\n".join(lines[start + 1 : end]), opening, outside


def contains_flow_mapping(node: yaml.nodes.Node) -> bool:
    if isinstance(node, yaml.MappingNode):
        if node.flow_style:
            return True
        return any(
            contains_flow_mapping(key) or contains_flow_mapping(value)
            for key, value in node.value
        )
    if isinstance(node, yaml.SequenceNode):
        return any(contains_flow_mapping(item) for item in node.value)
    return False


def parse_response(response: str, output_format: str) -> tuple[Any | None, list[str], dict[str, Any]]:
    block, opening, outside = extract_outer_fenced_block(response)
    details: dict[str, Any] = {
        "uses_code_block": opening is not None,
        "code_block_type": opening,
        "outside_code_block": outside,
    }
    if opening is not None and block is None:
        return None, ["Unclosed code block"], details
    content = block.strip() if block is not None else response.strip()
    lowered = (opening or "").lower()

    if output_format == "json":
        if lowered.startswith("```yaml") or lowered.startswith("```yml"):
            return None, ["Expected JSON output, got YAML code block"], details
        if not content or content[0] not in "[{":
            return None, ["No valid JSON found in response"], details
        try:
            decoder = json.JSONDecoder()
            parsed, end = decoder.raw_decode(content)
        except json.JSONDecodeError as error:
            return None, [f"JSON parse error: {error}"], details
        trailing = content[end:].strip()
        if trailing:
            return None, [f"Trailing content after JSON: {trailing[:100]!r}"], details
        details["json_valid"] = True
        return parsed, [], details

    if lowered.startswith("```json"):
        return None, ["Expected YAML output, got JSON code block"], details
    try:
        root = yaml.compose(content, Loader=_SafeLoaderNoDate)
        if root is not None and (
            getattr(root, "flow_style", None) is True or contains_flow_mapping(root)
        ):
            return None, ["Response is JSON/flow style, but YAML output was requested"], details
        parsed = yaml.load(content, Loader=_SafeLoaderNoDate)
    except (yaml.YAMLError, ValueError, RecursionError) as error:
        return None, [f"YAML parsing error: {error}"], details
    details["yaml_valid"] = True
    return parsed, [], details


def schema_errors(data: Any, schema: dict[str, Any], path: str = "root") -> list[str]:
    errors: list[str] = []
    schema_type = schema.get("type")
    allowed_types = schema_type if isinstance(schema_type, list) else [schema_type]
    if data is None and "null" in allowed_types:
        return errors
    non_null_types = [value for value in allowed_types if value != "null"]
    schema_type = non_null_types[0] if len(non_null_types) == 1 else schema_type

    if schema_type == "array":
        if not isinstance(data, list):
            return [f"{path}: expected array, got {type(data).__name__}"]
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if minimum is not None and len(data) < minimum:
            errors.append(f"{path}: array has {len(data)} items, minimum is {minimum}")
        if maximum is not None and len(data) > maximum:
            errors.append(f"{path}: array has {len(data)} items, maximum is {maximum}")
        if isinstance(schema.get("items"), dict):
            for index, item in enumerate(data):
                errors.extend(schema_errors(item, schema["items"], f"{path}[{index}]"))
        return errors

    if schema_type == "object":
        if not isinstance(data, dict):
            return [f"{path}: expected object, got {type(data).__name__}"]
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in data:
                errors.append(f"{path}.{key}: required field missing")
        for key, value in data.items():
            if key not in properties:
                errors.append(f"{path}.{key}: extraneous field")
            else:
                errors.extend(schema_errors(value, properties[key], f"{path}.{key}"))
        return errors

    type_ok = True
    if schema_type == "string":
        type_ok = isinstance(data, str)
    elif schema_type == "number":
        type_ok = isinstance(data, (int, float)) and not isinstance(data, bool)
    elif schema_type == "integer":
        type_ok = isinstance(data, int) and not isinstance(data, bool)
    elif schema_type == "boolean":
        type_ok = isinstance(data, bool)
    elif schema_type == "null":
        type_ok = data is None
    if not type_ok:
        return [f"{path}: expected {schema_type}, got {type(data).__name__}"]
    if schema.get("enum") is not None and data not in schema["enum"]:
        errors.append(f"{path}: {data!r} not in allowed values {schema['enum']}")
    if schema_type in ("number", "integer"):
        if schema.get("minimum") is not None and data < schema["minimum"]:
            errors.append(f"{path}: {data} is less than minimum {schema['minimum']}")
        if schema.get("maximum") is not None and data > schema["maximum"]:
            errors.append(f"{path}: {data} is greater than maximum {schema['maximum']}")
    return errors


def validate_response(response: str, example: dict[str, Any]) -> ValidationResult:
    response = remove_thinking_tags(response)
    output_format = example["output_format"]
    parsed, errors, details = parse_response(response, output_format)
    if example["require_code_block"] and not details.get("uses_code_block"):
        errors.insert(0, "Response must use a code block but none was found")
    if parsed is None:
        return ValidationResult(False, errors, details)
    if example["require_no_commentary"] and details.get("outside_code_block"):
        errors.append("Response contains commentary outside structured output")

    require_wrapper = example["require_wrapper_key"]
    expected_key = example["top_level_key"]
    if isinstance(parsed, list):
        if require_wrapper:
            errors.append(f"Expected wrapped object with key '{expected_key}', got bare list")
    elif isinstance(parsed, dict) and len(parsed) == 1:
        key = next(iter(parsed))
        value = parsed[key]
        if isinstance(value, list):
            if not require_wrapper:
                errors.append(f"Expected bare list, got wrapped object with key '{key}'")
            elif key != expected_key:
                errors.append(f"Expected top-level key '{expected_key}', got '{key}'")
            parsed = value

    errors.extend(schema_errors(parsed, example["json_schema"]))
    expected_count = example["top_level_count"]
    if isinstance(parsed, list):
        if isinstance(expected_count, int) and len(parsed) != expected_count:
            errors.append(f"Expected {expected_count} items, got {len(parsed)}")
        elif isinstance(expected_count, list) and len(expected_count) == 2:
            if not expected_count[0] <= len(parsed) <= expected_count[1]:
                errors.append(
                    f"Expected {expected_count[0]}-{expected_count[1]} items, got {len(parsed)}"
                )
    details["schema_valid"] = not any(
        "root" in error or "field" in error for error in errors
    )
    return ValidationResult(not errors, errors, details)


def discover_models(args: argparse.Namespace) -> list[Path]:
    if args.model:
        models = [path.expanduser().resolve() for path in args.model]
    else:
        root = args.models_dir.expanduser().resolve()
        models = [path for path in root.iterdir() if (path / "config.json").is_file()]
    if args.baseline_model:
        models.append(args.baseline_model.expanduser().resolve())
    models = list(dict.fromkeys(models))
    models = sorted(
        models,
        key=lambda path: (
            model_metadata(path).get("method", "baseline") != "baseline",
            path.name,
        ),
    )
    missing = [path for path in models if not (path / "config.json").is_file()]
    if missing:
        raise FileNotFoundError(f"Invalid model directories: {missing}")
    return models


def resolve_shared_tokenizer(
    args: argparse.Namespace, models: list[Path]
) -> Path | None:
    """Use one BF16 tokenizer so model inputs are identical across checkpoints."""
    if args.tokenizer:
        tokenizer = args.tokenizer.expanduser().resolve()
        if not (tokenizer / "tokenizer_config.json").is_file():
            raise FileNotFoundError(
                f"Invalid --tokenizer directory (tokenizer_config.json missing): {tokenizer}"
            )
        return tokenizer

    candidates = list(models)
    root = args.models_dir.expanduser().resolve()
    if root.is_dir():
        candidates.extend(
            path
            for path in root.iterdir()
            if path.is_dir() and "BF16" in path.name.upper()
        )
    if args.baseline_model:
        candidates.append(args.baseline_model.expanduser().resolve())

    for candidate in dict.fromkeys(candidates):
        is_baseline = (
            "BF16" in candidate.name.upper()
            or model_metadata(candidate).get("method") == "baseline"
        )
        if is_baseline and (candidate / "tokenizer_config.json").is_file():
            return candidate

    incompatible: list[Path] = []
    for model in models:
        config_path = model / "tokenizer_config.json"
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(config.get("extra_special_tokens"), list):
            incompatible.append(model)
    if incompatible:
        names = ", ".join(model.name for model in incompatible)
        raise RuntimeError(
            "No BF16/shared tokenizer was found, but these tokenizer configs contain "
            f"the version-sensitive extra_special_tokens list: {names}. "
            "Pass --tokenizer /path/to/Qwen3-8B-BF16."
        )
    return None


def build_accuracy_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(bool(sample["passed"]) for sample in samples)
    by_format: dict[str, dict[str, Any]] = {}
    for output_format in ("json", "yaml"):
        subset = [sample for sample in samples if sample["output_format"] == output_format]
        subset_passed = sum(bool(sample["passed"]) for sample in subset)
        by_format[output_format] = {
            "passed": subset_passed,
            "total": len(subset),
            "pass_rate": subset_passed / len(subset) if subset else 0.0,
        }
    errors = Counter()
    for sample in samples:
        for error in sample["errors"]:
            errors[error.split(":", 1)[-1].strip()[:100]] += 1
    return {
        "passed": passed,
        "total": len(samples),
        "pass_rate": passed / len(samples) if samples else 0.0,
        "by_format": by_format,
        "truncated": sum(sample["finish_reason"] == "length" for sample in samples),
        "common_errors": dict(errors.most_common(10)),
    }


def worker(args: argparse.Namespace) -> int:
    model_path = args.model[0].expanduser().resolve()
    tokenizer_path = args.tokenizer.expanduser().resolve() if args.tokenizer else None
    result_file = args._result_file.expanduser().resolve()
    dataset = ensure_dataset(args.dataset)
    examples = load_jsonl(dataset)
    if not examples:
        raise ValueError(f"Dataset has no rows: {dataset}")
    metadata = model_metadata(model_path)
    resumed_samples: list[dict[str, Any]] = []
    previous_status = ""
    if result_file.is_file():
        previous = json.loads(result_file.read_text(encoding="utf-8"))
        previous_status = str(previous.get("status", ""))
        candidate_samples = previous.get("samples", [])
        if isinstance(candidate_samples, list):
            keep = min(len(candidate_samples), len(examples))
            if keep < len(examples):
                keep -= keep % args.batch_size
            resumed_samples = candidate_samples[:keep]
            for index, sample in enumerate(resumed_samples):
                if sample.get("seed") != examples[index].get("seed"):
                    raise RuntimeError(
                        "Existing result sample order does not match the selected dataset"
                    )
    monitor = SystemMemoryMonitor(args.memory_sample_ms / 1000)
    monitor.start()
    result: dict[str, Any] = {
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": model_path.name,
        "model_path": str(model_path),
        "method": metadata.get("method", "baseline"),
        "status": "RUNNING",
        "error": "",
        "settings": {
            "dataset": str(dataset),
            "dataset_sha256": sha256_file(dataset),
            "ifstruct_commit": IFSTRUCT_COMMIT,
            "num_samples": len(examples),
            "batch_size": args.batch_size,
            "tokenizer": str(tokenizer_path) if tokenizer_path else "model-local",
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "enable_thinking": False,
            "constrained_decoding": False,
            "enforce_eager": args.enforce_eager,
        },
        "runtime_overrides": [],
        "samples": resumed_samples,
        "resume": {
            "previous_status": previous_status,
            "resumed_samples": len(resumed_samples),
        },
    }
    atomic_write_json(result_file, result)
    if resumed_samples:
        print(
            f"[RESUME] samples={len(resumed_samples)}/{len(examples)}",
            flush=True,
        )

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
            "max_num_seqs": args.batch_size,
            "max_num_batched_tokens": args.max_model_len * 2,
            "enforce_eager": args.enforce_eager,
            "enable_prefix_caching": False,
            "generation_config": "vllm",
            "disable_log_stats": True,
        }
        if tokenizer_path is not None:
            llm_kwargs["tokenizer"] = str(tokenizer_path)
        if hf_overrides is not None:
            llm_kwargs["hf_overrides"] = hf_overrides

        print(f"[LOAD] {model_path}", flush=True)
        load_started = time.perf_counter()
        llm = LLM(**llm_kwargs)
        result["load_seconds"] = round(time.perf_counter() - load_started, 6)

        print("[WARMUP] starting", flush=True)
        warmup_messages = [
            [{"role": "user", "content": f"Reply with the digit {index}."}]
            for index in range(args.batch_size)
        ]
        warmup_started = time.perf_counter()
        llm.chat(
            warmup_messages,
            sampling_params=SamplingParams(temperature=0.0, max_tokens=1, seed=42),
            use_tqdm=False,
            chat_template_kwargs={"enable_thinking": False},
        )
        result["warmup_seconds"] = round(time.perf_counter() - warmup_started, 6)
        print("[WARMUP] complete", flush=True)

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            seed=42,
        )
        previous_batch_latencies: dict[int, float] = {}
        for sample in resumed_samples:
            batch_index = int(sample["batch_index"])
            previous_batch_latencies.setdefault(
                batch_index, float(sample["batch_latency_seconds"])
            )
        batch_latencies = list(previous_batch_latencies.values())
        for offset in range(len(resumed_samples), len(examples), args.batch_size):
            batch = examples[offset : offset + args.batch_size]
            messages = [
                [{"role": "user", "content": example["prompt"]}]
                for example in batch
            ]
            batch_started = time.perf_counter()
            outputs = llm.chat(
                messages,
                sampling_params=sampling_params,
                use_tqdm=False,
                chat_template_kwargs={"enable_thinking": False},
            )
            batch_latency = time.perf_counter() - batch_started
            batch_latencies.append(batch_latency)
            for example, output in zip(batch, outputs, strict=True):
                completion = output.outputs[0]
                validation = validate_response(completion.text, example)
                result["samples"].append(
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
                        "response": completion.text,
                        "prompt_tokens": len(output.prompt_token_ids),
                        "output_tokens": len(completion.token_ids),
                        "finish_reason": completion.finish_reason,
                        "batch_index": offset // args.batch_size,
                        "batch_latency_seconds": round(batch_latency, 6),
                    }
                )
            partial = build_accuracy_summary(result["samples"])
            result["accuracy"] = partial
            atomic_write_json(result_file, result)
            print(
                f"[BATCH {offset // args.batch_size + 1}/"
                f"{(len(examples) + args.batch_size - 1) // args.batch_size}] "
                f"{batch_latency:.2f}s, pass={partial['passed']}/{partial['total']}",
                flush=True,
            )

        generation_seconds = sum(batch_latencies)
        prompt_tokens = sum(sample["prompt_tokens"] for sample in result["samples"])
        output_tokens = sum(sample["output_tokens"] for sample in result["samples"])
        result["accuracy"] = build_accuracy_summary(result["samples"])
        result["performance"] = {
            "generation_seconds": round(generation_seconds, 6),
            "batch_latency_mean_seconds": round(statistics.mean(batch_latencies), 6),
            "batch_latency_p50_seconds": round(statistics.median(batch_latencies), 6),
            "batch_latency_min_seconds": round(min(batch_latencies), 6),
            "batch_latency_max_seconds": round(max(batch_latencies), 6),
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "output_tokens_per_second": round(
                output_tokens / generation_seconds if generation_seconds else 0.0, 6
            ),
        }
        result["status"] = "SUCCESS"
    except BaseException as error:
        result["status"] = "FAILED"
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


def worker_command(
    args: argparse.Namespace, model: Path, result_file: Path, dataset: Path
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker",
        "--model",
        str(model),
        "--_result-file",
        str(result_file),
        "--dataset",
        str(dataset),
        "--batch-size",
        str(args.batch_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--max-tokens",
        str(args.max_tokens),
        "--memory-sample-ms",
        str(args.memory_sample_ms),
        "--enforce-eager" if args.enforce_eager else "--no-enforce-eager",
    ]
    if args.tokenizer:
        command.extend(["--tokenizer", str(args.tokenizer)])
    return command


def stop_worker_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def summary_row(
    result: dict[str, Any], result_file: Path, log_file: Path
) -> dict[str, Any]:
    accuracy = result.get("accuracy", {})
    formats = accuracy.get("by_format", {})
    performance = result.get("performance", {})
    memory = result.get("unified_memory", {})
    settings = result.get("settings", {})
    runtime = result.get("runtime", {})
    return {
        "model": result.get("model", result_file.stem),
        "method": result.get("method", ""),
        "status": result.get("status", "FAILED"),
        "error": result.get("error", ""),
        "passed": accuracy.get("passed", ""),
        "total": accuracy.get("total", ""),
        "pass_rate": accuracy.get("pass_rate", ""),
        "json_pass_rate": formats.get("json", {}).get("pass_rate", ""),
        "yaml_pass_rate": formats.get("yaml", {}).get("pass_rate", ""),
        "truncated": accuracy.get("truncated", ""),
        "generation_seconds": performance.get("generation_seconds", ""),
        "batch_latency_mean_seconds": performance.get("batch_latency_mean_seconds", ""),
        "prompt_tokens": performance.get("prompt_tokens", ""),
        "output_tokens": performance.get("output_tokens", ""),
        "output_tokens_per_second": performance.get("output_tokens_per_second", ""),
        "load_seconds": result.get("load_seconds", ""),
        "warmup_seconds": result.get("warmup_seconds", ""),
        "ram_peak_mib": memory.get("peak_mib", ""),
        "ram_peak_delta_mib": memory.get("peak_delta_mib", ""),
        "gpu_memory_utilization": settings.get("gpu_memory_utilization", ""),
        "batch_size": settings.get("batch_size", ""),
        "tokenizer": settings.get("tokenizer", ""),
        "max_model_len": settings.get("max_model_len", ""),
        "max_tokens": settings.get("max_tokens", ""),
        "vllm_version": runtime.get("vllm", ""),
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


def comparison_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = next(
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
    if baseline is None:
        return []
    base_accuracy = baseline["accuracy"]
    base_perf = baseline["performance"]
    base_samples = {sample["seed"]: sample for sample in baseline["samples"]}
    rows: list[dict[str, Any]] = []
    for result in results:
        accuracy = result.get("accuracy", {})
        performance = result.get("performance", {})
        memory = result.get("unified_memory", {})
        regressions = recoveries = agreement = ""
        if result.get("status") == "SUCCESS":
            regressions_count = recoveries_count = agreement_count = 0
            for sample in result["samples"]:
                base_sample = base_samples.get(sample["seed"])
                if base_sample is None:
                    continue
                if base_sample["passed"] and not sample["passed"]:
                    regressions_count += 1
                elif not base_sample["passed"] and sample["passed"]:
                    recoveries_count += 1
                else:
                    agreement_count += 1
            regressions, recoveries, agreement = (
                regressions_count,
                recoveries_count,
                agreement_count,
            )
        latency = performance.get("batch_latency_mean_seconds")
        throughput = performance.get("output_tokens_per_second")
        rows.append(
            {
                "model": result.get("model", ""),
                "method": result.get("method", ""),
                "status": result.get("status", "FAILED"),
                "passed": accuracy.get("passed", ""),
                "total": accuracy.get("total", ""),
                "pass_rate": accuracy.get("pass_rate", ""),
                "bf16_delta_points": round(
                    100 * (accuracy["pass_rate"] - base_accuracy["pass_rate"]), 3
                )
                if result.get("status") == "SUCCESS"
                else "",
                "paired_regressions": regressions,
                "paired_recoveries": recoveries,
                "paired_agreement": agreement,
                "batch_latency_mean_seconds": latency or "",
                "bf16_latency_speedup": round(
                    base_perf["batch_latency_mean_seconds"] / latency, 4
                )
                if latency
                else "",
                "output_tokens_per_second": throughput or "",
                "bf16_throughput_ratio": round(
                    throughput / base_perf["output_tokens_per_second"], 4
                )
                if throughput
                else "",
                "ram_peak_mib": memory.get("peak_mib", ""),
                "ram_peak_delta_mib": memory.get("peak_delta_mib", ""),
            }
        )
    return rows


def driver(args: argparse.Namespace) -> int:
    dataset = ensure_dataset(args.dataset)
    examples = load_jsonl(dataset)
    if not examples:
        print(f"Dataset has no rows: {dataset}", file=sys.stderr)
        return 2
    models = discover_models(args)
    if not models:
        print("No models found", file=sys.stderr)
        return 2
    args.tokenizer = resolve_shared_tokenizer(args, models)
    run_dir = args.results_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "sample_manifest.json"
    new_manifest = sample_manifest(dataset, examples, args)
    if manifest_path.is_file():
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        strict_keys = ("dataset_sha256", "num_samples", "batch_size")
        optional_legacy_keys = (
            "tokenizer",
            "max_model_len",
            "max_tokens",
            "gpu_memory_utilization",
            "enforce_eager",
        )
        for key in strict_keys + optional_legacy_keys:
            if key in old_manifest and old_manifest.get(key) != new_manifest.get(key):
                raise RuntimeError(
                    f"Cannot resume {run_dir}: manifest {key} changed "
                    f"({old_manifest.get(key)!r} != {new_manifest.get(key)!r}). "
                    "Use a different --results-dir."
                )
    atomic_write_json(manifest_path, new_manifest)
    print(f"Results: {run_dir}")
    print("Models: " + ", ".join(model.name for model in models))
    print(f"Shared tokenizer: {args.tokenizer or 'model-local'}")
    print(f"IFStruct samples: {len(examples)}, batch size: {args.batch_size}")

    results: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    interrupted = False
    for index, model in enumerate(models, 1):
        result_file = run_dir / f"{model.name}.json"
        log_file = run_dir / f"{model.name}.log"
        existing_result = (
            json.loads(result_file.read_text(encoding="utf-8"))
            if result_file.is_file()
            else None
        )
        if (
            existing_result
            and existing_result.get("status") == "SUCCESS"
            and len(existing_result.get("samples", [])) == len(examples)
        ):
            results.append(existing_result)
            rows.append(summary_row(existing_result, result_file, log_file))
            write_csv(run_dir / "summary.csv", SUMMARY_FIELDS, rows)
            atomic_write_json(run_dir / "summary.json", results)
            comparisons = comparison_rows(results)
            if comparisons:
                write_csv(
                    run_dir / "comparison_with_bf16.csv",
                    COMPARISON_FIELDS,
                    comparisons,
                )
            print(
                f"[{index}/{len(models)}] {model.name}: already SUCCESS, skipping",
                flush=True,
            )
            continue
        print(flush=True)
        environment = os.environ.copy()
        environment.setdefault("HF_HUB_OFFLINE", "1")
        environment.setdefault("TRANSFORMERS_OFFLINE", "1")
        environment.setdefault("TOKENIZERS_PARALLELISM", "false")
        total_batches = (len(examples) + args.batch_size - 1) // args.batch_size
        resumed_count = (
            len(existing_result.get("samples", [])) if existing_result else 0
        )
        resumed_count = min(resumed_count, len(examples))
        if resumed_count < len(examples):
            resumed_count -= resumed_count % args.batch_size
        initial_batches = min(
            total_batches,
            (resumed_count + args.batch_size - 1) // args.batch_size,
        )
        progress = tqdm(
            total=total_batches,
            initial=initial_batches,
            desc=f"[{index}/{len(models)}] {model.name}",
            unit="batch",
            dynamic_ncols=True,
            disable=not args.progress,
        )
        progress.set_postfix_str("starting worker")
        with log_file.open("a", encoding="utf-8") as log:
            log.write(f"\n===== resume attempt {datetime.now().astimezone().isoformat()} =====\n")
            log.flush()
            process = subprocess.Popen(
                worker_command(args, model, result_file, dataset),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            try:
                if process.stdout is None:
                    raise RuntimeError("worker stdout pipe was not created")
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    message = line.strip()
                    if message.startswith("[LOAD]"):
                        progress.set_postfix_str("loading model")
                        progress.refresh()
                    elif message == "[WARMUP] starting":
                        progress.set_postfix_str("warming up")
                        progress.refresh()
                    elif message == "[WARMUP] complete":
                        progress.set_postfix_str("generating")
                        progress.refresh()
                    elif batch_match := re.match(
                        r"^\[BATCH (\d+)/(\d+)\] ([0-9.]+)s, pass=(\d+)/(\d+)$",
                        message,
                    ):
                        completed = int(batch_match.group(1))
                        latency = batch_match.group(3)
                        passed = batch_match.group(4)
                        evaluated = batch_match.group(5)
                        progress.set_postfix_str(
                            f"last={latency}s, pass={passed}/{evaluated}"
                        )
                        progress.update(max(0, completed - progress.n))
                return_code = process.wait()
            except KeyboardInterrupt:
                interrupted = True
                stop_worker_group(process)
                return_code = process.returncode if process.returncode is not None else 130
            finally:
                if process.stdout is not None:
                    process.stdout.close()
                progress.close()
        if result_file.is_file():
            result = json.loads(result_file.read_text(encoding="utf-8"))
            if return_code != 0 and result.get("status") == "RUNNING":
                result["status"] = "INTERRUPTED"
                result["error"] = f"worker exited with {return_code} before completion"
                atomic_write_json(result_file, result)
        else:
            result = {
                "model": model.name,
                "method": model_metadata(model).get("method", "baseline"),
                "status": "FAILED",
                "error": f"worker exited with {return_code}; inspect log",
            }
            atomic_write_json(result_file, result)
        results.append(result)
        rows.append(summary_row(result, result_file, log_file))
        write_csv(run_dir / "summary.csv", SUMMARY_FIELDS, rows)
        atomic_write_json(run_dir / "summary.json", results)
        comparisons = comparison_rows(results)
        if comparisons:
            write_csv(run_dir / "comparison_with_bf16.csv", COMPARISON_FIELDS, comparisons)
        accuracy = result.get("accuracy", {})
        print(
            f"{result['status']}: {accuracy.get('passed', '-')}/"
            f"{accuracy.get('total', '-')} passed; log={log_file.name}",
            flush=True,
        )
        if interrupted:
            break

    successes = sum(result.get("status") == "SUCCESS" for result in results)
    print(f"\nCompleted: {successes}/{len(results)} successful")
    print(f"Summary: {run_dir / 'summary.csv'}")
    return 0 if successes == len(models) and not interrupted else 1


def main() -> None:
    args = parse_args()
    if args.prepare_datasets:
        raise SystemExit(prepare_sample_datasets())
    raise SystemExit(worker(args) if args._worker else driver(args))


if __name__ == "__main__":
    main()
