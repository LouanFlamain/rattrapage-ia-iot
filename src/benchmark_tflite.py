#!/usr/bin/env python3
"""Measure TensorFlow Lite model size and latency reproducibly."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from train_model import build_windows, load_manifest, load_recording


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("models/activity_cnn_int8.tflite"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"))
    parser.add_argument("--training-metadata", type=Path, default=Path("models/training_metadata.json"))
    parser.add_argument("--runs", type=int, default=1000, help="number of timed inferences")
    parser.add_argument("--warmup-runs", type=int, default=100, help="warm-up inferences excluded from the benchmark")
    parser.add_argument("--report-path", type=Path, default=Path("reports/edge_benchmark.json"))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_test_windows(
    manifest_path: Path,
    test_subjects: set[int],
    window_size: int,
    stride: int,
) -> np.ndarray:
    recordings: list[tuple[np.ndarray, int]] = []
    for row in load_manifest(manifest_path):
        if int(row["subject_id"]) not in test_subjects:
            continue
        recordings.append(
            (load_recording(manifest_path.parent / row["recording_path"]), int(row["class_id"]))
        )
    windows, _ = build_windows(recordings, window_size, stride)
    return windows


def quantize(values: np.ndarray, tensor_details: dict[str, Any]) -> np.ndarray:
    dtype = tensor_details["dtype"]
    scale, zero_point = tensor_details["quantization"]
    if np.issubdtype(dtype, np.integer):
        if scale == 0:
            raise ValueError("Invalid input quantization (zero scale).")
        limits = np.iinfo(dtype)
        encoded = np.round(values / scale + zero_point)
        return np.clip(encoded, limits.min, limits.max).astype(dtype)
    return values.astype(dtype)


def summary_milliseconds(samples_ns: list[int]) -> dict[str, float]:
    milliseconds = np.asarray(samples_ns, dtype=np.float64) / 1_000_000
    return {
        "mean_ms": float(milliseconds.mean()),
        "median_ms": float(np.median(milliseconds)),
        "p95_ms": float(np.percentile(milliseconds, 95)),
        "min_ms": float(milliseconds.min()),
        "max_ms": float(milliseconds.max()),
    }


def main() -> None:
    args = parse_arguments()
    if args.runs <= 0 or args.warmup_runs < 0:
        raise SystemExit("--runs must be positive and --warmup-runs cannot be negative.")

    metadata = load_json(args.training_metadata)
    test_subjects = {int(subject) for subject in metadata["participant_split"]["test"]}
    windows = load_test_windows(
        args.manifest,
        test_subjects,
        int(metadata["window_size_samples"]),
        int(metadata["stride_samples"]),
    )
    random_generator = np.random.default_rng(int(metadata["seed"]))
    selected_indices = random_generator.choice(len(windows), size=args.runs, replace=len(windows) < args.runs)
    selected_windows = windows[selected_indices, np.newaxis, ...]

    interpreter = tf.lite.Interpreter(model_path=str(args.model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    quantized_windows = [quantize(window, input_details) for window in selected_windows]

    # Loading and allocation are excluded because they happen only when the device
    # starts. Warm-up stabilizes the host CPU delegate.
    for window in quantized_windows[: min(args.warmup_runs, len(quantized_windows))]:
        interpreter.set_tensor(input_details["index"], window)
        interpreter.invoke()
        interpreter.get_tensor(output_details["index"])

    latency_ns: list[int] = []
    for window in quantized_windows:
        start = time.perf_counter_ns()
        interpreter.set_tensor(input_details["index"], window)
        interpreter.invoke()
        interpreter.get_tensor(output_details["index"])
        latency_ns.append(time.perf_counter_ns() - start)

    benchmark = {
        "model_path": str(args.model_path),
        "model_size_bytes": args.model_path.stat().st_size,
        "model_size_kib": args.model_path.stat().st_size / 1024,
        "inferences_measured": args.runs,
        "warmup_inferences": min(args.warmup_runs, len(quantized_windows)),
        "input_shape": input_details["shape"].tolist(),
        "input_dtype": np.dtype(input_details["dtype"]).name,
        "latency": summary_milliseconds(latency_ns),
        "measurement_scope": "set_tensor + invoke + get_tensor; allocation, CSV reading, windowing and quantization excluded",
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
        },
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    with args.report_path.open("w", encoding="utf-8") as handle:
        json.dump(benchmark, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    latency = benchmark["latency"]
    print(f"Model size: {benchmark['model_size_bytes']} bytes ({benchmark['model_size_kib']:.2f} KiB)")
    print(
        "Mean TFLite latency: "
        f"{latency['mean_ms']:.4f} ms "
        f"(median {latency['median_ms']:.4f} ms, p95 {latency['p95_ms']:.4f} ms)"
    )
    print(f"Report: {args.report_path}")


if __name__ == "__main__":
    main()
