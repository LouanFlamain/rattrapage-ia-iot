#!/usr/bin/env python3
"""Simulate progressive sensor measurements and TFLite inference."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterator

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


DEFAULT_INPUT_FILE = Path("data/processed/clean_recordings/jog_9/sub_2.csv")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="cleaned CSV replayed as a sensor stream",
    )
    parser.add_argument("--model-path", type=Path, default=Path("models/activity_cnn_int8.tflite"))
    parser.add_argument("--metadata-path", type=Path, default=Path("models/tflite_metadata.json"))
    parser.add_argument(
        "--max-predictions",
        type=int,
        default=10,
        help="maximum number of displayed predictions; 0 replays the whole file",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="wait between predictions to reproduce a 50 Hz sensor rate",
    )
    return parser.parse_args()


def load_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def quantize_input(values: np.ndarray, tensor_details: dict[str, Any]) -> np.ndarray:
    """Convert floating-point sensor data to the int8 type expected by TFLite."""
    dtype = tensor_details["dtype"]
    scale, zero_point = tensor_details["quantization"]
    if np.issubdtype(dtype, np.integer):
        if scale == 0:
            raise ValueError("Invalid input quantization (zero scale).")
        quantized = np.round(values / scale + zero_point)
        limits = np.iinfo(dtype)
        return np.clip(quantized, limits.min, limits.max).astype(dtype)
    return values.astype(dtype)


def dequantize_output(values: np.ndarray, tensor_details: dict[str, Any]) -> np.ndarray:
    """Convert the TFLite output to readable floating-point probabilities."""
    scale, zero_point = tensor_details["quantization"]
    if np.issubdtype(tensor_details["dtype"], np.integer):
        return (values.astype(np.float32) - zero_point) * scale
    return values.astype(np.float32)


def iter_sensor_rows(input_file: Path, sensor_columns: list[str]) -> Iterator[tuple[dict[str, str], np.ndarray]]:
    """Read the CSV one row at a time, as a sensor would send its measurements."""
    with input_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header: {input_file}")
        missing = set(sensor_columns).difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing sensor columns: {', '.join(sorted(missing))}")
        for row in reader:
            sample = np.asarray([float(row[column]) for column in sensor_columns], dtype=np.float32)
            yield row, sample


def main() -> None:
    args = parse_arguments()
    if args.max_predictions < 0:
        raise SystemExit("--max-predictions must be zero or positive.")

    metadata = load_metadata(args.metadata_path)
    class_names = metadata["class_names"]
    sensor_columns = metadata["sensor_columns"]
    window_size = int(metadata["window_size_samples"])
    sampling_frequency = float(metadata["sampling_frequency_hz"])

    interpreter = tf.lite.Interpreter(model_path=str(args.model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    expected_shape = (1, window_size, len(sensor_columns))
    if tuple(input_details["shape"]) != expected_shape:
        raise ValueError(f"Unexpected TFLite input: {input_details['shape']}, expected: {expected_shape}")

    # Each complete window receives 50 new measurements (one second). The first
    # decision is available after two seconds.
    stride = int(metadata.get("stride_samples", window_size // 2))
    buffer: deque[np.ndarray] = deque(maxlen=window_size)
    predictions = 0
    correct_predictions = 0
    print(f"Simulating file: {args.input_file}")
    print(f"Window: {window_size} samples ({window_size / sampling_frequency:.1f} s), stride: {stride} samples")

    for sample_index, (row, sample) in enumerate(iter_sensor_rows(args.input_file, sensor_columns), start=1):
        buffer.append(sample)
        if len(buffer) != window_size or (sample_index - window_size) % stride != 0:
            continue

        window = np.asarray(buffer, dtype=np.float32)[np.newaxis, ...]
        interpreter.set_tensor(input_details["index"], quantize_input(window, input_details))
        interpreter.invoke()
        probabilities = dequantize_output(interpreter.get_tensor(output_details["index"])[0], output_details)
        predicted_id = int(np.argmax(probabilities))
        confidence = float(np.clip(probabilities[predicted_id], 0.0, 1.0))
        true_activity = row.get("activity", "inconnue")
        correct_predictions += int(class_names[predicted_id] == true_activity)
        predictions += 1

        elapsed_seconds = sample_index / sampling_frequency
        print(
            f"t={elapsed_seconds:7.2f} s | detected activity: {class_names[predicted_id]:>3} "
            f"| confidence: {confidence * 100:5.1f} % | file label: {true_activity}"
        )
        if args.realtime:
            time.sleep(stride / sampling_frequency)
        if args.max_predictions and predictions >= args.max_predictions:
            break

    if predictions == 0:
        raise SystemExit("The file does not contain enough samples to form a window.")
    print(f"{predictions} simulated predictions; matches with the file label: {correct_predictions}/{predictions}.")


if __name__ == "__main__":
    main()
