#!/usr/bin/env python3
"""Convertit le CNN MotionSense en modèle TensorFlow Lite entièrement int8."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from train_model import CLASS_NAMES, build_windows, load_manifest, load_recording


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("models/activity_cnn.keras"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"))
    parser.add_argument("--training-metadata", type=Path, default=Path("models/training_metadata.json"))
    parser.add_argument("--output-path", type=Path, default=Path("models/activity_cnn_int8.tflite"))
    parser.add_argument("--metadata-path", type=Path, default=Path("models/tflite_metadata.json"))
    parser.add_argument(
        "--calibration-windows-per-class",
        type=int,
        default=20,
        help="nombre de fenêtres train représentatives utilisées par classe pour la quantification",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_training_windows(
    manifest_path: Path,
    train_subjects: set[int],
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Construit des fenêtres uniquement à partir des participants train."""
    recordings: list[tuple[np.ndarray, int]] = []
    for row in load_manifest(manifest_path):
        if int(row["subject_id"]) not in train_subjects:
            continue
        recording_path = manifest_path.parent / row["recording_path"]
        recordings.append((load_recording(recording_path), int(row["class_id"])))
    return build_windows(recordings, window_size, stride)


def select_calibration_windows(
    windows: np.ndarray,
    labels: np.ndarray,
    per_class: int,
    seed: int,
) -> np.ndarray:
    """Sélectionne un échantillon équilibré, sans utiliser validation ni test."""
    if per_class <= 0:
        raise ValueError("--calibration-windows-per-class doit être strictement positif.")
    random_generator = np.random.default_rng(seed)
    selected_indices: list[np.ndarray] = []
    for class_id in range(len(CLASS_NAMES)):
        candidates = np.flatnonzero(labels == class_id)
        if len(candidates) < per_class:
            raise ValueError(f"Pas assez de fenêtres train pour la classe {CLASS_NAMES[class_id]}.")
        selected_indices.append(random_generator.choice(candidates, size=per_class, replace=False))
    indices = np.concatenate(selected_indices)
    random_generator.shuffle(indices)
    return windows[indices]


def representative_dataset(windows: np.ndarray) -> Iterator[list[np.ndarray]]:
    for window in windows:
        yield [window[np.newaxis, ...].astype(np.float32)]


def tensor_spec(detail: dict[str, Any]) -> dict[str, Any]:
    scale, zero_point = detail["quantization"]
    return {
        "shape": detail["shape"].tolist(),
        "dtype": np.dtype(detail["dtype"]).name,
        "quantization": {"scale": scale, "zero_point": zero_point},
    }


def main() -> None:
    args = parse_arguments()
    metadata = load_json(args.training_metadata)
    train_subjects = {int(subject) for subject in metadata["participant_split"]["train"]}
    windows, labels = load_training_windows(
        args.manifest,
        train_subjects,
        int(metadata["window_size_samples"]),
        int(metadata["stride_samples"]),
    )
    calibration_windows = select_calibration_windows(
        windows,
        labels,
        args.calibration_windows_per_class,
        int(metadata["seed"]),
    )

    model = tf.keras.models.load_model(args.model_path, compile=False)
    # TensorFlow 2.16 / Keras 3 échoue parfois à convertir directement un fichier
    # .keras avec des poids variables sur macOS. Le SavedModel temporaire fige la
    # signature d'inférence et évite ce défaut du convertisseur MLIR.
    with tempfile.TemporaryDirectory(prefix="motionsense_tflite_") as temporary_directory:
        saved_model_path = Path(temporary_directory) / "saved_model"
        model.export(str(saved_model_path))
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: representative_dataset(calibration_windows)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_bytes(tflite_model)

    interpreter = tf.lite.Interpreter(model_path=str(args.output_path))
    interpreter.allocate_tensors()
    export_metadata = {
        "source_model": str(args.model_path),
        "format": "TensorFlow Lite",
        "quantization": "full_int8",
        "model_size_bytes": len(tflite_model),
        "class_names": metadata["class_names"],
        "sensor_columns": metadata["sensor_columns"],
        "window_size_samples": metadata["window_size_samples"],
        "stride_samples": metadata["stride_samples"],
        "sampling_frequency_hz": metadata["sampling_frequency_hz"],
        "calibration": {
            "participants": sorted(train_subjects),
            "windows_per_class": args.calibration_windows_per_class,
            "windows_total": int(len(calibration_windows)),
        },
        "input_tensor": tensor_spec(interpreter.get_input_details()[0]),
        "output_tensor": tensor_spec(interpreter.get_output_details()[0]),
    }
    with args.metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(export_metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Modèle TensorFlow Lite int8 exporté : {args.output_path}")
    print(f"Taille : {len(tflite_model)} octets")
    print(f"Calibration : {len(calibration_windows)} fenêtres issues uniquement de train")
    print(f"Métadonnées : {args.metadata_path}")


if __name__ == "__main__":
    main()
