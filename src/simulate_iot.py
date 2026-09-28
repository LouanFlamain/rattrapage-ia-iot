#!/usr/bin/env python3
"""Simule l'arrivée progressive de mesures capteur et l'inférence TFLite."""

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
        help="CSV nettoyé rejoué comme un flux de capteur",
    )
    parser.add_argument("--model-path", type=Path, default=Path("models/activity_cnn_int8.tflite"))
    parser.add_argument("--metadata-path", type=Path, default=Path("models/tflite_metadata.json"))
    parser.add_argument(
        "--max-predictions",
        type=int,
        default=10,
        help="limite d'affichages ; 0 rejoue le fichier entier",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="attend entre deux prédictions pour reproduire la cadence d'un capteur à 50 Hz",
    )
    return parser.parse_args()


def load_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def quantize_input(values: np.ndarray, tensor_details: dict[str, Any]) -> np.ndarray:
    """Convertit les flottants capteur vers le type int8 attendu par TFLite."""
    dtype = tensor_details["dtype"]
    scale, zero_point = tensor_details["quantization"]
    if np.issubdtype(dtype, np.integer):
        if scale == 0:
            raise ValueError("Quantification d'entrée invalide (échelle nulle).")
        quantized = np.round(values / scale + zero_point)
        limits = np.iinfo(dtype)
        return np.clip(quantized, limits.min, limits.max).astype(dtype)
    return values.astype(dtype)


def dequantize_output(values: np.ndarray, tensor_details: dict[str, Any]) -> np.ndarray:
    """Convertit la sortie TFLite en probabilités flottantes lisibles."""
    scale, zero_point = tensor_details["quantization"]
    if np.issubdtype(tensor_details["dtype"], np.integer):
        return (values.astype(np.float32) - zero_point) * scale
    return values.astype(np.float32)


def iter_sensor_rows(input_file: Path, sensor_columns: list[str]) -> Iterator[tuple[dict[str, str], np.ndarray]]:
    """Lit le CSV une ligne à la fois, comme un capteur enverrait ses mesures."""
    with input_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"En-tête absent : {input_file}")
        missing = set(sensor_columns).difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Colonnes capteur absentes : {', '.join(sorted(missing))}")
        for row in reader:
            sample = np.asarray([float(row[column]) for column in sensor_columns], dtype=np.float32)
            yield row, sample


def main() -> None:
    args = parse_arguments()
    if args.max_predictions < 0:
        raise SystemExit("--max-predictions doit être positif ou nul.")

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
        raise ValueError(f"Entrée TFLite inattendue : {input_details['shape']}, attendu : {expected_shape}")

    # À chaque nouvelle fenêtre complète, le microcontrôleur reçoit ici 50 mesures
    # supplémentaires (une seconde). La première décision est disponible après 2 s.
    stride = int(metadata.get("stride_samples", window_size // 2))
    buffer: deque[np.ndarray] = deque(maxlen=window_size)
    predictions = 0
    correct_predictions = 0
    print(f"Simulation du fichier : {args.input_file}")
    print(f"Fenêtre : {window_size} mesures ({window_size / sampling_frequency:.1f} s), pas : {stride} mesures")

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
            f"t={elapsed_seconds:7.2f} s | mouvement détecté : {class_names[predicted_id]:>3} "
            f"| confiance : {confidence * 100:5.1f} % | vérité fichier : {true_activity}"
        )
        if args.realtime:
            time.sleep(stride / sampling_frequency)
        if args.max_predictions and predictions >= args.max_predictions:
            break

    if predictions == 0:
        raise SystemExit("Le fichier ne contient pas assez de mesures pour former une fenêtre.")
    print(f"{predictions} prédictions simulées ; accord avec le label du fichier : {correct_predictions}/{predictions}.")


if __name__ == "__main__":
    main()
