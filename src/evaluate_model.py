#!/usr/bin/env python3
"""Evaluate the MotionSense model on test-set participants only."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from train_model import CLASS_NAMES, build_windows, load_manifest, load_recording


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"))
    parser.add_argument("--model-path", type=Path, default=Path("models/activity_cnn.keras"))
    parser.add_argument("--metadata-path", type=Path, default=Path("models/training_metadata.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def load_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_test_windows(
    manifest_path: Path,
    test_subjects: set[int],
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild test windows without crossing recording boundaries."""
    recordings: list[tuple[np.ndarray, int]] = []
    for row in load_manifest(manifest_path):
        if int(row["subject_id"]) not in test_subjects:
            continue
        recordings.append(
            (
                load_recording(manifest_path.parent / row["recording_path"]),
                int(row["class_id"]),
            )
        )
    if not recordings:
        raise ValueError("No test recording was found in the manifest.")
    return build_windows(recordings, window_size, stride)


def confusion_matrix(y_true: np.ndarray, y_predicted: np.ndarray, class_count: int) -> np.ndarray:
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(matrix, (y_true, y_predicted), 1)
    return matrix


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    )


def compute_metrics(matrix: np.ndarray) -> tuple[list[dict[str, float | int | str]], dict[str, float]]:
    true_positive = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.float64)
    predicted_count = matrix.sum(axis=0).astype(np.float64)
    precision = safe_divide(true_positive, predicted_count)
    recall = safe_divide(true_positive, support)
    f1_score = safe_divide(2 * precision * recall, precision + recall)

    per_class: list[dict[str, float | int | str]] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1_score": float(f1_score[class_id]),
                "support": int(support[class_id]),
            }
        )

    total = matrix.sum()
    metrics = {
        "accuracy": float(true_positive.sum() / total),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1_score": float(f1_score.mean()),
        "weighted_f1_score": float(np.average(f1_score, weights=support)),
    }
    return per_class, metrics


def identify_confusions(matrix: np.ndarray) -> list[dict[str, float | int | str]]:
    """Return true-to-predicted errors sorted by descending frequency."""
    errors: list[dict[str, float | int | str]] = []
    for true_id, true_name in enumerate(CLASS_NAMES):
        row_total = int(matrix[true_id].sum())
        for predicted_id, predicted_name in enumerate(CLASS_NAMES):
            if true_id == predicted_id or matrix[true_id, predicted_id] == 0:
                continue
            count = int(matrix[true_id, predicted_id])
            errors.append(
                {
                    "true_class": true_name,
                    "predicted_class": predicted_name,
                    "count": count,
                    "rate_within_true_class": count / row_total,
                }
            )
    return sorted(errors, key=lambda error: int(error["count"]), reverse=True)


def write_reports(
    output_dir: Path,
    matrix: np.ndarray,
    per_class: list[dict[str, float | int | str]],
    report: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_class \\ predicted_class", *CLASS_NAMES])
        for class_name, row in zip(CLASS_NAMES, matrix):
            writer.writerow([class_name, *row.tolist()])

    with (output_dir / "classification_report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_id", "class_name", "precision", "recall", "f1_score", "support"])
        writer.writeheader()
        writer.writerows(per_class)

    with (output_dir / "evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    args = parse_arguments()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be strictly positive.")

    metadata = load_metadata(args.metadata_path)
    test_subjects = {int(subject) for subject in metadata["participant_split"]["test"]}
    x_test, y_test = load_test_windows(
        args.manifest,
        test_subjects,
        int(metadata["window_size_samples"]),
        int(metadata["stride_samples"]),
    )
    model = tf.keras.models.load_model(args.model_path, compile=False)
    expected_shape = (None, int(metadata["window_size_samples"]), len(metadata["sensor_columns"]))
    if tuple(model.input_shape) != expected_shape:
        raise ValueError(f"Unexpected model input: {model.input_shape}, expected: {expected_shape}")

    probabilities = model.predict(x_test, batch_size=args.batch_size, verbose=0)
    y_predicted = probabilities.argmax(axis=1).astype(np.int32)
    matrix = confusion_matrix(y_test, y_predicted, len(CLASS_NAMES))
    per_class, overall_metrics = compute_metrics(matrix)

    report = {
        "evaluation_split": "test",
        "test_participants": sorted(test_subjects),
        "window_count": int(len(y_test)),
        "class_names": list(CLASS_NAMES),
        "overall_metrics": overall_metrics,
        "per_class_metrics": per_class,
        "confusion_matrix": matrix.tolist(),
        "most_frequent_confusions": identify_confusions(matrix),
    }
    write_reports(args.output_dir, matrix, per_class, report)

    print(f"Test set: {len(y_test)} windows, participants {sorted(test_subjects)}")
    print(f"Accuracy: {overall_metrics['accuracy']:.4f}")
    print(f"Macro F1: {overall_metrics['macro_f1_score']:.4f}")
    print(f"Reports written to: {args.output_dir}")


if __name__ == "__main__":
    main()
