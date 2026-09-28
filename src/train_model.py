#!/usr/bin/env python3
"""Construit les fenêtres MotionSense et entraîne un petit CNN 1D.

La séparation est réalisée par participant avant tout découpage en fenêtres.
Ainsi, aucune séquence d'un participant de test ne peut être vue à l'entraînement.
Les paramètres de normalisation sont calculés uniquement avec les enregistrements
d'entraînement et sont intégrés au modèle Keras sauvegardé.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


CLASS_NAMES = ("dws", "ups", "wlk", "jog", "sit", "std")
SENSOR_COLUMNS = (
    "gravity.x",
    "gravity.y",
    "gravity.z",
    "userAcceleration.x",
    "userAcceleration.y",
    "userAcceleration.z",
    "rotationRate.x",
    "rotationRate.y",
    "rotationRate.z",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/manifest.csv"),
        help="manifeste généré par src/clean_data.py",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/activity_cnn.keras"),
        help="chemin de sauvegarde du modèle Keras entraîné",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("models/training_metadata.json"),
        help="chemin des métadonnées de préparation et d'entraînement",
    )
    parser.add_argument("--window-size", type=int, default=100, help="taille d'une fenêtre en échantillons")
    parser.add_argument("--stride", type=int, default=50, help="décalage entre deux fenêtres")
    parser.add_argument("--epochs", type=int, default=25, help="nombre maximal d'époques")
    parser.add_argument("--batch-size", type=int, default=128, help="taille des lots d'entraînement")
    parser.add_argument("--seed", type=int, default=42, help="graine de reproductibilité")
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Le manifeste {path} est vide.")
    required = {"recording_path", "class_id", "subject_id"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Colonnes absentes du manifeste : {', '.join(sorted(missing))}")
    return rows


def participant_split(subject_ids: set[int], seed: int) -> dict[str, list[int]]:
    """Crée un split 16 / 4 / 4, fixe et indépendant des fenêtres."""
    if len(subject_ids) != 24:
        raise ValueError(f"24 participants sont attendus, {len(subject_ids)} trouvés.")
    shuffled = np.random.default_rng(seed).permutation(sorted(subject_ids)).tolist()
    return {
        "train": sorted(shuffled[:16]),
        "validation": sorted(shuffled[16:20]),
        "test": sorted(shuffled[20:]),
    }


def load_recording(path: Path) -> np.ndarray:
    """Charge les neuf canaux nettoyés d'un enregistrement en float32."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [[float(row[column]) for column in SENSOR_COLUMNS] for row in reader]
    if not rows:
        raise ValueError(f"Enregistrement vide : {path}")
    return np.asarray(rows, dtype=np.float32)


def build_windows(recordings: list[tuple[np.ndarray, int]], window_size: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Découpe uniquement à l'intérieur de chaque enregistrement individuel."""
    windows: list[np.ndarray] = []
    labels: list[int] = []
    for values, class_id in recordings:
        for start in range(0, len(values) - window_size + 1, stride):
            windows.append(values[start : start + window_size])
            labels.append(class_id)
    if not windows:
        raise ValueError("Aucune fenêtre n'a été créée : vérifiez la taille et le décalage.")
    return np.asarray(windows, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def build_model(window_size: int, mean: np.ndarray, std: np.ndarray) -> tf.keras.Model:
    """Construit un CNN 1D d'environ 2,4 k paramètres entraînables."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(window_size, len(SENSOR_COLUMNS)), name="sensor_window"),
            # Les statistiques, apprises uniquement sur train, voyagent avec le modèle.
            tf.keras.layers.Normalization(mean=mean, variance=np.square(std), name="normalization"),
            tf.keras.layers.Conv1D(16, kernel_size=5, padding="same", activation="relu", name="conv_1"),
            tf.keras.layers.MaxPooling1D(pool_size=2, name="pool_1"),
            tf.keras.layers.Conv1D(24, kernel_size=3, padding="same", activation="relu", name="conv_2"),
            tf.keras.layers.GlobalAveragePooling1D(name="global_average_pooling"),
            tf.keras.layers.Dense(16, activation="relu", name="dense_1"),
            tf.keras.layers.Dropout(0.15, name="dropout"),
            tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax", name="activity"),
        ],
        name="motionsense_tiny_cnn",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def count_labels(labels: np.ndarray) -> dict[str, int]:
    counts = Counter(int(label) for label in labels)
    return {CLASS_NAMES[class_id]: counts[class_id] for class_id in range(len(CLASS_NAMES))}


def json_ready(value: Any) -> Any:
    """Convertit les scalaires et tableaux NumPy avant sérialisation JSON."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def main() -> None:
    args = parse_arguments()
    if args.window_size <= 0 or args.stride <= 0:
        raise SystemExit("--window-size et --stride doivent être strictement positifs.")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise SystemExit("--epochs et --batch-size doivent être strictement positifs.")

    # Ces paramètres assurent des résultats reproductibles dans la mesure permise
    # par les bibliothèques numériques de la machine.
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()

    manifest_path = args.manifest.resolve()
    manifest_rows = load_manifest(manifest_path)
    split = participant_split({int(row["subject_id"]) for row in manifest_rows}, args.seed)
    split_by_subject = {
        subject: partition
        for partition, subjects in split.items()
        for subject in subjects
    }

    recordings: dict[str, list[tuple[np.ndarray, int]]] = {name: [] for name in split}
    manifest_directory = manifest_path.parent
    for row in manifest_rows:
        partition = split_by_subject[int(row["subject_id"])]
        recording_path = manifest_directory / row["recording_path"]
        recordings[partition].append((load_recording(recording_path), int(row["class_id"])))

    # Statistiques calculées avant le fenêtrage, exclusivement sur train. Chaque
    # échantillon brut compte une fois, sans surpondération due au recouvrement.
    train_samples = np.concatenate([values for values, _ in recordings["train"]], axis=0)
    mean = train_samples.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_samples.std(axis=0, dtype=np.float64).astype(np.float32)
    if np.any(std == 0):
        raise ValueError("Au moins un canal d'entraînement a une variance nulle.")

    x_train, y_train = build_windows(recordings["train"], args.window_size, args.stride)
    x_validation, y_validation = build_windows(recordings["validation"], args.window_size, args.stride)
    x_test, y_test = build_windows(recordings["test"], args.window_size, args.stride)

    # La durée des essais est inégale : ce poids donne la même importance à chaque
    # activité pendant l'apprentissage, sans modifier le jeu de test naturel.
    train_counts = np.bincount(y_train, minlength=len(CLASS_NAMES))
    class_weight = {
        class_id: float(len(y_train) / (len(CLASS_NAMES) * count))
        for class_id, count in enumerate(train_counts)
        if count > 0
    }

    model = build_model(args.window_size, mean, std)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", mode="max", patience=5, restore_best_weights=True
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=[early_stopping],
        verbose=2,
    )

    train_metrics = model.evaluate(x_train, y_train, verbose=0, return_dict=True)
    validation_metrics = model.evaluate(x_validation, y_validation, verbose=0, return_dict=True)
    test_metrics = model.evaluate(x_test, y_test, verbose=0, return_dict=True)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_path)

    metadata = {
        "dataset": "MotionSense A_DeviceMotion_data",
        "sampling_frequency_hz": 50,
        "class_names": list(CLASS_NAMES),
        "sensor_columns": list(SENSOR_COLUMNS),
        "window_size_samples": args.window_size,
        "window_duration_seconds": args.window_size / 50,
        "stride_samples": args.stride,
        "stride_seconds": args.stride / 50,
        "seed": args.seed,
        "participant_split": split,
        "normalization_from_train_only": {"mean": mean, "std": std},
        "trainable_parameters": model.count_params(),
        "class_weight": class_weight,
        "window_counts": {
            "train": len(y_train),
            "validation": len(y_validation),
            "test": len(y_test),
        },
        "window_class_counts": {
            "train": count_labels(y_train),
            "validation": count_labels(y_validation),
            "test": count_labels(y_test),
        },
        "epochs_completed": len(history.history["loss"]),
        "history": history.history,
        "metrics": {
            "train": train_metrics,
            "validation": validation_metrics,
            "test": test_metrics,
        },
    }
    with args.metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(metadata), handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Modèle sauvegardé : {args.model_path}")
    print(f"Paramètres entraînables : {model.count_params()}")
    print(
        "Fenêtres train / validation / test : "
        f"{len(y_train)} / {len(y_validation)} / {len(y_test)}"
    )
    print(
        "Accuracy train / validation / test : "
        f"{train_metrics['accuracy']:.4f} / {validation_metrics['accuracy']:.4f} / {test_metrics['accuracy']:.4f}"
    )
    print(f"Métadonnées : {args.metadata_path}")


if __name__ == "__main__":
    main()
