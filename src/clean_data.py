#!/usr/bin/env python3
"""Nettoie les enregistrements MotionSense DeviceMotion.

Le script est volontairement indépendant de bibliothèques tierces. Il produit un
CSV propre par enregistrement et deux fichiers d'audit dans ``data/processed``.
La normalisation n'est pas effectuée ici : ses statistiques devront être
apprises exclusivement sur le sous-ensemble d'entraînement à l'étape suivante.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable


ACTIVITY_TO_ID = {
    "dws": 0,  # descente d'escaliers
    "ups": 1,  # montée d'escaliers
    "wlk": 2,  # marche
    "jog": 3,  # jogging
    "sit": 4,  # assis
    "std": 5,  # debout
}

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

RECORDING_DIRECTORY = re.compile(r"^(dws|ups|wlk|jog|sit|std)_(\d+)$")
SUBJECT_FILE = re.compile(r"^sub_(\d+)\.csv$")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/A_DeviceMotion_data"),
        help="répertoire A_DeviceMotion_data extrait de l'archive MotionSense",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="répertoire où écrire les enregistrements nettoyés et les audits",
    )
    return parser.parse_args()


def iter_recordings(input_dir: Path) -> Iterable[tuple[str, int, int, Path]]:
    """Retourne les CSV valides avec leur activité, essai et participant."""
    for activity_dir in sorted(input_dir.iterdir()):
        match = RECORDING_DIRECTORY.fullmatch(activity_dir.name)
        if not activity_dir.is_dir() or match is None:
            continue
        activity, trial_text = match.groups()
        for csv_file in sorted(activity_dir.glob("sub_*.csv")):
            subject_match = SUBJECT_FILE.fullmatch(csv_file.name)
            if subject_match is not None:
                yield activity, int(trial_text), int(subject_match.group(1)), csv_file


def to_finite_float(value: str | None) -> float | None:
    """Convertit une cellule non vide en flottant fini ou renvoie None."""
    if value is None or not value.strip():
        return None
    try:
        converted = float(value)
    except ValueError:
        return None
    return converted if math.isfinite(converted) else None


def interpolate_missing(values: list[float | None]) -> tuple[list[float], int]:
    """Comble les manques d'une colonne sans modifier le nombre d'échantillons.

    Une interpolation linéaire est appliquée entre deux valeurs valides. Les
    manques au début ou à la fin reçoivent la première ou dernière valeur valide.
    La cadence de 50 Hz est ainsi conservée, contrairement à une suppression de
    lignes qui décalerait les fenêtres temporelles.
    """
    valid_indices = [index for index, value in enumerate(values) if value is not None]
    if not valid_indices:
        raise ValueError("une colonne entière ne contient aucune valeur numérique")

    cleaned = [float(value) if value is not None else math.nan for value in values]
    first, last = valid_indices[0], valid_indices[-1]
    for index in range(first):
        cleaned[index] = cleaned[first]
    for index in range(last + 1, len(cleaned)):
        cleaned[index] = cleaned[last]

    previous = first
    for following in valid_indices[1:]:
        distance = following - previous
        if distance > 1:
            start, end = cleaned[previous], cleaned[following]
            for index in range(previous + 1, following):
                fraction = (index - previous) / distance
                cleaned[index] = start + fraction * (end - start)
        previous = following

    return cleaned, len(values) - len(valid_indices)


def clean_recording(source: Path) -> tuple[list[dict[str, float]], int, int]:
    """Lit, valide et nettoie un fichier source unique.

    Renvoie les valeurs capteur, le nombre de cellules imputées et le nombre de
    lignes contenant au moins une cellule invalide avant nettoyage.
    """
    with source.open("r", encoding="utf-8-sig", newline="") as csv_handle:
        reader = csv.DictReader(csv_handle)
        if reader.fieldnames is None:
            raise ValueError("en-tête CSV absent")
        missing_columns = set(SENSOR_COLUMNS).difference(reader.fieldnames)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"colonnes capteur absentes : {missing}")

        raw_rows = list(reader)

    if not raw_rows:
        raise ValueError("fichier CSV vide")

    columns: dict[str, list[float | None]] = {column: [] for column in SENSOR_COLUMNS}
    invalid_rows = 0
    for row in raw_rows:
        row_has_invalid_value = False
        for column in SENSOR_COLUMNS:
            value = to_finite_float(row[column])
            columns[column].append(value)
            row_has_invalid_value |= value is None
        invalid_rows += int(row_has_invalid_value)

    imputed_cells = 0
    for column, values in columns.items():
        columns[column], imputed = interpolate_missing(values)
        imputed_cells += imputed

    cleaned_rows = [
        {column: columns[column][index] for column in SENSOR_COLUMNS}
        for index in range(len(raw_rows))
    ]
    return cleaned_rows, imputed_cells, invalid_rows


def write_clean_recording(
    destination: Path,
    sensor_rows: list[dict[str, float]],
    activity: str,
    trial: int,
    subject: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_index", *SENSOR_COLUMNS, "activity", "class_id", "trial", "subject_id"]
    with destination.open("w", encoding="utf-8", newline="") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample_index, sensor_values in enumerate(sensor_rows):
            writer.writerow(
                {
                    "sample_index": sample_index,
                    **sensor_values,
                    "activity": activity,
                    "class_id": ACTIVITY_TO_ID[activity],
                    "trial": trial,
                    "subject_id": subject,
                }
            )


def main() -> None:
    args = parse_arguments()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        raise SystemExit(
            f"Répertoire introuvable : {input_dir}. "
            "Extrayez d'abord A_DeviceMotion_data.zip dans data/raw/."
        )

    clean_root = output_dir / "clean_recordings"
    if clean_root.exists():
        shutil.rmtree(clean_root)
    clean_root.mkdir(parents=True)

    manifest: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    sample_counts: Counter[str] = Counter()
    imputed_cells_total = 0
    invalid_rows_total = 0

    for activity, trial, subject, source in iter_recordings(input_dir):
        relative_source = source.relative_to(input_dir)
        destination = clean_root / relative_source
        try:
            sensor_rows, imputed_cells, invalid_rows = clean_recording(source)
        except ValueError as error:
            skipped.append({"source": str(relative_source), "reason": str(error)})
            continue

        write_clean_recording(destination, sensor_rows, activity, trial, subject)
        sample_count = len(sensor_rows)
        sample_counts[activity] += sample_count
        imputed_cells_total += imputed_cells
        invalid_rows_total += invalid_rows
        manifest.append(
            {
                "recording_path": str(destination.relative_to(output_dir)),
                "source_path": str(relative_source),
                "activity": activity,
                "class_id": ACTIVITY_TO_ID[activity],
                "trial": trial,
                "subject_id": subject,
                "sample_count": sample_count,
                "imputed_cells": imputed_cells,
                "invalid_rows_before_cleaning": invalid_rows,
            }
        )

    if not manifest:
        raise SystemExit("Aucun enregistrement MotionSense valide n'a été trouvé.")

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    audit = {
        "input_directory": str(input_dir),
        "selected_sensor_columns": list(SENSOR_COLUMNS),
        "sampling_frequency_hz": 50,
        "activity_to_id": ACTIVITY_TO_ID,
        "recordings_written": len(manifest),
        "recordings_skipped": len(skipped),
        "samples_written_by_activity": dict(sorted(sample_counts.items())),
        "samples_written_total": sum(sample_counts.values()),
        "invalid_rows_before_cleaning": invalid_rows_total,
        "imputed_cells": imputed_cells_total,
        "skipped_recordings": skipped,
    }
    with (output_dir / "cleaning_audit.json").open("w", encoding="utf-8") as audit_handle:
        json.dump(audit, audit_handle, ensure_ascii=False, indent=2)
        audit_handle.write("\n")

    print(
        "Nettoyage terminé : "
        f"{audit['recordings_written']} enregistrements, "
        f"{audit['samples_written_total']} échantillons, "
        f"{audit['imputed_cells']} cellules imputées."
    )
    print(f"Audit : {output_dir / 'cleaning_audit.json'}")


if __name__ == "__main__":
    main()
