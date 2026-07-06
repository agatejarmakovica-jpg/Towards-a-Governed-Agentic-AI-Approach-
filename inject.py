from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import write_json


def make_synthetic_health_dataset(rows: int = 600, seed: int = 42) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2023-01-01")
    dates = start + pd.to_timedelta(rng.integers(0, 730, size=rows), unit="D")
    df = pd.DataFrame(
        {
            "Record ID": [f"R{i:06d}" for i in range(rows)],
            "Age": rng.integers(20, 71, size=rows),
            "Glucose": rng.integers(80, 181, size=rows),
            "Insulin": rng.integers(20, 251, size=rows),
            "Cholesterol": rng.integers(120, 241, size=rows),
            "Gender": rng.choice(["Female", "Male"], size=rows),
            "Condition": rng.choice(["Control", "Case"], size=rows, p=[0.55, 0.45]),
            "Visit Date": dates.strftime("%Y-%m-%d"),
            "Phone Number": [f"+371{rng.integers(20000000, 99999999)}" for _ in range(rows)],
        }
    )
    # Legitimate heavy-tail values remain in the clean reference. They test whether
    # a statistical detector over-labels rare but uncorrupted observations.
    heavy_tail_count = max(2, rows // 250)
    heavy_rows = rng.choice(np.arange(rows), size=heavy_tail_count, replace=False)
    df.loc[heavy_rows, "Insulin"] = rng.integers(520, 760, size=heavy_tail_count)
    config = {
        "dataset_id": "synthetic_health_tabular",
        "dataset_metadata": {
            "title": "Synthetic health-themed tabular benchmark",
            "description": "Representative data-science benchmark; not a clinical study.",
            "provenance": "Generated deterministically by C-DQC v4 benchmark code.",
            "version": "1.0",
            "license": "CC0 synthetic benchmark",
            "intended_reuse": "Evaluation of tabular data-quality detection methods.",
            "privacy_status": "Synthetic; no personal data."
        },
        "missing_tokens": ["NA", "N/A", "NULL", "?"],
        "metadata_required_fields": ["role", "description"],
        "columns": {
            "Record ID": {"role": "text", "description": "Synthetic record identifier"},
            "Age": {"role": "numeric", "description": "Synthetic age-like benchmark variable"},
            "Glucose": {"role": "numeric", "description": "Synthetic measurement benchmark variable"},
            "Insulin": {"role": "numeric", "description": "Synthetic measurement benchmark variable"},
            "Cholesterol": {"role": "numeric", "description": "Synthetic measurement benchmark variable"},
            "Gender": {"role": "categorical", "description": "Synthetic category benchmark variable", "allowed_values": ["Female", "Male"]},
            "Condition": {"role": "categorical", "description": "Synthetic binary case-control benchmark label", "allowed_values": ["Control", "Case"]},
            "Visit Date": {"role": "date", "description": "Synthetic visit date", "target_format": "%Y-%m-%d", "allow_format_normalization": False},
            "Phone Number": {"role": "phone", "description": "Synthetic phone-format benchmark variable", "regex": r"\+371\d{8}"},
        },
    }
    return df, config


def inject_benchmark_defects(
    clean_df: pd.DataFrame,
    clean_config: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    difficulty = seed % 3  # 0=easy, 1=medium, 2=hard
    df = clean_df.copy(deep=True)
    config = deepcopy(clean_config)
    truth: list[dict[str, Any]] = []
    available = np.arange(len(df))

    def choose(count: int, excluded: set[int] | None = None) -> np.ndarray:
        excluded = excluded or set()
        pool = np.array([x for x in available if int(x) not in excluded], dtype=int)
        return rng.choice(pool, size=count, replace=False)

    used_by_column: dict[str, set[int]] = {column: set() for column in df.columns}

    # 1) Explicit missingness.
    rows = choose(int(rng.integers(24, 37)))
    for row in rows:
        original = df.at[int(row), "Glucose"]
        df.at[int(row), "Glucose"] = np.nan
        used_by_column["Glucose"].add(int(row))
        truth.append({"issue_type": "explicit_missingness", "scope": "cell", "row_index": int(row), "column": "Glucose", "original_value": original, "corrupted_value": None})

    # 2) Disguised missingness sentinel with no sentinel declaration in metadata.
    rows = choose(int(rng.integers(8, 27)), used_by_column["Insulin"])
    sentinel_value = -999 if difficulty == 0 else (0 if difficulty == 1 else 18)
    for row in rows:
        original = df.at[int(row), "Insulin"]
        df.at[int(row), "Insulin"] = sentinel_value
        used_by_column["Insulin"].add(int(row))
        truth.append({"issue_type": "disguised_missingness_candidate", "scope": "cell", "row_index": int(row), "column": "Insulin", "original_value": original, "corrupted_value": sentinel_value})

    # 3) Numeric extremes with unique values so they are not a repeated sentinel cluster.
    rows = choose(int(rng.integers(10, 23)), used_by_column["Age"])
    for offset, row in enumerate(rows):
        original = df.at[int(row), "Age"]
        corrupted = (900 + offset) if difficulty == 0 else ((150 + offset) if difficulty == 1 else (96 + offset % 8))
        df.at[int(row), "Age"] = corrupted
        used_by_column["Age"].add(int(row))
        truth.append({"issue_type": "numeric_extreme_candidate", "scope": "cell", "row_index": int(row), "column": "Age", "original_value": original, "corrupted_value": corrupted})

    # 4) Categorical representation variants only, preserving category meaning.
    rows = choose(int(rng.integers(18, 34)), used_by_column["Gender"])
    if difficulty == 2:
        config["columns"]["Gender"].pop("allowed_values", None)
    for index, row in enumerate(rows):
        original = str(df.at[int(row), "Gender"])
        if difficulty == 2:
            corrupted = "F" if original == "Female" else "M"
        else:
            corrupted = f" {original.casefold()} " if index % 2 == 0 else original.upper()
        df.at[int(row), "Gender"] = corrupted
        used_by_column["Gender"].add(int(row))
        truth.append({"issue_type": "categorical_representation_inconsistency", "scope": "cell", "row_index": int(row), "column": "Gender", "original_value": original, "corrupted_value": corrupted})

    # 5) Date representation inconsistency against a declared target format.
    rows = choose(int(rng.integers(20, 38)), used_by_column["Visit Date"])
    for row in rows:
        original = str(df.at[int(row), "Visit Date"])
        corrupted = pd.Timestamp(original).strftime("%d/%m/%Y")
        df.at[int(row), "Visit Date"] = corrupted
        used_by_column["Visit Date"].add(int(row))
        truth.append({"issue_type": "date_format_inconsistency", "scope": "cell", "row_index": int(row), "column": "Visit Date", "original_value": original, "corrupted_value": corrupted})

    if difficulty == 2:
        config["columns"]["Visit Date"].pop("target_format", None)

    # 6) Phone-format violations against a declared regular expression.
    rows = choose(int(rng.integers(16, 31)), used_by_column["Phone Number"])
    for row in rows:
        original = str(df.at[int(row), "Phone Number"])
        corrupted = original.replace("+371", "00371-")
        df.at[int(row), "Phone Number"] = corrupted
        used_by_column["Phone Number"].add(int(row))
        truth.append({"issue_type": "format_inconsistency", "scope": "cell", "row_index": int(row), "column": "Phone Number", "original_value": original, "corrupted_value": corrupted})

    # 7) Exact duplicate rows, copied only from rows without injected cell defects.
    protected = set().union(*used_by_column.values())
    source_rows = choose(int(rng.integers(12, 26)), protected)
    duplicates = df.loc[source_rows].copy(deep=True)
    start_index = len(df)
    duplicates.index = np.arange(start_index, start_index + len(duplicates))
    df = pd.concat([df, duplicates], axis=0)
    for row in duplicates.index:
        truth.append({"issue_type": "exact_duplicate_row", "scope": "row", "row_index": int(row), "column": None, "original_value": None, "corrupted_value": "duplicate_row"})

    # 8) Metadata/schema gaps.
    for column in ["Insulin", "Phone Number"]:
        original = config["columns"][column].pop("description")
        truth.append({"issue_type": "schema_metadata_gap", "scope": "column", "row_index": None, "column": column, "original_value": original, "corrupted_value": None})

    return df, config, pd.DataFrame(truth)


def write_injected_benchmark(output_dir: Path, seed: int, rows: int = 600) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    clean, clean_config = make_synthetic_health_dataset(rows=rows, seed=101)
    corrupted, corrupted_config, truth = inject_benchmark_defects(clean, clean_config, seed=seed)
    clean_path = output_dir / "clean_reference.csv"
    corrupted_path = output_dir / "corrupted_input.csv"
    truth_path = output_dir / "corruption_manifest.csv"
    config_path = output_dir / "dataset_config.json"
    clean.to_csv(clean_path, index=False)
    corrupted.to_csv(corrupted_path, index=False)
    truth.to_csv(truth_path, index=False)
    write_json(config_path, corrupted_config)
    return {
        "clean_reference": str(clean_path),
        "corrupted_input": str(corrupted_path),
        "corruption_manifest": str(truth_path),
        "dataset_config": str(config_path),
    }
