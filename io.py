from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

SUPPORTED_TABLE_SUFFIXES = {".csv", ".xlsx", ".xls", ".parquet"}
DEFAULT_MAX_FILE_SIZE_MB = 250


def validate_input_file(path: Path, max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.casefold() not in SUPPORTED_TABLE_SUFFIXES:
        raise ValueError(
            f"Unsupported input format '{path.suffix}'. Supported formats: "
            f"{', '.join(sorted(SUPPORTED_TABLE_SUFFIXES))}."
        )
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_file_size_mb:
        raise ValueError(f"Input file is {size_mb:.1f} MB; configured limit is {max_file_size_mb} MB.")


def read_table(path: Path, *, max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB) -> pd.DataFrame:
    validate_input_file(path, max_file_size_mb=max_file_size_mb)
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        df = pd.read_csv(path, low_memory=False)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:  # pragma: no cover - guarded by validate_input_file
        raise ValueError(f"Unsupported input format: {suffix}")

    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].astype(str).tolist()
        raise ValueError(f"Duplicate column names are not supported: {duplicates}")
    if len(df.columns) == 0:
        raise ValueError("The dataset contains no columns.")
    return df


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
    elif suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {suffix}")


def dataset_profile(df: pd.DataFrame) -> dict[str, Any]:
    total_cells = int(df.size)
    numeric_columns = [str(column) for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
    text_columns = [str(column) for column in df.columns if column not in numeric_columns]
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "cells": total_cells,
        "missing_cells": int(df.isna().to_numpy().sum()),
        "duplicate_rows": int(df.duplicated(keep=False).sum()),
        "numeric_columns": numeric_columns,
        "non_numeric_columns": text_columns,
        "memory_bytes": int(df.memory_usage(deep=True).sum()),
    }
