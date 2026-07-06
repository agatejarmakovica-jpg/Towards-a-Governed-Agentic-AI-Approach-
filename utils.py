from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd


def stable_id(*parts: Any, prefix: str = "ISSUE") -> str:
    payload = "|".join("" if part is None else str(part) for part in parts)
    return f"{prefix}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=json_default)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    raise TypeError(type(value).__name__)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def normalized_text(value: Any) -> str | None:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return " ".join(str(value).strip().split()).casefold()


def exact_text(value: Any) -> str | None:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def f1_score(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return cleaned.strip("_") or "dataset"


def zip_directory(source_dir: Path, destination_without_suffix: Path) -> Path:
    destination_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    archive = shutil.make_archive(
        str(destination_without_suffix),
        "zip",
        root_dir=source_dir.parent,
        base_dir=source_dir.name,
    )
    return Path(archive)
