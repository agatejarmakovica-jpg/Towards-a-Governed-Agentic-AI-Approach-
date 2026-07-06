from __future__ import annotations

from collections import Counter
from datetime import datetime
import math
import re
from typing import Any, Iterable, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .contract import ColumnContract
else:
    ColumnContract = Any
from .models import Issue
from .utils import exact_text, normalized_text, stable_id

DEFAULT_MISSING_TOKENS = {"na", "n/a", "null", "none", "missing", "?", "-"}


def infer_role(column: str, series: pd.Series) -> str:
    name = column.casefold()
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if any(token in name for token in ("date", "time", "dob")):
        return "date"
    if any(token in name for token in ("email", "mail")):
        return "email"
    if any(token in name for token in ("phone", "mobile", "telephone")):
        return "phone"
    observed = series.dropna()
    if len(observed) and observed.nunique() <= max(20, int(math.sqrt(len(observed)))):
        return "categorical"
    return "text"


def make_issue(
    issue_type: str,
    scope: str,
    column: str | None,
    row_index: int | None,
    value: Any,
    evidence_status: str,
    confidence: float,
    severity: float,
    evidence: dict[str, Any],
    automatic_action_allowed: bool = False,
) -> Issue:
    issue_id = stable_id(issue_type, scope, column, row_index, repr(value), repr(sorted(evidence.items())))
    return Issue(
        issue_id=issue_id,
        issue_type=issue_type,
        scope=scope,  # type: ignore[arg-type]
        column=column,
        row_index=row_index,
        value=value,
        evidence_status=evidence_status,  # type: ignore[arg-type]
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        severity=float(np.clip(severity, 0.0, 1.0)),
        evidence=evidence,
        automatic_action_allowed=automatic_action_allowed,
    )


def target_format_matches(value: str, target_format: str) -> bool:
    try:
        parsed = datetime.strptime(value, target_format)
        return parsed.strftime(target_format) == value
    except (TypeError, ValueError):
        return False


def normalize_for_comparison(value: Any, *, case_sensitive: bool = False) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text if case_sensitive else text.casefold()


def normalize_role_value(value: Any, *, role: str, case_sensitive: bool = False) -> str | None:
    """Normalize a value for role-aware semantic comparison without dataset-specific rules."""
    normalized = normalize_for_comparison(value, case_sensitive=case_sensitive)
    if normalized is None:
        return None
    role_name = str(role).strip().casefold()
    if role_name == "phone":
        digits = re.sub(r"\D+", "", normalized)
        return digits or normalized
    if role_name == "email":
        return normalized.casefold()
    return normalized


def effective_allowed_values(meta: ColumnContract | dict[str, Any]) -> list[Any]:
    if not isinstance(meta, dict):
        return meta.effective_allowed_values()
    approved = list(meta.get("approved_allowed_values", []) or [])
    legacy = list(meta.get("allowed_values", []) or [])
    return approved or legacy


def canonicalize_value(value: Any, meta: ColumnContract | dict[str, Any]) -> Any:
    if value is None or pd.isna(value):
        return value
    if not isinstance(meta, dict):
        mapping = meta.canonical_mapping
        case_sensitive = meta.case_sensitive
    else:
        mapping = meta.get("canonical_mapping", {}) or {}
        case_sensitive = bool(meta.get("case_sensitive", False))
    if not mapping:
        return value
    value_key = normalize_for_comparison(value, case_sensitive=case_sensitive)
    for raw, canonical in mapping.items():
        raw_key = normalize_for_comparison(raw, case_sensitive=case_sensitive)
        if value_key == raw_key:
            return canonical
    return value


def parsed_numeric_series(series: pd.Series, meta: ColumnContract | dict[str, Any]) -> tuple[pd.Series, pd.Series]:
    canonical = series.map(lambda value: canonicalize_value(value, meta))
    numeric = pd.to_numeric(canonical, errors="coerce")
    parse_failure = series.notna() & numeric.isna()
    return numeric, parse_failure


def declared_missing_token_set(global_tokens: Iterable[Any], meta: ColumnContract | dict[str, Any]) -> set[str]:
    tokens = {str(token).strip().casefold() for token in global_tokens}
    if not isinstance(meta, dict):
        semantics = meta.missing_semantics
    else:
        semantics = meta.get("missing_semantics", {}) or {}
    for token, meaning in semantics.items():
        if str(meaning).strip().casefold() == "missing":
            tokens.add(str(token).strip().casefold())
    return tokens


def sentinel_missing_values(meta: ColumnContract | dict[str, Any]) -> dict[str, str]:
    sentinel = meta.sentinel_values if not isinstance(meta, dict) else (meta.get("sentinel_values", {}) or {})
    return {str(raw): str(meaning) for raw, meaning in sentinel.items()}


def is_missing_by_contract(
    value: Any,
    *,
    global_tokens: Iterable[Any],
    meta: ColumnContract | dict[str, Any],
) -> tuple[bool, str | None]:
    if value is None or pd.isna(value):
        return True, "explicit_null"

    case_sensitive = meta.case_sensitive if not isinstance(meta, dict) else bool(meta.get("case_sensitive", False))
    key = normalize_for_comparison(value, case_sensitive=case_sensitive)

    # Explicit sentinel semantics have highest authority.
    for raw, meaning in sentinel_missing_values(meta).items():
        if key == normalize_for_comparison(raw, case_sensitive=case_sensitive):
            if meaning.strip().casefold() == "missing":
                return True, "sentinel_value"
            return False, None

    allowed = {
        normalize_for_comparison(item, case_sensitive=case_sensitive)
        for item in effective_allowed_values(meta)
    }
    # Approved categories are valid unless the same raw value is explicitly a missing sentinel.
    if key in allowed:
        return False, None

    declared = declared_missing_token_set(global_tokens, meta)
    if key is not None and key.casefold() in declared:
        return True, "declared_missing_token"
    return False, None


def numeric_sentinel_candidates(series: pd.Series) -> list[dict[str, Any]]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) < 20 or numeric.nunique() < 3:
        return []
    counts = numeric.value_counts(dropna=True)
    values = np.sort(numeric.unique().astype(float))
    adjacent = np.diff(values)
    positive_adjacent = adjacent[adjacent > 0]
    if len(positive_adjacent) == 0:
        return []
    typical_gap = float(np.median(positive_adjacent))
    if not np.isfinite(typical_gap) or typical_gap <= 0:
        return []

    results: list[dict[str, Any]] = []
    for position, candidate in (("minimum", float(values[0])), ("maximum", float(values[-1]))):
        gap = float(values[1] - values[0]) if position == "minimum" else float(values[-1] - values[-2])
        relative_gap = gap / typical_gap
        frequency = int(counts.get(candidate, 0))
        share = frequency / len(numeric)
        gap_score = min(1.0, relative_gap / 5.0)
        frequency_score = min(1.0, share / 0.05)
        score = 0.55 * gap_score + 0.25 * frequency_score + 0.20
        if score >= 0.72 and relative_gap >= 2.5 and frequency >= 2:
            results.append(
                {
                    "candidate": candidate,
                    "position": position,
                    "frequency": frequency,
                    "frequency_share": share,
                    "gap": gap,
                    "typical_adjacent_gap": typical_gap,
                    "relative_gap": relative_gap,
                    "score": score,
                }
            )
    return results


def deduplicate_issues(issues: list[Issue]) -> list[Issue]:
    unique = {issue.issue_id: issue for issue in issues}
    return sorted(
        unique.values(),
        key=lambda item: (
            item.issue_type,
            "" if item.column is None else item.column,
            -1 if item.row_index is None else item.row_index,
        ),
    )


def representation_groups(series: pd.Series, *, case_sensitive: bool = False) -> dict[str, list[tuple[int, str]]]:
    groups: dict[str, list[tuple[int, str]]] = {}
    for row_index, value in series.dropna().items():
        key = normalize_for_comparison(value, case_sensitive=case_sensitive)
        if key is None:
            continue
        groups.setdefault(key, []).append((int(row_index), exact_text(value) or ""))
    return groups


def most_common_raw(entries: list[tuple[int, str]]) -> str:
    counts = Counter(raw for _, raw in entries)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
