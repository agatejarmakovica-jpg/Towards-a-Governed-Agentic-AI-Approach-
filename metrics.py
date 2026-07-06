from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .models import AgentResult, Issue
from .utils import f1_score


def _clamp(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def truth_key(record: dict[str, Any]) -> str:
    scope = str(record.get("scope", "cell"))
    row = record.get("row_index")
    column = record.get("column")
    if pd.isna(column):
        column = None
    issue_type = str(record.get("issue_type"))
    if pd.isna(row):
        row = None
    elif isinstance(row, (float, np.floating)) and float(row).is_integer():
        row = int(row)
    return f"{scope}|{row}|{column}|{issue_type}"


def issue_truth_key(issue: Issue) -> str:
    return f"{issue.scope}|{issue.row_index}|{issue.column}|{issue.issue_type}"


def detection_metrics(
    issues: Iterable[Issue],
    truth_records: list[dict[str, Any]],
    *,
    df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    predicted_issues = [issue for issue in issues if issue.scope in {"cell", "row", "column"}]
    predicted = {issue_truth_key(issue) for issue in predicted_issues}
    truth = {truth_key(record) for record in truth_records}
    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    truth_by_type: dict[str, set[str]] = defaultdict(set)
    pred_by_type: dict[str, set[str]] = defaultdict(set)
    for record in truth_records:
        truth_by_type[str(record.get("issue_type"))].add(truth_key(record))
    for issue in predicted_issues:
        pred_by_type[issue.issue_type].add(issue_truth_key(issue))

    all_types = sorted(set(truth_by_type) | set(pred_by_type))
    per_type: dict[str, dict[str, Any]] = {}
    macro_f1_values: list[float] = []
    for issue_type in all_types:
        t = truth_by_type[issue_type]
        p = pred_by_type[issue_type]
        type_tp = len(t & p)
        type_fp = len(p - t)
        type_fn = len(t - p)
        type_precision = type_tp / (type_tp + type_fp) if type_tp + type_fp else 0.0
        type_recall = type_tp / (type_tp + type_fn) if type_tp + type_fn else 0.0
        type_f1 = f1_score(type_precision, type_recall)
        macro_f1_values.append(type_f1)
        per_type[issue_type] = {
            "true_positives": type_tp,
            "false_positives": type_fp,
            "false_negatives": type_fn,
            "precision": type_precision,
            "recall": type_recall,
            "f1": type_f1,
            "truth_count": len(t),
            "predicted_count": len(p),
        }

    true_negatives: int | None = None
    specificity: float | None = None
    false_positive_rate: float | None = None
    mcc: float | None = None
    if df is not None:
        # Scientific unit: typed coordinates. Cell issue types are evaluated over every
        # cell, row issue types over every row, and column issue types over every column.
        cell_types = {
            str(record.get("issue_type"))
            for record in truth_records
            if str(record.get("scope", "cell")) == "cell"
        } | {issue.issue_type for issue in predicted_issues if issue.scope == "cell"}
        row_types = {
            str(record.get("issue_type"))
            for record in truth_records
            if str(record.get("scope")) == "row"
        } | {issue.issue_type for issue in predicted_issues if issue.scope == "row"}
        column_types = {
            str(record.get("issue_type"))
            for record in truth_records
            if str(record.get("scope")) == "column"
        } | {issue.issue_type for issue in predicted_issues if issue.scope == "column"}
        universe = int(df.size) * len(cell_types) + len(df) * len(row_types) + len(df.columns) * len(column_types)
        true_negatives = max(0, universe - tp - fp - fn)
        specificity = true_negatives / (true_negatives + fp) if true_negatives + fp else 0.0
        false_positive_rate = fp / (fp + true_negatives) if fp + true_negatives else 0.0
        denominator = np.sqrt((tp + fp) * (tp + fn) * (true_negatives + fp) * (true_negatives + fn))
        mcc = float((tp * true_negatives - fp * fn) / denominator) if denominator else 0.0

    micro_f1 = f1_score(precision, recall)
    macro_f1 = float(np.mean(macro_f1_values)) if macro_f1_values else 0.0
    return {
        "evaluation_unit": "typed cell/row/column coordinates",
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": true_negatives,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "f1": micro_f1,
        "micro_F1": micro_f1,
        "macro_F1": macro_f1,
        "MCC": mcc,
        "truth_count": len(truth),
        "predicted_count": len(predicted),
        "per_issue_type": per_type,
    }


def exact_reference_metrics(input_df: pd.DataFrame, reference_df: pd.DataFrame) -> dict[str, Any]:
    """Backward-compatible exact positional reference comparison.

    The AccuracyAgent provides the contract-aware key alignment and tolerance metrics.
    """
    common_columns = [column for column in input_df.columns if column in reference_df.columns]
    common_rows = min(len(input_df), len(reference_df))
    left = input_df.iloc[:common_rows][common_columns].reset_index(drop=True)
    right = reference_df.iloc[:common_rows][common_columns].reset_index(drop=True)
    equal = pd.DataFrame(False, index=left.index, columns=common_columns)
    for column in common_columns:
        l = left[column]
        r = right[column]
        both_missing = l.isna() & r.isna()
        if pd.api.types.is_numeric_dtype(l) and pd.api.types.is_numeric_dtype(r):
            values = np.isclose(
                pd.to_numeric(l, errors="coerce").to_numpy(float),
                pd.to_numeric(r, errors="coerce").to_numpy(float),
                equal_nan=False,
            )
            equal[column] = both_missing | pd.Series(values, index=left.index)
        else:
            equal[column] = both_missing | l.astype("string").str.strip().eq(r.astype("string").str.strip()).fillna(False)
    wrong = ~equal
    wrong_count = int(wrong.to_numpy().sum())
    total = int(wrong.size)
    per_column = {
        column: {
            "wrong_cells": int(wrong[column].sum()),
            "exact_cell_accuracy": 1.0 - int(wrong[column].sum()) / max(1, common_rows),
        }
        for column in common_columns
    }
    return {
        "aligned_rows": common_rows,
        "aligned_columns": len(common_columns),
        "total_cells": total,
        "wrong_cells": wrong_count,
        "exact_cell_accuracy": 1.0 - wrong_count / max(1, total),
        "per_column": per_column,
    }


def aggregate_scientific_metrics(
    df: pd.DataFrame,
    completeness: AgentResult,
    accuracy: AgentResult,
    reuse: AgentResult,
    contract: Any,
    all_issues: list[Issue],
) -> dict[str, Any]:
    c = float(completeness.metrics.get("critical_variable_completeness", completeness.metrics.get("overall_completeness", 0.0)))
    a_support = float(accuracy.metrics.get("contract_validity_support", 0.0))
    r = float(reuse.metrics.get("reuse_readiness_support", 0.0))

    reference = accuracy.metrics.get("reference_metrics") or {}
    manifest = accuracy.metrics.get("manifest_detection_metrics")
    if reference.get("state") == "COMPUTED" and manifest is not None:
        evidence_mode = "REFERENCE_AND_MANIFEST"
        a_validated = float(reference.get("tolerance_adjusted_accuracy", 0.0))
        validated_basis = "clean_reference_accuracy"
    elif reference.get("state") == "COMPUTED":
        evidence_mode = "CLEAN_REFERENCE"
        a_validated = float(reference.get("tolerance_adjusted_accuracy", 0.0))
        validated_basis = "clean_reference_accuracy"
    elif manifest is not None:
        evidence_mode = "CORRUPTION_MANIFEST"
        a_validated = float(manifest.get("f1", 0.0))
        validated_basis = "manifest_detection_F1"
    else:
        evidence_mode = "CONTRACT_ONLY"
        a_validated = None
        validated_basis = None

    policy = contract.decision_policy
    dqi_support = _clamp(
        policy.completeness_weight * c
        + policy.accuracy_weight * a_support
        + policy.reuse_weight * r
    )
    dqi_validated = None
    if a_validated is not None:
        dqi_validated = _clamp(
            policy.completeness_weight * c
            + policy.accuracy_weight * a_validated
            + policy.reuse_weight * r
        )

    decision_index = dqi_validated if dqi_validated is not None else dqi_support
    if decision_index >= policy.ready_threshold:
        verdict = "READY"
    elif decision_index >= policy.conditional_threshold:
        verdict = "CONDITIONALLY_READY"
    else:
        verdict = "NOT_READY"

    verdict_reasons: list[str] = []
    if policy.require_approved_contract_for_ready and contract.approval.status != "approved":
        verdict = "NOT_READY"
        verdict_reasons.append("The effective data contract is not approved.")
    direct_identifier_count = int(round(float(reuse.metrics.get("direct_identifier_exposure", 0.0)) * max(1, len(df.columns))))
    if direct_identifier_count > policy.max_direct_identifier_columns_for_ready:
        verdict = "NOT_READY"
        verdict_reasons.append("Direct identifier exposure exceeds the ready-state policy.")
    missing_rate = 1.0 - float(completeness.metrics.get("overall_completeness", 0.0))
    if missing_rate > policy.max_confirmed_missing_rate_for_ready:
        if verdict == "READY":
            verdict = "CONDITIONALLY_READY"
        verdict_reasons.append("Confirmed missing-like rate exceeds the ready-state policy.")
    violation_rate = 1.0 - a_support
    if violation_rate > policy.max_confirmed_accuracy_violation_rate_for_ready:
        if verdict == "READY":
            verdict = "CONDITIONALLY_READY"
        verdict_reasons.append("Confirmed contract-violation rate exceeds the ready-state policy.")
    for blocker in reuse.metrics.get("blocking_conditions", []):
        if blocker not in verdict_reasons:
            verdict_reasons.append(blocker)
    if not verdict_reasons:
        verdict_reasons.append("No blocking or conditional policy gate was triggered.")

    by_type = Counter(issue.issue_type for issue in all_issues)
    by_status = Counter(issue.evidence_status for issue in all_issues)
    by_scope = Counter(issue.scope for issue in all_issues)
    metric_availability = [
        *[item.to_dict() for item in completeness.metric_availability],
        *[item.to_dict() for item in accuracy.metric_availability],
        *[item.to_dict() for item in reuse.metric_availability],
    ]
    return {
        "metric_contract": {
            "completeness_support": "Contract-driven confirmed completeness score.",
            "confirmed_validity_support": "Support score from confirmed contract-backed violations; not ground-truth accuracy.",
            "accuracy_support": "Backward-compatible alias for contract_validity_support unless validated evidence is supplied.",
            "validated_accuracy": "Reference accuracy or manifest detection F1, available only in a validated evidence mode.",
            "reuse_readiness_support": "Weighted documentation, schema, standardization, privacy, and machine-readability score.",
            "DQI_support": "0.30*C + 0.40*contract-validity support + 0.30*R.",
            "DQI_validated": "0.30*C + 0.40*validated A + 0.30*R; not available in CONTRACT_ONLY mode.",
        },
        "evidence_mode": evidence_mode,
        "shape": {"rows": int(len(df)), "columns": int(len(df.columns)), "cells": int(df.size)},
        "C": c,
        "A_support": a_support,
        "A_validated": a_validated,
        "A_validated_basis": validated_basis,
        "R": r,
        "DQI_support": dqi_support,
        "DQI_validated": dqi_validated,
        "completeness_support": c,
        "confirmed_validity_support": a_support,
        "accuracy_support": a_validated if a_validated is not None else a_support,
        "accuracy_support_basis": validated_basis or "contract_confirmed_violations",
        "reuse_readiness_support": r,
        "quality_support_index": decision_index,
        "verdict": verdict,
        "verdict_reasons": verdict_reasons,
        "decision_policy": {
            "weights": {
                "completeness": policy.completeness_weight,
                "accuracy": policy.accuracy_weight,
                "reuse": policy.reuse_weight,
            },
            "ready_threshold": policy.ready_threshold,
            "conditional_threshold": policy.conditional_threshold,
        },
        "human_review_required": len(all_issues),
        "candidate_or_supported_review_items": sum(
            issue.evidence_status in {"candidate", "supported"} for issue in all_issues
        ),
        "issue_counts_by_type": dict(sorted(by_type.items())),
        "issue_counts_by_status": dict(sorted(by_status.items())),
        "issue_counts_by_scope": dict(sorted(by_scope.items())),
        "metric_availability": metric_availability,
        "agent_metrics": {
            completeness.agent_name: completeness.metrics,
            accuracy.agent_name: accuracy.metrics,
            reuse.agent_name: reuse.metrics,
        },
    }


def quality_support_metrics(df: pd.DataFrame, issues: list[Issue], config: dict[str, Any]) -> dict[str, Any]:
    """Legacy compatibility wrapper.

    New production execution uses aggregate_scientific_metrics with independent
    agent results. This wrapper keeps external callers functional.
    """
    from .contract import DatasetContract
    from .completeness_agent import CompletenessAgent
    from .accuracy_agent import AccuracyAgent
    from .reuse_agent import ReuseAgent

    raw = dict(config or {})
    raw.setdefault("dataset_id", "dataset")
    raw.setdefault("columns", {})
    contract = DatasetContract.model_validate(raw)
    c = CompletenessAgent().run(df, contract, evidence_mode="CONTRACT_ONLY")
    a = AccuracyAgent().run(df, contract, evidence_mode="CONTRACT_ONLY")
    r = ReuseAgent().run(df, contract, evidence_mode="CONTRACT_ONLY")
    return aggregate_scientific_metrics(df, c, a, r, contract, issues)
