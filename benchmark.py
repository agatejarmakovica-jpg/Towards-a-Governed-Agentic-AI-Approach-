from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .detectors import detect_issues, infer_role
from .inject import inject_benchmark_defects, make_synthetic_health_dataset
from .metrics import detection_metrics
from .models import Issue
from .utils import stable_id, write_json


def _baseline_issue(issue_type: str, scope: str, column: str | None, row_index: int | None, value: Any = None) -> Issue:
    return Issue(
        issue_id=stable_id("baseline", issue_type, scope, column, row_index, repr(value)),
        issue_type=issue_type,
        scope=scope,  # type: ignore[arg-type]
        column=column,
        row_index=row_index,
        value=value,
        evidence_status="candidate",
        confidence=0.5,
        severity=0.5,
        evidence={"baseline": True},
        automatic_action_allowed=False,
    )


def explicit_only_baseline(df: pd.DataFrame, config: dict[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    for column in df.columns:
        for row_index in df.index[df[column].isna()]:
            issues.append(_baseline_issue("explicit_missingness", "cell", str(column), int(row_index)))
    return issues


def naive_rules_baseline(df: pd.DataFrame, config: dict[str, Any]) -> list[Issue]:
    issues = explicit_only_baseline(df, config)
    column_cfg = config.get("columns", {}) or {}

    duplicate_mask = df.duplicated(keep="first")
    for row_index in df.index[duplicate_mask]:
        issues.append(_baseline_issue("exact_duplicate_row", "row", None, int(row_index)))

    for column in df.columns:
        series = df[column]
        meta = column_cfg.get(column, {}) or {}
        role = str(meta.get("role") or infer_role(str(column), series)).casefold()

        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if len(numeric) >= 10:
                q1, q3 = numeric.quantile([0.25, 0.75])
                iqr = float(q3 - q1)
                if iqr > 0:
                    mask = (pd.to_numeric(series, errors="coerce") < q1 - 1.5 * iqr) | (pd.to_numeric(series, errors="coerce") > q3 + 1.5 * iqr)
                    for row_index in series.index[mask.fillna(False)]:
                        issues.append(_baseline_issue("numeric_extreme_candidate", "cell", str(column), int(row_index), series.loc[row_index]))

        if role == "categorical":
            normalized = series.astype("string").str.strip().str.casefold()
            for key in normalized.dropna().unique():
                rows = series.index[normalized.eq(key).fillna(False)]
                raw = series.loc[rows].astype(str).str.strip()
                canonical = raw.value_counts().index[0]
                for row_index in rows[raw.to_numpy() != canonical]:
                    issues.append(_baseline_issue("categorical_representation_inconsistency", "cell", str(column), int(row_index), series.loc[row_index]))

        if role == "date" and meta.get("target_format"):
            target = str(meta["target_format"])
            expected_pattern = r"^\d{4}-\d{2}-\d{2}$" if target == "%Y-%m-%d" else None
            if expected_pattern:
                observed = series.dropna().astype(str)
                invalid = ~observed.str.match(expected_pattern)
                for row_index in observed.index[invalid]:
                    issues.append(_baseline_issue("date_format_inconsistency", "cell", str(column), int(row_index), series.loc[row_index]))

        regex = meta.get("regex")
        if regex:
            observed = series.dropna().astype(str)
            invalid = ~observed.str.fullmatch(str(regex), na=False)
            for row_index in observed.index[invalid]:
                issues.append(_baseline_issue("format_inconsistency", "cell", str(column), int(row_index), series.loc[row_index]))

    required_fields = list(config.get("metadata_required_fields", []))
    for column in df.columns:
        meta = column_cfg.get(column, {}) or {}
        if any(meta.get(field) in (None, "", []) for field in required_fields):
            issues.append(_baseline_issue("schema_metadata_gap", "column", str(column), None))
    return issues


def broad_column_baseline(df: pd.DataFrame, config: dict[str, Any]) -> list[Issue]:
    seed_issues = detect_issues(df, config)
    broadcast: list[Issue] = []
    for issue in seed_issues:
        if issue.scope == "cell" and issue.column is not None:
            for row_index in df.index:
                broadcast.append(_baseline_issue(issue.issue_type, "cell", issue.column, int(row_index), df.at[row_index, issue.column]))
        else:
            broadcast.append(replace(issue, issue_id=stable_id("broad", issue.issue_id)))
    unique = {f"{x.scope}|{x.row_index}|{x.column}|{x.issue_type}": x for x in broadcast}
    return list(unique.values())


def _bootstrap_mean_ci(values: np.ndarray, seed: int = 4000, iterations: int = 10000) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_lower": float(np.quantile(samples, 0.025)),
        "ci_upper": float(np.quantile(samples, 0.975)),
        "standard_deviation": float(values.std(ddof=1)),
    }


def _paired_effect(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        statistic, p_value = 0.0, 1.0
        rank_biserial = 0.0
    else:
        result = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        statistic, p_value = float(result.statistic), float(result.pvalue)
        positives = int((nonzero > 0).sum())
        negatives = int((nonzero < 0).sum())
        rank_biserial = (positives - negatives) / len(nonzero)
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    cohen_dz = float(diff.mean() / sd) if sd > 1e-12 else None
    return {
        "mean_paired_difference": float(diff.mean()),
        "wilcoxon_statistic": statistic,
        "p_value": p_value,
        "paired_rank_biserial": float(rank_biserial),
        "cohen_dz": cohen_dz,
    }


def run_repeated_benchmark(output_dir: Path, runs: int = 30, rows: int = 600) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = {
        "cdqc_v5": detect_issues,
        "baseline_explicit_only": explicit_only_baseline,
        "baseline_naive_rules": naive_rules_baseline,
        "baseline_broad_column": broad_column_baseline,
    }
    run_rows: list[dict[str, Any]] = []
    per_type_rows: list[dict[str, Any]] = []

    first_artifacts: dict[str, Any] | None = None
    for run_index in range(runs):
        seed = 1000 + run_index
        clean, clean_config = make_synthetic_health_dataset(rows=rows, seed=5000 + run_index)
        corrupted, config, truth_df = inject_benchmark_defects(clean, clean_config, seed=seed)
        truth_records = truth_df.to_dict(orient="records")
        if first_artifacts is None:
            first_artifacts = {"clean": clean.copy(deep=True), "corrupted": corrupted, "config": config, "truth": truth_df}
        for method_name, method in methods.items():
            issues = method(corrupted, config)
            metrics = detection_metrics(issues, truth_records)
            run_rows.append(
                {
                    "run": run_index,
                    "seed": seed,
                    "method": method_name,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "true_positives": metrics["true_positives"],
                    "false_positives": metrics["false_positives"],
                    "false_negatives": metrics["false_negatives"],
                }
            )
            for issue_type, values in metrics["per_issue_type"].items():
                per_type_rows.append({"run": run_index, "seed": seed, "method": method_name, "issue_type": issue_type, **values})

    runs_df = pd.DataFrame(run_rows)
    per_type_df = pd.DataFrame(per_type_rows)
    runs_df.to_csv(output_dir / "benchmark_runs.csv", index=False)
    per_type_df.to_csv(output_dir / "benchmark_per_issue_type.csv", index=False)

    summaries: list[dict[str, Any]] = []
    for method_name in methods:
        subset = runs_df[runs_df["method"] == method_name]
        for metric in ("precision", "recall", "f1"):
            ci = _bootstrap_mean_ci(subset[metric].to_numpy(float), seed=4100 + len(summaries))
            summaries.append({"method": method_name, "metric": metric, **ci, "runs": runs})
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "benchmark_bootstrap_summary.csv", index=False)

    comparisons: list[dict[str, Any]] = []
    core = runs_df[runs_df["method"] == "cdqc_v5"].sort_values("run")
    for baseline in ("baseline_explicit_only", "baseline_naive_rules", "baseline_broad_column"):
        other = runs_df[runs_df["method"] == baseline].sort_values("run")
        for metric in ("precision", "recall", "f1"):
            comparisons.append(
                {
                    "comparison": f"cdqc_v5_vs_{baseline}",
                    "metric": metric,
                    **_paired_effect(core[metric].to_numpy(float), other[metric].to_numpy(float)),
                    "runs": runs,
                }
            )
    comparisons_df = pd.DataFrame(comparisons)
    comparisons_df.to_csv(output_dir / "paired_statistical_comparisons.csv", index=False)

    if first_artifacts:
        first_artifacts["clean"].to_csv(output_dir / "example_clean_reference.csv", index=False)
        first_artifacts["corrupted"].to_csv(output_dir / "example_corrupted_input.csv", index=False)
        first_artifacts["truth"].to_csv(output_dir / "example_corruption_manifest.csv", index=False)
        write_json(output_dir / "example_dataset_config.json", first_artifacts["config"])

    report = {
        "benchmark_design": {
            "runs": runs,
            "rows_before_duplicates": rows,
            "evaluation_unit": "typed cell/row/column coordinates",
            "defect_types": sorted(per_type_df["issue_type"].unique().tolist()),
            "baselines": list(methods),
            "random_seeds": [1000 + i for i in range(runs)],
            "claim_boundary": "Synthetic repeated benchmark validates detection logic under known injected defects; it does not establish external-dataset generalization.",
        },
        "summary": summary_df.to_dict(orient="records"),
        "paired_comparisons": comparisons_df.to_dict(orient="records"),
    }
    write_json(output_dir / "benchmark_report.json", report)
    return report
