from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import operator
import re
from typing import Any

import numpy as np
import pandas as pd

from .agent_utils import (
    canonicalize_value,
    deduplicate_issues,
    effective_allowed_values,
    is_missing_by_contract,
    make_issue,
    most_common_raw,
    normalize_for_comparison,
    normalize_role_value,
    numeric_sentinel_candidates,
    parsed_numeric_series,
    representation_groups,
    target_format_matches,
)
from .contract import CrossFieldRule, DatasetContract
from .models import AgentRecommendation, AgentResult, MetricAvailability
from .utils import stable_id


class AccuracyAgent:
    name = "AccuracyAgent"

    def run(
        self,
        df: pd.DataFrame,
        contract: DatasetContract,
        *,
        evidence_mode: str,
        reference_df: pd.DataFrame | None = None,
        truth_manifest: pd.DataFrame | None = None,
    ) -> AgentResult:
        issues = []
        total_nonmissing = 0
        parse_attempts = 0
        parse_successes = 0
        range_attempts = 0
        range_successes = 0
        categorical_attempts = 0
        categorical_successes = 0
        format_attempts = 0
        format_successes = 0
        date_attempts = 0
        date_successes = 0
        cross_attempts = 0
        cross_successes = 0
        confirmed_contract_cells: set[tuple[int, str]] = set()
        candidate_anomaly_cells: set[tuple[int, str]] = set()
        warnings: list[str] = []

        # Exact duplicate rows are accuracy/consistency findings.
        duplicate_mask = df.duplicated(keep="first")
        duplicate_count = 0
        for position, row_index in enumerate(df.index):
            if not bool(duplicate_mask.loc[row_index]):
                continue
            duplicate_count += 1
            prior = df.iloc[:position]
            current = df.iloc[position]
            matches = prior.eq(current).all(axis=1)
            duplicate_of = int(matches.index[matches][0]) if matches.any() else None
            issues.append(
                make_issue(
                    "exact_duplicate_row",
                    "row",
                    None,
                    int(row_index),
                    None,
                    "confirmed",
                    1.0,
                    0.7,
                    {"duplicate_of": duplicate_of},
                )
            )

        numeric_series_by_column: dict[str, pd.Series] = {}

        for column in df.columns:
            name = str(column)
            series = df[column]
            meta = contract.columns.get(name)
            if meta is None:
                continue

            nonmissing_mask = series.map(
                lambda value: not is_missing_by_contract(
                    value,
                    global_tokens=contract.missing_tokens,
                    meta=meta,
                )[0]
            )
            total_nonmissing += int(nonmissing_mask.sum())
            canonical = series.map(lambda value: canonicalize_value(value, meta))

            if meta.role == "numeric":
                numeric, _ = parsed_numeric_series(series, meta)
                numeric_series_by_column[name] = numeric
                attempted = nonmissing_mask
                failed = attempted & numeric.isna()
                parse_attempts += int(attempted.sum())
                parse_successes += int((attempted & numeric.notna()).sum())
                for row_index in series.index[failed]:
                    issues.append(
                        make_issue(
                            "numeric_parse_failure",
                            "cell",
                            name,
                            int(row_index),
                            series.loc[row_index],
                            "confirmed",
                            1.0,
                            0.8,
                            {
                                "declared_role": "numeric",
                                "canonical_mapping_applied": bool(meta.canonical_mapping),
                            },
                        )
                    )
                    confirmed_contract_cells.add((int(row_index), name))

                if meta.valid_range is not None:
                    low, high = map(float, meta.valid_range)
                    assessable = attempted & numeric.notna()
                    valid = assessable & numeric.between(low, high, inclusive="both")
                    range_attempts += int(assessable.sum())
                    range_successes += int(valid.sum())
                    invalid = assessable & ~numeric.between(low, high, inclusive="both")
                    for row_index in series.index[invalid]:
                        issues.append(
                            make_issue(
                                "range_violation",
                                "cell",
                                name,
                                int(row_index),
                                series.loc[row_index],
                                "confirmed",
                                1.0,
                                0.8,
                                {
                                    "valid_range": [low, high],
                                    "parsed_value": float(numeric.loc[row_index]),
                                    "source": "approved_contract",
                                },
                            )
                        )
                        confirmed_contract_cells.add((int(row_index), name))

            allowed_values = effective_allowed_values(meta)
            if meta.role == "categorical" or allowed_values:
                allowed_map = {
                    normalize_for_comparison(value, case_sensitive=meta.case_sensitive): value
                    for value in allowed_values
                }
                groups = representation_groups(series[nonmissing_mask], case_sensitive=meta.case_sensitive)
                if allowed_values:
                    categorical_attempts += int(nonmissing_mask.sum())
                    for row_index in series.index[nonmissing_mask]:
                        key = normalize_for_comparison(canonical.loc[row_index], case_sensitive=meta.case_sensitive)
                        if key in allowed_map:
                            categorical_successes += 1
                        else:
                            issues.append(
                                make_issue(
                                    "categorical_invalid_value",
                                    "cell",
                                    name,
                                    int(row_index),
                                    series.loc[row_index],
                                    "confirmed",
                                    1.0,
                                    0.7,
                                    {
                                        "approved_allowed_values": allowed_values,
                                        "dirty_observations_not_used_as_dictionary": True,
                                    },
                                )
                            )
                            confirmed_contract_cells.add((int(row_index), name))

                for key, entries in groups.items():
                    raw_values = sorted({raw for _, raw in entries})
                    if len(raw_values) <= 1:
                        continue
                    approved_canonical = allowed_map.get(key)
                    canonical_raw = str(approved_canonical) if approved_canonical is not None else most_common_raw(entries)
                    for row_index, raw in entries:
                        if raw == canonical_raw:
                            continue
                        issues.append(
                            make_issue(
                                "categorical_representation_inconsistency",
                                "cell",
                                name,
                                int(row_index),
                                series.loc[row_index],
                                "supported",
                                0.90,
                                0.4,
                                {
                                    "canonical": canonical_raw,
                                    "variant_group": raw_values,
                                    "approved_mapping_available": bool(meta.canonical_mapping),
                                },
                                automatic_action_allowed=bool(meta.canonical_mapping),
                            )
                        )

            if meta.role == "date":
                observed = canonical[nonmissing_mask].astype("string").str.strip()
                parsed = pd.to_datetime(observed, errors="coerce", format="mixed")
                date_attempts += int(len(observed))
                date_successes += int(parsed.notna().sum())
                for row_index in observed.index[parsed.isna()]:
                    issues.append(
                        make_issue(
                            "unparseable_date",
                            "cell",
                            name,
                            int(row_index),
                            series.loc[row_index],
                            "confirmed",
                            1.0,
                            0.7,
                            {"parser": "pandas_mixed", "declared_role": "date"},
                        )
                    )
                    confirmed_contract_cells.add((int(row_index), name))

                if meta.target_format:
                    parseable = observed.index[parsed.notna()]
                    format_attempts += len(parseable)
                    for row_index in parseable:
                        value = str(observed.loc[row_index])
                        if target_format_matches(value, meta.target_format):
                            format_successes += 1
                        else:
                            issues.append(
                                make_issue(
                                    "date_format_inconsistency",
                                    "cell",
                                    name,
                                    int(row_index),
                                    series.loc[row_index],
                                    "confirmed",
                                    1.0,
                                    0.35,
                                    {"target_format": meta.target_format, "source": "approved_contract"},
                                    automatic_action_allowed=meta.allow_format_normalization,
                                )
                            )
                            confirmed_contract_cells.add((int(row_index), name))
                else:
                    patterns = observed[parsed.notna()].map(lambda value: re.sub(r"\d", "#", str(value))).nunique()
                    if patterns > 1:
                        issues.append(
                            make_issue(
                                "date_representation_diversity",
                                "column",
                                name,
                                None,
                                None,
                                "candidate",
                                0.60,
                                0.20,
                                {"pattern_count": int(patterns), "not_a_cell_level_error": True},
                            )
                        )

            if meta.regex:
                observed = canonical[nonmissing_mask].astype("string")
                valid = observed.str.fullmatch(meta.regex, na=False)
                format_attempts += int(len(observed))
                format_successes += int(valid.sum())
                for row_index in observed.index[~valid]:
                    issues.append(
                        make_issue(
                            "format_inconsistency",
                            "cell",
                            name,
                            int(row_index),
                            series.loc[row_index],
                            "confirmed",
                            1.0,
                            0.6,
                            {"regex": meta.regex, "source": "approved_contract"},
                        )
                    )
                    confirmed_contract_cells.add((int(row_index), name))

            if meta.uniqueness_required:
                duplicated = canonical.notna() & canonical.duplicated(keep=False)
                for row_index in series.index[duplicated]:
                    issues.append(
                        make_issue(
                            "uniqueness_violation",
                            "cell",
                            name,
                            int(row_index),
                            series.loc[row_index],
                            "confirmed",
                            1.0,
                            0.8,
                            {"uniqueness_required": True},
                        )
                    )
                    confirmed_contract_cells.add((int(row_index), name))

        # Approved cross-field rules are evaluated without arbitrary code execution.
        cross_issues, cross_stats = self._evaluate_cross_field_rules(df, contract)
        issues.extend(cross_issues)
        cross_attempts += cross_stats["attempts"]
        cross_successes += cross_stats["successes"]
        cross_not_evaluable = cross_stats.get("not_evaluable", 0)
        for item in cross_issues:
            # Evidence-status separation: only confirmed rules may contribute to
            # confirmed contract-violation cells. Candidate and supported rule
            # findings stay in the review queue and never inflate confirmed rates.
            if item.scope == "row" and item.row_index is not None and item.evidence_status == "confirmed":
                confirmed_contract_cells.add((int(item.row_index), "<cross_field>"))

        anomaly_issues = self._detect_anomalies(df, contract, numeric_series_by_column)
        issues.extend(anomaly_issues)
        candidate_anomaly_cells.update(
            (int(issue.row_index), str(issue.column))
            for issue in anomaly_issues
            if issue.row_index is not None and issue.column is not None
        )

        reference_metrics = None
        if reference_df is not None:
            reference_metrics, reference_issues, reference_warnings = self._reference_assessment(df, reference_df, contract)
            issues.extend(reference_issues)
            warnings.extend(reference_warnings)

        issues = deduplicate_issues(issues)
        confirmed_contract_issue_cells = {
            (int(issue.row_index), str(issue.column))
            for issue in issues
            if issue.scope == "cell"
            and issue.row_index is not None
            and issue.column is not None
            and issue.evidence_status == "confirmed"
            and issue.issue_type != "reference_cell_mismatch"
        }
        assessable_contract_cells = max(1, total_nonmissing)
        contract_validity_support = 1.0 - len(confirmed_contract_issue_cells) / assessable_contract_cells

        metrics: dict[str, Any] = {
            "contract_validity_support": float(np.clip(contract_validity_support, 0.0, 1.0)),
            "type_parse_success_rate": parse_successes / parse_attempts if parse_attempts else None,
            "range_validity_rate": range_successes / range_attempts if range_attempts else None,
            "categorical_validity_rate": categorical_successes / categorical_attempts if categorical_attempts else None,
            "format_validity_rate": format_successes / format_attempts if format_attempts else None,
            "date_parse_success_rate": date_successes / date_attempts if date_attempts else None,
            "cross_field_consistency_rate": cross_successes / cross_attempts if cross_attempts else None,
            "cross_field_not_evaluable_count": cross_not_evaluable,
            "duplicate_free_rate": 1.0 - duplicate_count / max(1, len(df)),
            "candidate_anomaly_burden": len(candidate_anomaly_cells) / max(1, int(df.size)),
            "confirmed_contract_violation_cells": len(confirmed_contract_issue_cells),
            "reference_metrics": reference_metrics,
        }

        if reference_metrics is not None and reference_metrics.get("state") == "COMPUTED":
            metrics["accuracy_basis"] = "clean_reference_and_contract"
            metrics["validated_accuracy"] = reference_metrics["tolerance_adjusted_accuracy"]
        else:
            metrics["accuracy_basis"] = "contract_confirmed_violations"
            metrics["validated_accuracy"] = None

        if truth_manifest is not None:
            from .metrics import detection_metrics

            accuracy_manifest_types = {
                "numeric_parse_failure",
                "range_violation",
                "categorical_invalid_value",
                "categorical_representation_inconsistency",
                "unparseable_date",
                "date_format_inconsistency",
                "date_representation_diversity",
                "format_inconsistency",
                "uniqueness_violation",
                "cross_field_rule_violation",
                "numeric_anomaly_candidate",
                "numeric_extreme_candidate",
                "exact_duplicate_row",
            }
            manifest_records = [
                record
                for record in truth_manifest.to_dict(orient="records")
                if str(record.get("issue_type")) in accuracy_manifest_types
            ]
            manifest_predictions = [
                issue for issue in issues if issue.issue_type in accuracy_manifest_types
            ]
            metrics["manifest_detection_metrics"] = detection_metrics(
                manifest_predictions,
                manifest_records,
                df=df,
            )
        else:
            metrics["manifest_detection_metrics"] = None

        recommendations = self._recommendations(issues, reference_metrics)
        availability = self._metric_availability(metrics, evidence_mode)
        return AgentResult(
            agent_name=self.name,
            evidence_mode=evidence_mode,  # type: ignore[arg-type]
            issues=issues,
            metrics=metrics,
            recommendations=recommendations,
            uncertainty={
                "candidate_anomaly_cells": len(candidate_anomaly_cells),
                "statistical_anomalies_are_not_confirmed_errors": True,
                "warnings": warnings,
            },
            execution_details={
                "analyzed_rows": int(len(df)),
                "analyzed_columns": int(len(df.columns)),
                "analyzed_cells": int(df.size),
                "contract_rules_evaluated": int(
                    sum(
                        bool(meta.valid_range)
                        + bool(effective_allowed_values(meta))
                        + bool(meta.regex)
                        + bool(meta.target_format)
                        + bool(meta.uniqueness_required)
                        for meta in contract.columns.values()
                    )
                    + len(contract.cross_field_rules)
                ),
                "reference_supplied": reference_df is not None,
                "clean_reference_used_by_accuracy_agent": reference_metrics is not None,
                "manifest_supplied": truth_manifest is not None,
                "tools": [
                    "contract_role_parser",
                    "range_validator",
                    "categorical_dictionary_validator",
                    "format_validator",
                    "cross_field_rule_engine",
                    "MAD",
                    "IQR_Tukey",
                    "IsolationForest",
                    "LocalOutlierFactor",
                    "clean_reference_comparator",
                ],
            },
            metric_availability=availability,
        )

    def _detect_anomalies(
        self,
        df: pd.DataFrame,
        contract: DatasetContract,
        numeric_series_by_column: dict[str, pd.Series],
    ) -> list[Any]:
        issues = []
        for name, numeric in numeric_series_by_column.items():
            observed = numeric.dropna()
            if len(observed) < 20 or observed.nunique() < 3:
                continue
            detector_hits: dict[int, set[str]] = defaultdict(set)
            sentinel_values = {float(item["candidate"]) for item in numeric_sentinel_candidates(observed)}

            q1, q3 = observed.quantile([0.25, 0.75])
            iqr = float(q3 - q1)
            if iqr > 0:
                mask = (observed < q1 - 3.0 * iqr) | (observed > q3 + 3.0 * iqr)
                for idx in observed.index[mask]:
                    detector_hits[int(idx)].add("IQR_Tukey_outer")

            median = float(observed.median())
            mad = float(np.median(np.abs(observed - median)))
            if mad > 0:
                robust_z = 0.6745 * (observed - median) / mad
                for idx in observed.index[robust_z.abs() > 4.5]:
                    detector_hits[int(idx)].add("MAD_robust_z")

            try:
                from sklearn.ensemble import IsolationForest

                model = IsolationForest(random_state=42, contamination="auto")
                labels = model.fit_predict(observed.to_numpy().reshape(-1, 1))
                for idx, label in zip(observed.index, labels):
                    if int(label) == -1:
                        detector_hits[int(idx)].add("IsolationForest")
            except Exception:
                pass

            try:
                from sklearn.neighbors import LocalOutlierFactor

                neighbors = min(20, max(2, len(observed) - 1))
                model = LocalOutlierFactor(n_neighbors=neighbors, contamination="auto")
                labels = model.fit_predict(observed.to_numpy().reshape(-1, 1))
                for idx, label in zip(observed.index, labels):
                    if int(label) == -1:
                        detector_hits[int(idx)].add("LocalOutlierFactor")
            except Exception:
                pass

            for row_index, detectors in sorted(detector_hits.items()):
                if float(observed.loc[row_index]) in sentinel_values:
                    continue
                # Emit a candidate only when a robust univariate detector (IQR or MAD)
                # flags the value. IsolationForest and LOF strengthen the evidence but
                # do not independently create hundreds of weak candidates.
                if not ({"IQR_Tukey_outer", "MAD_robust_z"} & detectors):
                    continue
                confidence = min(0.95, 0.55 + 0.10 * len(detectors))
                issues.append(
                    make_issue(
                        "numeric_extreme_candidate",
                        "cell",
                        name,
                        row_index,
                        df.at[row_index, name],
                        "candidate",
                        confidence,
                        0.35,
                        {
                            "detector_names": sorted(detectors),
                            "number_of_detectors_flagging_cell": len(detectors),
                            "ensemble_score": len(detectors) / 4.0,
                            "requires_domain_or_reference_confirmation": True,
                        },
                    )
                )
        return issues

    def _evaluate_cross_field_rules(
        self,
        df: pd.DataFrame,
        contract: DatasetContract,
    ) -> tuple[list[Any], dict[str, int]]:
        issues = []
        attempts = 0
        successes = 0
        not_evaluable = 0
        operators = {
            "<": operator.lt,
            "<=": operator.le,
            "==": operator.eq,
            "!=": operator.ne,
            ">=": operator.ge,
            ">": operator.gt,
            "in": lambda left, right: left in right,
            "not_in": lambda left, right: left not in right,
        }

        def _coerce_series(series: pd.Series, role: str | None) -> pd.Series:
            """Coerce operands to the column's contract role before comparison.

            Raw comparison previously produced lexicographic string ordering
            ('25' <= '120' is False) and TypeError on mixed dtypes (31.0 >= '0'),
            and the exception path counted as a violation — every non-missing
            value 'violated' the rule. Values that cannot be coerced are
            excluded from evaluation (not_evaluable), never counted as violations.
            """
            if role == "numeric":
                return pd.to_numeric(series.astype("string").str.strip(), errors="coerce")
            if role == "date":
                try:
                    return pd.to_datetime(series.astype("string").str.strip(), errors="coerce", format="mixed")
                except (TypeError, ValueError):
                    return pd.to_datetime(series.astype("string").str.strip(), errors="coerce")
            return series.astype("string").str.strip()

        def _coerce_scalar(value: Any, role: str | None) -> Any:
            if role == "numeric":
                coerced = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                return None if pd.isna(coerced) else float(coerced)
            if role == "date":
                coerced = pd.to_datetime(str(value), errors="coerce")
                return None if pd.isna(coerced) else coerced
            return str(value).strip()

        for rule in contract.cross_field_rules:
            resolved = self._resolve_rule(rule, df)
            if resolved is None:
                continue
            left_column, op_name, right_column, right_value = resolved
            if left_column not in df.columns or (right_column and right_column not in df.columns):
                continue
            fn = operators[op_name]
            left_meta = contract.columns.get(left_column)
            role = left_meta.role if left_meta is not None else None
            left_series = _coerce_series(df[left_column], role)
            if right_column:
                right_series = _coerce_series(df[right_column], role)
                right_scalar = None
            else:
                right_series = None
                right_scalar = _coerce_scalar(right_value, role)
                if right_scalar is None:
                    # Rule constant cannot be interpreted in the column's role;
                    # the whole rule is not evaluable rather than universally violated.
                    not_evaluable += int(left_series.notna().sum())
                    continue
            for row_index in df.index:
                raw_left = df.at[row_index, left_column]
                raw_right = df.at[row_index, right_column] if right_column else right_value
                if pd.isna(raw_left) or (right_column and pd.isna(raw_right)):
                    continue
                left = left_series.at[row_index]
                right = right_series.at[row_index] if right_series is not None else right_scalar
                if pd.isna(left) or (right_series is not None and pd.isna(right)):
                    # Present but uncoercible in the column's role: report the
                    # parse problem separately, do not assert a rule violation.
                    not_evaluable += 1
                    continue
                attempts += 1
                try:
                    valid = bool(fn(left, right))
                except Exception:
                    not_evaluable += 1
                    attempts -= 1
                    continue
                if valid:
                    successes += 1
                    continue
                issues.append(
                    make_issue(
                        "cross_field_rule_violation",
                        "row",
                        None,
                        int(row_index),
                        None,
                        rule.severity,
                        1.0 if rule.severity == "confirmed" else 0.8,
                        0.8,
                        {
                            "rule_id": rule.rule_id,
                            "description": rule.description,
                            "left_column": left_column,
                            "operator": op_name,
                            "right_column": right_column,
                            "right_value": right_value,
                            "observed_left": left,
                            "observed_right": right,
                        },
                    )
                )
        return issues, {"attempts": attempts, "successes": successes, "not_evaluable": not_evaluable}

    def _resolve_rule(self, rule: CrossFieldRule, df: pd.DataFrame) -> tuple[str, str, str | None, Any] | None:
        if rule.left_column and rule.operator:
            return rule.left_column, rule.operator, rule.right_column, rule.right_value
        if not rule.expression:
            return None
        expression = rule.expression.strip()
        match = re.fullmatch(r"`?(.+?)`?\s*(<=|>=|==|!=|<|>)\s*`?(.+?)`?", expression)
        if not match:
            return None
        left, op_name, right = match.groups()
        left = left.strip(" `")
        right = right.strip(" `")
        if left not in df.columns:
            return None
        if right in df.columns:
            return left, op_name, right, None
        try:
            value: Any = float(right)
        except ValueError:
            value = right.strip("'\"")
        return left, op_name, None, value

    def _reference_assessment(
        self,
        df: pd.DataFrame,
        reference_df: pd.DataFrame,
        contract: DatasetContract,
    ) -> tuple[dict[str, Any], list[Any], list[str]]:
        warnings: list[str] = []
        keys = list(contract.primary_key)
        common_columns = [str(column) for column in df.columns if column in reference_df.columns and str(column) not in keys]
        if not common_columns:
            return {
                "state": "NOT_COMPUTABLE",
                "reason": "No common non-key columns between input and reference.",
            }, [], warnings

        aligned: list[tuple[int, pd.Series, pd.Series]] = []
        alignment_method = ""
        unmatched_input = 0
        unmatched_reference = 0

        if keys and all(key in df.columns and key in reference_df.columns for key in keys):
            if df.duplicated(keys).any() or reference_df.duplicated(keys).any():
                return {
                    "state": "NOT_COMPUTABLE",
                    "reason": "Primary-key alignment is ambiguous because duplicate keys exist.",
                    "alignment_method": "primary_key",
                }, [], warnings
            left_map = {tuple(row[key] for key in keys): idx for idx, row in df.iterrows()}
            right_map = {tuple(row[key] for key in keys): idx for idx, row in reference_df.iterrows()}
            common_keys = sorted(set(left_map) & set(right_map), key=repr)
            for key in common_keys:
                source_idx = int(left_map[key])
                aligned.append((source_idx, df.loc[source_idx], reference_df.loc[right_map[key]]))
            unmatched_input = len(set(left_map) - set(right_map))
            unmatched_reference = len(set(right_map) - set(left_map))
            alignment_method = "primary_key"
        elif contract.row_alignment_policy in {"position", "position_with_warning"}:
            count = min(len(df), len(reference_df))
            for position in range(count):
                source_idx = int(df.index[position])
                aligned.append((source_idx, df.iloc[position], reference_df.iloc[position]))
            unmatched_input = max(0, len(df) - count)
            unmatched_reference = max(0, len(reference_df) - count)
            alignment_method = "row_position"
            if contract.row_alignment_policy == "position_with_warning":
                warnings.append("Reference rows were aligned by position because no approved primary key was supplied.")
        else:
            return {
                "state": "NOT_COMPUTABLE",
                "reason": "Reference alignment requires an approved primary key.",
                "alignment_method": "blocked",
            }, [], warnings

        exact_equal = 0
        tolerance_equal = 0
        comparable = 0
        row_exact = 0
        per_column_counts: dict[str, dict[str, int]] = {
            name: {"comparable": 0, "exact_equal": 0, "tolerance_equal": 0, "wrong": 0}
            for name in common_columns
        }
        numeric_errors: list[float] = []
        categorical_true: list[str] = []
        categorical_pred: list[str] = []
        issues = []

        for source_idx, left_row, right_row in aligned:
            this_row_exact = True
            for name in common_columns:
                meta = contract.columns.get(name)
                if meta is None:
                    continue
                left = left_row[name]
                right = right_row[name]
                both_missing = pd.isna(left) and pd.isna(right)
                if both_missing:
                    exact = tolerance = True
                elif pd.isna(left) != pd.isna(right):
                    exact = tolerance = False
                elif meta.role == "numeric":
                    left_value = pd.to_numeric(pd.Series([canonicalize_value(left, meta)]), errors="coerce").iloc[0]
                    right_value = pd.to_numeric(pd.Series([canonicalize_value(right, meta)]), errors="coerce").iloc[0]
                    if pd.isna(left_value) or pd.isna(right_value):
                        exact = tolerance = False
                    else:
                        difference = abs(float(left_value) - float(right_value))
                        exact = difference == 0.0
                        tolerance = difference <= float(meta.numeric_tolerance)
                        numeric_errors.append(difference)
                elif meta.role == "date":
                    left_date = pd.to_datetime(left, errors="coerce")
                    right_date = pd.to_datetime(right, errors="coerce")
                    if pd.isna(left_date) or pd.isna(right_date):
                        exact = tolerance = False
                    else:
                        difference = abs((left_date - right_date).total_seconds()) / 86400.0
                        exact = difference == 0.0
                        tolerance = difference <= float(meta.date_tolerance_days)
                else:
                    left_norm = normalize_role_value(
                        left,
                        role=meta.role,
                        case_sensitive=meta.case_sensitive,
                    )
                    right_norm = normalize_role_value(
                        right,
                        role=meta.role,
                        case_sensitive=meta.case_sensitive,
                    )
                    exact = str(left).strip() == str(right).strip()
                    tolerance = left_norm == right_norm
                    if meta.role == "categorical":
                        categorical_true.append(str(right_norm))
                        categorical_pred.append(str(left_norm))

                comparable += 1
                per_column_counts[name]["comparable"] += 1
                if exact:
                    exact_equal += 1
                    per_column_counts[name]["exact_equal"] += 1
                else:
                    this_row_exact = False
                if tolerance:
                    tolerance_equal += 1
                    per_column_counts[name]["tolerance_equal"] += 1
                else:
                    per_column_counts[name]["wrong"] += 1
                    issues.append(
                        make_issue(
                            "reference_cell_mismatch",
                            "cell",
                            name,
                            source_idx,
                            left,
                            "confirmed",
                            1.0,
                            0.9,
                            {
                                "reference_value": None if pd.isna(right) else right,
                                "alignment_method": alignment_method,
                                "numeric_tolerance": meta.numeric_tolerance,
                                "date_tolerance_days": meta.date_tolerance_days,
                            },
                        )
                    )
            if this_row_exact:
                row_exact += 1

        per_column = {
            name: {
                **counts,
                "exact_cell_accuracy": counts["exact_equal"] / counts["comparable"] if counts["comparable"] else None,
                "tolerance_adjusted_accuracy": counts["tolerance_equal"] / counts["comparable"] if counts["comparable"] else None,
            }
            for name, counts in per_column_counts.items()
        }
        categorical_accuracy = None
        categorical_macro_f1 = None
        if categorical_true:
            categorical_accuracy = float(np.mean(np.array(categorical_true) == np.array(categorical_pred)))
            try:
                from sklearn.metrics import f1_score

                categorical_macro_f1 = float(f1_score(categorical_true, categorical_pred, average="macro", zero_division=0))
            except Exception:
                categorical_macro_f1 = None

        return {
            "state": "COMPUTED",
            "alignment_method": alignment_method,
            "aligned_rows": len(aligned),
            "unmatched_input_rows": unmatched_input,
            "unmatched_reference_rows": unmatched_reference,
            "common_columns": common_columns,
            "excluded_columns": keys,
            "comparable_cells": comparable,
            "mismatched_cells": comparable - tolerance_equal,
            "wrong_cell_count": comparable - tolerance_equal,
            "exact_cell_accuracy": exact_equal / comparable if comparable else None,
            "tolerance_adjusted_accuracy": tolerance_equal / comparable if comparable else None,
            "row_exact_match_rate": row_exact / len(aligned) if aligned else None,
            "per_column_accuracy": per_column,
            "numeric_MAE": float(np.mean(numeric_errors)) if numeric_errors else None,
            "numeric_RMSE": float(np.sqrt(np.mean(np.square(numeric_errors)))) if numeric_errors else None,
            "categorical_accuracy": categorical_accuracy,
            "categorical_macro_F1": categorical_macro_f1,
        }, issues, warnings

    def _recommendations(self, issues: list[Any], reference_metrics: dict[str, Any] | None) -> list[AgentRecommendation]:
        grouped = Counter((issue.issue_type, issue.column) for issue in issues)
        rows: list[AgentRecommendation] = []
        for (issue_type, column), count in sorted(grouped.items(), key=lambda item: (item[0][0], str(item[0][1]))):
            rows.append(
                AgentRecommendation(
                    recommendation_id=stable_id(self.name, issue_type, column, prefix="AREC"),
                    agent_name=self.name,
                    issue_type=issue_type,
                    column=column,
                    method="contract_or_reference_guided_review",
                    rationale=f"Review {count} findings using the approved contract and available evidence.",
                    prerequisites=["approved contract", "HITL confirmation"],
                    validation_metrics=reference_metrics or {},
                    risk="medium" if issue_type in {"reference_cell_mismatch", "categorical_invalid_value"} else "low",
                    priority=1,
                )
            )
        return rows

    def _metric_availability(self, metrics: dict[str, Any], evidence_mode: str) -> list[MetricAvailability]:
        result = [MetricAvailability("contract_validity_support", "COMPUTED", evidence_mode=evidence_mode)]
        reference = metrics.get("reference_metrics")
        if reference and reference.get("state") == "COMPUTED":
            result.append(MetricAvailability("exact_cell_accuracy", "COMPUTED", evidence_mode=evidence_mode))
            result.append(MetricAvailability("tolerance_adjusted_accuracy", "COMPUTED", evidence_mode=evidence_mode))
        else:
            reason = reference.get("reason") if isinstance(reference, dict) else "Clean reference dataset was not supplied."
            result.append(MetricAvailability("exact_cell_accuracy", "NOT_COMPUTABLE", str(reason), evidence_mode=evidence_mode))
            result.append(MetricAvailability("tolerance_adjusted_accuracy", "NOT_COMPUTABLE", str(reason), evidence_mode=evidence_mode))
        if metrics.get("manifest_detection_metrics") is not None:
            result.append(MetricAvailability("precision_recall_f1_mcc", "COMPUTED", evidence_mode=evidence_mode))
        else:
            result.append(
                MetricAvailability(
                    "precision_recall_f1_mcc",
                    "NOT_COMPUTABLE",
                    "Corruption manifest was not supplied.",
                    evidence_mode=evidence_mode,
                )
            )
        return result
