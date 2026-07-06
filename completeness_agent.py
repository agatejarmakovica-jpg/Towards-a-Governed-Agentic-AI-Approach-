from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from .agent_utils import (
    DEFAULT_MISSING_TOKENS,
    deduplicate_issues,
    effective_allowed_values,
    is_missing_by_contract,
    make_issue,
    normalize_for_comparison,
    numeric_sentinel_candidates,
    parsed_numeric_series,
)
from .contract import DatasetContract
from .models import AgentRecommendation, AgentResult, MetricAvailability
from .utils import stable_id


class CompletenessAgent:
    name = "CompletenessAgent"

    def run(self, df: pd.DataFrame, contract: DatasetContract, *, evidence_mode: str) -> AgentResult:
        issues = []
        confirmed_cells: set[tuple[int, str]] = set()
        candidate_cells: set[tuple[int, str]] = set()
        missing_reason_counts: Counter[str] = Counter()
        column_missing: dict[str, set[int]] = {str(column): set() for column in df.columns}

        for column in df.columns:
            column_name = str(column)
            series = df[column]
            meta = contract.columns.get(column_name)
            if meta is None:
                continue

            allowed_normalized = {
                normalize_for_comparison(value, case_sensitive=meta.case_sensitive)
                for value in effective_allowed_values(meta)
            }
            declared_tokens = {
                str(value).strip().casefold() for value in contract.missing_tokens
            }

            for row_index, value in series.items():
                is_missing, reason = is_missing_by_contract(
                    value,
                    global_tokens=contract.missing_tokens,
                    meta=meta,
                )
                if is_missing:
                    issue_type = {
                        "explicit_null": "explicit_missingness",
                        "sentinel_value": "confirmed_disguised_missingness",
                        "declared_missing_token": "explicit_missing_token",
                    }.get(str(reason), "explicit_missingness")
                    issues.append(
                        make_issue(
                            issue_type,
                            "cell",
                            column_name,
                            int(row_index),
                            None if pd.isna(value) else value,
                            "confirmed",
                            1.0,
                            0.7,
                            {"missing_reason": reason, "contract_driven": True},
                            automatic_action_allowed=issue_type in {
                                "explicit_missing_token",
                                "confirmed_disguised_missingness",
                            },
                        )
                    )
                    confirmed_cells.add((int(row_index), column_name))
                    column_missing[column_name].add(int(row_index))
                    missing_reason_counts[str(reason)] += 1
                    continue

                if value is None or pd.isna(value):
                    continue
                key = str(value).strip().casefold()
                if key in DEFAULT_MISSING_TOKENS - declared_tokens:
                    normalized = normalize_for_comparison(value, case_sensitive=meta.case_sensitive)
                    if normalized in allowed_normalized:
                        continue
                    issues.append(
                        make_issue(
                            "suspected_missing_token",
                            "cell",
                            column_name,
                            int(row_index),
                            value,
                            "candidate",
                            0.60,
                            0.35,
                            {
                                "token_source": "generic_lexicon",
                                "requires_confirmation": True,
                                "allowed_value_priority_applied": True,
                            },
                        )
                    )
                    candidate_cells.add((int(row_index), column_name))

            if meta.role == "numeric" and not meta.sentinel_values:
                numeric, _ = parsed_numeric_series(series, meta)
                for candidate in numeric_sentinel_candidates(numeric):
                    mask = numeric.eq(candidate["candidate"])
                    for row_index in numeric.index[mask.fillna(False)]:
                        coordinate = (int(row_index), column_name)
                        if coordinate in confirmed_cells:
                            continue
                        issues.append(
                            make_issue(
                                "disguised_missingness_candidate",
                                "cell",
                                column_name,
                                int(row_index),
                                series.loc[row_index],
                                "candidate",
                                float(candidate["score"]),
                                min(1.0, candidate["frequency_share"] * 2 + 0.2),
                                {**candidate, "requires_dataset_metadata": True},
                            )
                        )
                        candidate_cells.add(coordinate)

        issues = deduplicate_issues(issues)
        rows = len(df)
        columns = len(df.columns)
        total_cells = int(df.size)
        confirmed_count = len(confirmed_cells)
        overall = 1.0 - confirmed_count / max(1, total_cells)

        column_completeness: dict[str, float] = {}
        weights: dict[str, float] = {}
        for column in df.columns:
            name = str(column)
            column_completeness[name] = 1.0 - len(column_missing.get(name, set())) / max(1, rows)
            meta = contract.columns.get(name)
            weights[name] = float(meta.criticality_weight if meta is not None else 1.0)
        weighted = sum(column_completeness[name] * weights[name] for name in column_completeness) / max(
            1e-12, sum(weights.values())
        )

        row_missing_counts = pd.Series(0, index=df.index, dtype=float)
        for name, row_set in column_missing.items():
            if row_set:
                row_missing_counts.loc[list(row_set)] += 1
        row_completeness = 1.0 - row_missing_counts / max(1, columns)
        patterns = Counter()
        for row_index in df.index:
            pattern = tuple(name for name in map(str, df.columns) if int(row_index) in column_missing.get(name, set()))
            if pattern:
                patterns["|".join(pattern)] += 1

        imputation_evaluation = self._evaluate_imputation(df, contract, column_missing)
        recommendations = self._recommendations(column_missing, imputation_evaluation)

        metrics = {
            "assessable_cells": total_cells,
            "explicit_missing_cells": int(missing_reason_counts.get("explicit_null", 0)),
            "declared_missing_token_cells": int(missing_reason_counts.get("declared_missing_token", 0)),
            "sentinel_missing_cells": int(missing_reason_counts.get("sentinel_value", 0)),
            "confirmed_missing_cells": confirmed_count,
            "candidate_missing_cells": len(candidate_cells),
            "overall_completeness": float(overall),
            "critical_variable_completeness": float(weighted),
            "column_completeness": column_completeness,
            "row_completeness_mean": float(row_completeness.mean()) if rows else 1.0,
            "row_completeness_distribution": {
                "min": float(row_completeness.min()) if rows else 1.0,
                "q25": float(row_completeness.quantile(0.25)) if rows else 1.0,
                "median": float(row_completeness.median()) if rows else 1.0,
                "q75": float(row_completeness.quantile(0.75)) if rows else 1.0,
                "max": float(row_completeness.max()) if rows else 1.0,
            },
            "complete_row_rate": float((row_completeness == 1.0).mean()) if rows else 1.0,
            "missing_pattern_count": len(patterns),
            "missing_pattern_frequencies": dict(patterns.most_common(100)),
            "imputation_evaluation": imputation_evaluation,
        }

        availability = [
            MetricAvailability("overall_completeness", "COMPUTED", evidence_mode=evidence_mode),
            MetricAvailability("critical_variable_completeness", "COMPUTED", evidence_mode=evidence_mode),
        ]
        if any(value.get("state") == "COMPUTED" for value in imputation_evaluation.values()):
            availability.append(MetricAvailability("imputation_validation", "COMPUTED", evidence_mode=evidence_mode))
        else:
            availability.append(
                MetricAvailability(
                    "imputation_validation",
                    "NOT_COMPUTABLE",
                    "No affected column had enough observed values for masked validation.",
                    evidence_mode=evidence_mode,
                )
            )

        return AgentResult(
            agent_name=self.name,
            evidence_mode=evidence_mode,  # type: ignore[arg-type]
            issues=issues,
            metrics=metrics,
            recommendations=recommendations,
            uncertainty={
                "candidate_missing_cells": len(candidate_cells),
                "candidate_values_are_not_confirmed_missingness": True,
            },
            execution_details={
                "analyzed_rows": rows,
                "analyzed_columns": columns,
                "analyzed_cells": total_cells,
                "rule_source": "approved_dataset_contract",
                "tools": ["pandas_isna", "contract_missing_tokens", "sentinel_analysis", "masked_imputation_validation"],
            },
            metric_availability=availability,
        )

    def _evaluate_imputation(
        self,
        df: pd.DataFrame,
        contract: DatasetContract,
        column_missing: dict[str, set[int]],
    ) -> dict[str, Any]:
        results: dict[str, Any] = {}
        rng = np.random.default_rng(42)

        for column in df.columns:
            name = str(column)
            if not column_missing.get(name):
                continue
            meta = contract.columns.get(name)
            if meta is None:
                continue
            observed = df[name].dropna()
            if len(observed) < 20:
                results[name] = {
                    "state": "NOT_COMPUTABLE",
                    "reason": "Insufficient observed values for masked validation.",
                    "observed_values": int(len(observed)),
                }
                continue

            sample_size = min(max(5, int(round(len(observed) * 0.20))), 100)
            sample_indices = rng.choice(observed.index.to_numpy(), size=sample_size, replace=False)
            truth = observed.loc[sample_indices]

            if meta.role == "numeric":
                numeric_observed = pd.to_numeric(observed, errors="coerce").dropna()
                numeric_truth = pd.to_numeric(truth, errors="coerce")
                valid_truth = numeric_truth.notna()
                if valid_truth.sum() < 5 or numeric_observed.empty:
                    results[name] = {
                        "state": "NOT_COMPUTABLE",
                        "reason": "Insufficient parseable numeric values for masked validation.",
                    }
                    continue
                prediction = float(numeric_observed.median())
                errors = numeric_truth[valid_truth].astype(float) - prediction
                rmse = float(np.sqrt(np.mean(np.square(errors))))
                mae = float(np.mean(np.abs(errors)))
                scale = float(numeric_observed.max() - numeric_observed.min())
                results[name] = {
                    "state": "COMPUTED",
                    "selected_method": "median",
                    "methods": {
                        "median": {
                            "MAE": mae,
                            "RMSE": rmse,
                            "normalized_RMSE": rmse / scale if scale > 0 else None,
                            "coverage": float(valid_truth.mean()),
                            "failed_imputations": int((~valid_truth).sum()),
                        }
                    },
                    "masked_values": int(sample_size),
                }
            else:
                mode = observed.astype("string").mode(dropna=True)
                if mode.empty:
                    results[name] = {"state": "NOT_COMPUTABLE", "reason": "No modal category available."}
                    continue
                predicted = str(mode.iloc[0])
                truth_text = truth.astype("string")
                accuracy = float((truth_text == predicted).mean())
                labels = sorted(set(truth_text.dropna().tolist()) | {predicted})
                per_label_f1 = []
                for label in labels:
                    y_true = truth_text.eq(label)
                    y_pred = pd.Series([predicted == label] * len(truth_text), index=truth_text.index)
                    tp = int((y_true & y_pred).sum())
                    fp = int((~y_true & y_pred).sum())
                    fn = int((y_true & ~y_pred).sum())
                    precision = tp / (tp + fp) if tp + fp else 0.0
                    recall = tp / (tp + fn) if tp + fn else 0.0
                    per_label_f1.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
                results[name] = {
                    "state": "COMPUTED",
                    "selected_method": "mode",
                    "methods": {
                        "mode": {
                            "categorical_accuracy": accuracy,
                            "macro_F1": float(np.mean(per_label_f1)) if per_label_f1 else 0.0,
                            "coverage": 1.0,
                            "failed_imputations": 0,
                        }
                    },
                    "masked_values": int(sample_size),
                }
        return results

    def _recommendations(
        self,
        column_missing: dict[str, set[int]],
        imputation_evaluation: dict[str, Any],
    ) -> list[AgentRecommendation]:
        rows: list[AgentRecommendation] = []
        for column, affected in sorted(column_missing.items()):
            if not affected:
                continue
            evaluation = imputation_evaluation.get(column, {})
            rows.append(
                AgentRecommendation(
                    recommendation_id=stable_id("CompletenessAgent", column, "missingness_mechanism_review", prefix="AREC"),
                    agent_name=self.name,
                    issue_type="explicit_missingness",
                    column=column,
                    method="missingness_mechanism_review",
                    rationale=f"Review the missingness mechanism before changing {len(affected)} affected cells.",
                    prerequisites=["variable definition", "missingness mechanism assessment"],
                    validation_metrics=evaluation,
                    risk="low",
                    priority=1,
                )
            )
            if evaluation.get("state") == "COMPUTED":
                rows.append(
                    AgentRecommendation(
                        recommendation_id=stable_id("CompletenessAgent", column, "validated_imputation_preview", prefix="AREC"),
                        agent_name=self.name,
                        issue_type="explicit_missingness",
                        column=column,
                        method=f"preview_{evaluation.get('selected_method')}_imputation",
                        rationale="The method has a masked-value validation result and may be previewed for HITL review.",
                        prerequisites=["human approval", "curated-copy output", "same-contract re-evaluation"],
                        validation_metrics=evaluation,
                        risk="medium",
                        priority=2,
                    )
                )
        return rows
