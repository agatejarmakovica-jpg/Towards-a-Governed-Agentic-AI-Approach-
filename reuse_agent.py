from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .agent_utils import deduplicate_issues, effective_allowed_values, make_issue, normalize_for_comparison
from .contract import DatasetContract
from .models import AgentRecommendation, AgentResult, MetricAvailability
from .utils import stable_id


class ReuseAgent:
    name = "ReuseAgent"

    def run(self, df: pd.DataFrame, contract: DatasetContract, *, evidence_mode: str) -> AgentResult:
        issues = []
        columns = [str(column) for column in df.columns]
        declared = set(contract.columns)
        observed = set(columns)

        dataset_meta = contract.dataset_metadata.model_dump(mode="json")
        required_dataset_fields = list(contract.dataset_metadata_required_fields)
        dataset_present = sum(dataset_meta.get(field) not in (None, "", [], {}) for field in required_dataset_fields)
        dataset_documentation_completeness = dataset_present / max(1, len(required_dataset_fields))
        missing_dataset_fields = [
            field for field in required_dataset_fields if dataset_meta.get(field) in (None, "", [], {})
        ]
        if missing_dataset_fields:
            issues.append(
                make_issue(
                    "dataset_metadata_gap",
                    "dataset",
                    None,
                    None,
                    None,
                    "confirmed",
                    1.0,
                    min(1.0, len(missing_dataset_fields) / max(1, len(required_dataset_fields))),
                    {"missing_fields": missing_dataset_fields},
                )
            )

        column_metadata_scores: dict[str, float] = {}
        role_defined = 0
        unit_applicable = 0
        unit_defined = 0
        code_applicable = 0
        code_defined = 0
        categorical_standardized_values = 0
        categorical_values = 0
        date_standardized_values = 0
        date_values = 0
        mixed_type_columns = 0

        for column in columns:
            meta = contract.columns.get(column)
            if meta is None:
                issues.append(
                    make_issue(
                        "schema_metadata_gap",
                        "column",
                        column,
                        None,
                        None,
                        "confirmed",
                        1.0,
                        1.0,
                        {"missing_fields": contract.metadata_required_fields},
                    )
                )
                column_metadata_scores[column] = 0.0
                continue

            present_fields = sum(
                getattr(meta, field, None) not in (None, "", [], {}) for field in contract.metadata_required_fields
            )
            score = present_fields / max(1, len(contract.metadata_required_fields))
            column_metadata_scores[column] = score
            missing_fields = [
                field for field in contract.metadata_required_fields
                if getattr(meta, field, None) in (None, "", [], {})
            ]
            if missing_fields:
                issues.append(
                    make_issue(
                        "schema_metadata_gap",
                        "column",
                        column,
                        None,
                        None,
                        "confirmed",
                        1.0,
                        min(1.0, len(missing_fields) / max(1, len(contract.metadata_required_fields))),
                        {"missing_fields": missing_fields},
                    )
                )
            if meta.role:
                role_defined += 1

            if meta.role == "numeric":
                unit_applicable += 1
                if meta.measurement_unit:
                    unit_defined += 1
            if meta.role == "categorical":
                code_applicable += 1
                if meta.standard_or_code_system or effective_allowed_values(meta):
                    code_defined += 1
                allowed = {
                    normalize_for_comparison(value, case_sensitive=meta.case_sensitive)
                    for value in effective_allowed_values(meta)
                }
                nonmissing = df[column].dropna()
                categorical_values += len(nonmissing)
                if allowed:
                    categorical_standardized_values += sum(
                        normalize_for_comparison(value, case_sensitive=meta.case_sensitive) in allowed
                        for value in nonmissing
                    )
            if meta.role == "date":
                nonmissing = df[column].dropna().astype("string").str.strip()
                date_values += len(nonmissing)
                if meta.target_format:
                    import datetime as _dt

                    for value in nonmissing:
                        try:
                            parsed = _dt.datetime.strptime(str(value), meta.target_format)
                            if parsed.strftime(meta.target_format) == str(value):
                                date_standardized_values += 1
                        except Exception:
                            pass

            observed_types = df[column].dropna().map(lambda value: type(value).__name__).nunique()
            if observed_types > 1:
                mixed_type_columns += 1

        column_metadata_completeness = float(np.mean(list(column_metadata_scores.values()))) if columns else 1.0
        schema_coverage = len(observed & declared) / max(1, len(observed))
        role_definition_coverage = role_defined / max(1, len(columns))
        unit_definition_coverage = unit_defined / unit_applicable if unit_applicable else None
        code_system_coverage = code_defined / code_applicable if code_applicable else None
        categorical_standardization_rate = (
            categorical_standardized_values / categorical_values if categorical_values else None
        )
        date_standardization_rate = date_standardized_values / date_values if date_values else None

        format_components = [
            value for value in (categorical_standardization_rate, date_standardization_rate) if value is not None
        ]
        format_standardization_rate = float(np.mean(format_components)) if format_components else None

        duplicate_free_rate = 1.0 - int(df.duplicated(keep="first").sum()) / max(1, len(df))
        direct_identifier_columns = []
        quasi_identifier_columns = []
        for column, meta in contract.columns.items():
            if column not in df.columns:
                continue
            if meta.sensitivity == "direct_identifier" or meta.role in {"email", "phone"}:
                direct_identifier_columns.append(column)
                issues.append(
                    make_issue(
                        "direct_identifier_column",
                        "column",
                        column,
                        None,
                        None,
                        "confirmed",
                        1.0,
                        0.9,
                        {
                            "declared_sensitivity": meta.sensitivity,
                            "role": meta.role,
                            "non_null_values": int(df[column].notna().sum()),
                        },
                    )
                )
            elif meta.sensitivity == "quasi_identifier":
                quasi_identifier_columns.append(column)

        privacy_documentation_completeness = 1.0 if str(dataset_meta.get("privacy_status", "")).strip() else 0.0
        if privacy_documentation_completeness == 0:
            issues.append(
                make_issue(
                    "privacy_documentation_gap",
                    "dataset",
                    None,
                    None,
                    None,
                    "confirmed",
                    1.0,
                    0.8,
                    {"missing_field": "privacy_status"},
                )
            )

        direct_identifier_exposure = len(direct_identifier_columns) / max(1, len(columns))
        quasi_identifier_exposure = len(quasi_identifier_columns) / max(1, len(columns))
        privacy_support = float(
            np.clip(
                0.5 * privacy_documentation_completeness
                + 0.35 * (1.0 - direct_identifier_exposure)
                + 0.15 * (1.0 - quasi_identifier_exposure),
                0.0,
                1.0,
            )
        )

        # Machine readability is computed from observable structural indicators, never a constant.
        indicators = {
            "table_readable": 1.0,
            "unique_column_names": 1.0 if len(columns) == len(set(columns)) else 0.0,
            "nonempty_column_names": 1.0 if all(str(column).strip() for column in columns) else 0.0,
            "no_unnamed_columns": 1.0 if not any(str(column).casefold().startswith("unnamed") for column in columns) else 0.0,
            "single_header_assumption": 1.0 if not any(str(column).isdigit() for column in columns) else 0.5,
            "controlled_type_consistency": 1.0 - mixed_type_columns / max(1, len(columns)),
            "machine_readable_schema": schema_coverage,
        }
        machine_readability = float(np.mean(list(indicators.values())))

        required_for_reuse = list(contract.required_columns_for_reuse)
        missing_required_reuse = [column for column in required_for_reuse if column not in df.columns]
        reuse_purpose_completeness = 1.0 if str(dataset_meta.get("intended_reuse", "")).strip() else 0.0
        provenance_completeness = 1.0 if str(dataset_meta.get("provenance", "")).strip() else 0.0
        license_completeness = 1.0 if str(dataset_meta.get("license", "")).strip() else 0.0
        reproducibility_artifact_coverage = float(
            np.mean(
                [
                    1.0 if contract.approval.status == "approved" else 0.0,
                    provenance_completeness,
                    license_completeness,
                    1.0 if str(dataset_meta.get("version", "")).strip() else 0.0,
                    schema_coverage,
                ]
            )
        )

        documentation_component = float(
            np.mean(
                [
                    dataset_documentation_completeness,
                    column_metadata_completeness,
                    provenance_completeness,
                    license_completeness,
                    reuse_purpose_completeness,
                ]
            )
        )
        schema_component_values = [schema_coverage, role_definition_coverage]
        if unit_definition_coverage is not None:
            schema_component_values.append(unit_definition_coverage)
        if code_system_coverage is not None:
            schema_component_values.append(code_system_coverage)
        schema_component = float(np.mean(schema_component_values))
        standardization_component = format_standardization_rate if format_standardization_rate is not None else 0.0

        policy = contract.reuse_policy
        reuse_readiness = float(
            np.clip(
                policy.documentation_weight * documentation_component
                + policy.schema_weight * schema_component
                + policy.standardization_weight * standardization_component
                + policy.privacy_weight * privacy_support
                + policy.machine_readability_weight * machine_readability,
                0.0,
                1.0,
            )
        )

        blockers = []
        if missing_required_reuse:
            blockers.append(f"Required reuse columns are missing: {missing_required_reuse}")
        if direct_identifier_columns and contract.decision_policy.max_direct_identifier_columns_for_ready == 0:
            blockers.append("Direct identifiers exceed the ready-state policy.")
        if license_completeness == 0:
            blockers.append("Dataset license is missing.")
        if provenance_completeness == 0:
            blockers.append("Dataset provenance is missing.")
        if machine_readability < 0.5:
            blockers.append("Machine readability is below 0.5.")

        metrics = {
            "dataset_documentation_completeness": dataset_documentation_completeness,
            "column_metadata_completeness": column_metadata_completeness,
            "schema_coverage": schema_coverage,
            "role_definition_coverage": role_definition_coverage,
            "unit_definition_coverage": unit_definition_coverage,
            "code_system_coverage": code_system_coverage,
            "format_standardization_rate": format_standardization_rate,
            "categorical_standardization_rate": categorical_standardization_rate,
            "date_standardization_rate": date_standardization_rate,
            "provenance_completeness": provenance_completeness,
            "license_completeness": license_completeness,
            "reuse_purpose_completeness": reuse_purpose_completeness,
            "privacy_documentation_completeness": privacy_documentation_completeness,
            "direct_identifier_exposure": direct_identifier_exposure,
            "quasi_identifier_exposure": quasi_identifier_exposure,
            "privacy_support": privacy_support,
            "machine_readability": machine_readability,
            "machine_readability_indicators": indicators,
            "duplicate_free_rate": duplicate_free_rate,
            "reproducibility_artifact_coverage": reproducibility_artifact_coverage,
            "reuse_readiness_support": reuse_readiness,
            "blocking_conditions": blockers,
            "missing_required_columns_for_reuse": missing_required_reuse,
        }

        issues = deduplicate_issues(issues)
        recommendations = self._recommendations(metrics, direct_identifier_columns, missing_required_reuse)
        availability = [
            MetricAvailability("reuse_readiness_support", "COMPUTED", evidence_mode=evidence_mode),
            MetricAvailability("machine_readability", "COMPUTED", evidence_mode=evidence_mode),
        ]
        for metric_name, value, reason in (
            ("unit_definition_coverage", unit_definition_coverage, "No numeric variables were declared."),
            ("code_system_coverage", code_system_coverage, "No categorical variables were declared."),
            ("format_standardization_rate", format_standardization_rate, "No categorical or date variables were assessable."),
        ):
            availability.append(
                MetricAvailability(
                    metric_name,
                    "COMPUTED" if value is not None else "NOT_APPLICABLE",
                    "" if value is not None else reason,
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
                "unobservable_file_features": [
                    "merged_cells_after_dataframe_import",
                    "multiple_tables_in_one_spreadsheet_sheet",
                    "source_file_encoding_after_successful_parse",
                ],
                "blocking_conditions": blockers,
            },
            execution_details={
                "analyzed_rows": int(len(df)),
                "analyzed_columns": int(len(df.columns)),
                "analyzed_cells": int(df.size),
                "tools": [
                    "metadata_completeness",
                    "schema_coverage",
                    "standardization_assessment",
                    "privacy_exposure_assessment",
                    "machine_readability_indicators",
                ],
                "reuse_weights": contract.reuse_policy.model_dump(mode="json"),
            },
            metric_availability=availability,
        )

    def _recommendations(
        self,
        metrics: dict[str, Any],
        direct_identifier_columns: list[str],
        missing_required_reuse: list[str],
    ) -> list[AgentRecommendation]:
        rows: list[AgentRecommendation] = []
        if metrics["dataset_documentation_completeness"] < 1.0:
            rows.append(
                AgentRecommendation(
                    recommendation_id=stable_id(self.name, "dataset_metadata", prefix="AREC"),
                    agent_name=self.name,
                    issue_type="dataset_metadata_gap",
                    column=None,
                    method="complete_dataset_metadata",
                    rationale="Complete provenance, license, reuse purpose, privacy, and version metadata.",
                    prerequisites=["data owner confirmation"],
                    risk="low",
                    priority=1,
                )
            )
        for column in direct_identifier_columns:
            rows.append(
                AgentRecommendation(
                    recommendation_id=stable_id(self.name, "direct_identifier", column, prefix="AREC"),
                    agent_name=self.name,
                    issue_type="direct_identifier_column",
                    column=column,
                    method="restrict_remove_or_pseudonymize_for_reuse",
                    rationale="The declared reuse policy does not allow unrestricted direct-identifier exposure.",
                    prerequisites=["privacy review", "approved reuse purpose", "raw-data preservation"],
                    risk="high",
                    priority=1,
                )
            )
        if missing_required_reuse:
            rows.append(
                AgentRecommendation(
                    recommendation_id=stable_id(self.name, "required_reuse_columns", prefix="AREC"),
                    agent_name=self.name,
                    issue_type="required_reuse_column_missing",
                    column=None,
                    method="supply_or_redefine_required_reuse_columns",
                    rationale=f"The declared reuse task requires missing columns: {missing_required_reuse}.",
                    prerequisites=["approved reuse-purpose definition"],
                    risk="medium",
                    priority=1,
                )
            )
        if metrics["machine_readability"] < 1.0:
            rows.append(
                AgentRecommendation(
                    recommendation_id=stable_id(self.name, "machine_readability", prefix="AREC"),
                    agent_name=self.name,
                    issue_type="machine_readability_gap",
                    column=None,
                    method="normalize_tabular_structure_and_schema",
                    rationale="Resolve failed structural indicators before repository or ML reuse.",
                    prerequisites=["preserved source snapshot", "documented transformation"],
                    validation_metrics=metrics["machine_readability_indicators"],
                    risk="low",
                    priority=2,
                )
            )
        return rows
