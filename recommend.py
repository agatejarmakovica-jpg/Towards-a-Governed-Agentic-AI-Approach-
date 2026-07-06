from __future__ import annotations

from collections import defaultdict

from .models import Issue, Recommendation
from .utils import stable_id

CATALOG: dict[str, list[tuple[str, str, list[str], str, str]]] = {
    "explicit_missingness": [
        ("missingness_mechanism_review", "Determine whether omission is structural, random, or systematic before treatment.", ["variable definition", "missingness analysis"], "low", "manual_or_analysis"),
        ("multiple_imputation_or_model_specific_handling", "Use a validated imputation method only when downstream analysis requires complete inputs.", ["sufficient observed data", "masked validation", "train-fold-only fitting"], "medium", "preview_then_authorize"),
    ],
    "explicit_missing_token": [
        ("normalize_confirmed_missing_token_to_null", "The token is declared as missing in dataset-scoped metadata.", ["declared missing-token mapping"], "low", "safe_if_confirmed"),
    ],
    "suspected_missing_token": [
        ("confirm_token_semantics", "A generic token lexicon is not enough to modify source data.", ["codebook or data-owner confirmation"], "low", "human_review"),
    ],
    "confirmed_disguised_missingness": [
        ("recode_confirmed_sentinel_to_null", "Dataset-scoped metadata confirms that the value encodes missingness.", ["versioned codebook", "preserved raw snapshot"], "low", "safe_if_confirmed"),
    ],
    "disguised_missingness_candidate": [
        ("retrieve_or_request_variable_codebook", "Boundary frequency and distribution gaps support a sentinel hypothesis but do not establish meaning.", ["dataset-specific documentation"], "low", "human_review"),
        ("sensitivity_analysis_with_and_without_candidate", "Quantify downstream sensitivity without overwriting the observed value.", ["declared reuse task"], "medium", "analysis_only"),
    ],
    "numeric_extreme_candidate": [
        ("domain_range_confirmation", "Robust statistical extremeness does not imply error.", ["units", "valid range", "measurement protocol"], "low", "human_review"),
        ("robust_model_or_winsorization_sensitivity_only", "Use robust downstream methods or a sensitivity analysis; do not silently overwrite extremes.", ["task-specific justification"], "medium", "analysis_only"),
    ],
    "range_violation": [
        ("quarantine_and_source_verification", "The value conflicts with an explicit dataset-scoped valid range.", ["verified range metadata", "source record access"], "low", "human_review"),
    ],
    "categorical_representation_inconsistency": [
        ("canonicalize_with_explicit_mapping", "Equivalent labels differ only in representation.", ["approved canonical mapping", "raw-value preservation"], "low", "safe_if_confirmed"),
    ],
    "categorical_invalid_value": [
        ("map_or_quarantine_invalid_category", "The observed label is outside the declared category set.", ["approved category dictionary"], "medium", "human_review"),
    ],
    "unparseable_date": [
        ("source_verification_or_manual_parse_rule", "The date cannot be parsed reliably by the configured parser.", ["source format specification"], "medium", "human_review"),
    ],
    "date_format_inconsistency": [
        ("normalize_to_declared_date_format", "The value is parseable but does not match the declared target representation.", ["declared target format", "round-trip validation"], "low", "safe_if_confirmed"),
    ],
    "date_representation_diversity": [
        ("document_or_declare_target_date_format", "Multiple parseable representations are present, but no target format is declared.", ["reuse requirement"], "low", "metadata_action"),
    ],
    "format_inconsistency": [
        ("normalize_against_declared_pattern", "The value violates an explicit field pattern.", ["approved regex or format specification"], "low", "preview_then_authorize"),
    ],
    "exact_duplicate_row": [
        ("deduplicate_with_record-identity_policy", "An exact duplicate row exists; removal requires a declared record-identity policy.", ["stable identifier or duplicate policy"], "medium", "human_review"),
    ],
    "schema_metadata_gap": [
        ("complete_column_metadata", "Required column-level documentation is absent.", ["data owner or codebook"], "low", "metadata_action"),
    ],
    "dataset_metadata_gap": [
        ("complete_dataset_metadata", "Required dataset-level documentation is absent.", ["data owner", "repository record", "versioned documentation"], "low", "metadata_action"),
    ],
    "privacy_documentation_gap": [
        ("document_privacy_and_access_conditions", "Reuse readiness cannot be established without a privacy and access statement.", ["data controller or repository policy"], "low", "governance_action"),
    ],
    "direct_identifier_column": [
        ("restrict_or_remove_direct_identifiers_for_reuse", "A direct identifier requires access control, de-identification, or exclusion from the reusable dataset.", ["approved reuse purpose", "privacy review", "raw-data preservation"], "high", "human_review"),
    ],
}


def recommendations_for(issues: list[Issue]) -> list[Recommendation]:
    grouped: dict[tuple[str, str | None, str], list[Issue]] = defaultdict(list)
    value_sensitive = {
        "explicit_missing_token", "suspected_missing_token", "confirmed_disguised_missingness",
        "disguised_missingness_candidate", "categorical_invalid_value",
        "categorical_representation_inconsistency"
    }
    for issue in issues:
        value_key = repr(issue.value) if issue.issue_type in value_sensitive else "<grouped>"
        grouped[(issue.issue_type, issue.column, value_key)].append(issue)

    rows: list[Recommendation] = []
    for (issue_type, column, value_key), group in sorted(
        grouped.items(), key=lambda item: (item[0][0], "" if item[0][1] is None else item[0][1], item[0][2])
    ):
        entries = CATALOG.get(
            issue_type,
            [("manual_review", "No evidence-safe automated method is defined for this issue type.", ["expert review"], "medium", "human_review")],
        )
        group_id = stable_id("recommendation_group", issue_type, column, value_key, prefix="GROUP")
        issue_ids = [issue.issue_id for issue in group]
        for rank, (method, rationale, prerequisites, risk, mode) in enumerate(entries, start=1):
            rows.append(
                Recommendation(
                    issue_group_id=group_id,
                    issue_type=issue_type,
                    column=column,
                    affected_count=len(group),
                    issue_ids=issue_ids,
                    method=method,
                    rank=rank,
                    rationale=rationale,
                    prerequisites=prerequisites,
                    expected_risk=risk,
                    application_mode=mode,
                )
            )
    return rows
