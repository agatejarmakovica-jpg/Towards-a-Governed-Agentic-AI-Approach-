from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .accuracy_agent import AccuracyAgent
from .agent_utils import deduplicate_issues
from .completeness_agent import CompletenessAgent
from .contract import DatasetContract
from .io import dataset_profile
from .metrics import aggregate_scientific_metrics, detection_metrics
from .models import AgentResult, AgentTrace, Issue
from .reuse_agent import ReuseAgent
from .semantic_advisor import SemanticAdvisorConfig, run_semantic_advisor

COMPLETENESS_TYPES = {
    "explicit_missingness",
    "explicit_missing_token",
    "suspected_missing_token",
    "confirmed_disguised_missingness",
    "disguised_missingness_candidate",
}
ACCURACY_TYPES = {
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
    "reference_cell_mismatch",
    "exact_duplicate_row",
}
REUSE_TYPES = {
    "schema_metadata_gap",
    "dataset_metadata_gap",
    "privacy_documentation_gap",
    "direct_identifier_column",
    "required_reuse_column_missing",
    "machine_readability_gap",
}

# Evidence-only comparison findings are not detector predictions and must not
# be scored as false positives against a corruption manifest.
MANIFEST_EVALUABLE_TYPES = (COMPLETENESS_TYPES | ACCURACY_TYPES | REUSE_TYPES) - {
    "reference_cell_mismatch",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trace(
    sequence: int,
    name: str,
    objective: str,
    started: str,
    input_summary: dict[str, Any],
    output_summary: dict[str, Any],
    message: str = "",
    status: str = "completed",
) -> AgentTrace:
    return AgentTrace(
        sequence=sequence,
        agent=name,
        objective=objective,
        status=status,  # type: ignore[arg-type]
        started_at_utc=started,
        finished_at_utc=_now(),
        input_summary=input_summary,
        output_summary=output_summary,
        message=message,
    )


def _evidence_mode(reference_df: pd.DataFrame | None, truth_manifest: pd.DataFrame | None) -> str:
    if reference_df is not None and truth_manifest is not None:
        return "REFERENCE_AND_MANIFEST"
    if reference_df is not None:
        return "CLEAN_REFERENCE"
    if truth_manifest is not None:
        return "CORRUPTION_MANIFEST"
    return "CONTRACT_ONLY"


def _verify_issues(df: pd.DataFrame, issues: list[Issue]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for issue in issues:
        if issue.issue_id in seen:
            errors.append(f"Duplicate issue_id: {issue.issue_id}")
        seen.add(issue.issue_id)
        if issue.scope == "cell":
            if issue.column not in df.columns:
                errors.append(f"Cell issue references unknown column: {issue.issue_id}")
            if issue.row_index is None or issue.row_index not in df.index:
                errors.append(f"Cell issue references unknown row: {issue.issue_id}")
        if issue.scope == "column" and issue.column not in df.columns:
            errors.append(f"Column issue references unknown column: {issue.issue_id}")
        if issue.evidence_status == "candidate" and issue.automatic_action_allowed:
            errors.append(f"Candidate issue cannot authorize automatic action: {issue.issue_id}")
    return errors


def _attach_per_agent_manifest_metrics(
    results: list[AgentResult],
    truth_manifest: pd.DataFrame | None,
    df: pd.DataFrame,
) -> None:
    if truth_manifest is None:
        return
    domains = {
        "CompletenessAgent": COMPLETENESS_TYPES,
        "AccuracyAgent": ACCURACY_TYPES,
        "ReuseAgent": REUSE_TYPES,
    }
    records = truth_manifest.to_dict(orient="records")
    for result in results:
        domain = domains[result.agent_name]
        filtered = [record for record in records if str(record.get("issue_type")) in domain]
        predicted = [
            issue
            for issue in result.issues
            if issue.issue_type in domain and issue.issue_type != "reference_cell_mismatch"
        ]
        result.metrics["agent_validation_metrics"] = detection_metrics(predicted, filtered, df=df)


def run_agentic_assessment(
    df: pd.DataFrame,
    contract: DatasetContract,
    *,
    contract_warnings: list[str] | None = None,
    reference_df: pd.DataFrame | None = None,
    truth_manifest: pd.DataFrame | None = None,
    semantic_config: SemanticAdvisorConfig | None = None,
) -> dict[str, Any]:
    traces: list[AgentTrace] = []
    warnings = list(contract_warnings or [])
    evidence_mode = _evidence_mode(reference_df, truth_manifest)

    started = _now()
    profile = dataset_profile(df)
    traces.append(
        _trace(
            1,
            "IntakeStage",
            "Validate the contract, preserve the immutable input, and prepare the evidence context.",
            started,
            {"dataset_id": contract.dataset_id},
            {
                "profile": profile,
                "contract_warnings": warnings,
                "evidence_mode": evidence_mode,
                "reference_supplied": reference_df is not None,
                "manifest_supplied": truth_manifest is not None,
            },
        )
    )

    started = _now()
    completeness = CompletenessAgent().run(df, contract, evidence_mode=evidence_mode)
    traces.append(
        _trace(
            2,
            "CompletenessAgent",
            "Independently analyze missingness, quantify completeness, validate imputation candidates, and recommend treatments.",
            started,
            {"rows": len(df), "columns": len(df.columns), "evidence_mode": evidence_mode},
            {
                "issue_count": len(completeness.issues),
                "issue_types": dict(sorted(Counter(issue.issue_type for issue in completeness.issues).items())),
                "metrics": completeness.metrics,
                "tools": completeness.execution_details.get("tools", []),
            },
        )
    )

    started = _now()
    accuracy = AccuracyAgent().run(
        df,
        contract,
        evidence_mode=evidence_mode,
        reference_df=reference_df,
        truth_manifest=truth_manifest,
    )
    traces.append(
        _trace(
            3,
            "AccuracyAgent",
            "Independently analyze contract validity, logical consistency, anomaly candidates, and validated reference/manifest evidence.",
            started,
            {
                "rows": len(df),
                "columns": len(df.columns),
                "evidence_mode": evidence_mode,
                "clean_reference_supplied": reference_df is not None,
                "corruption_manifest_supplied": truth_manifest is not None,
            },
            {
                "issue_count": len(accuracy.issues),
                "issue_types": dict(sorted(Counter(issue.issue_type for issue in accuracy.issues).items())),
                "metrics": accuracy.metrics,
                "tools": accuracy.execution_details.get("tools", []),
            },
        )
    )

    started = _now()
    reuse = ReuseAgent().run(df, contract, evidence_mode=evidence_mode)
    traces.append(
        _trace(
            4,
            "ReuseAgent",
            "Independently analyze documentation, schema, standardization, privacy, reproducibility, and machine readability.",
            started,
            {"rows": len(df), "columns": len(df.columns), "evidence_mode": evidence_mode},
            {
                "issue_count": len(reuse.issues),
                "issue_types": dict(sorted(Counter(issue.issue_type for issue in reuse.issues).items())),
                "metrics": reuse.metrics,
                "tools": reuse.execution_details.get("tools", []),
            },
        )
    )

    agent_results = [completeness, accuracy, reuse]
    _attach_per_agent_manifest_metrics(agent_results, truth_manifest, df)

    started = _now()
    all_issues = deduplicate_issues([issue for result in agent_results for issue in result.issues])
    aggregate_metrics = aggregate_scientific_metrics(
        df,
        completeness,
        accuracy,
        reuse,
        contract,
        all_issues,
    )
    if truth_manifest is not None:
        manifest_predictions = [
            issue for issue in all_issues if issue.issue_type in MANIFEST_EVALUABLE_TYPES
        ]
        aggregate_metrics["manifest_detection_metrics"] = detection_metrics(
            manifest_predictions,
            truth_manifest.to_dict(orient="records"),
            df=df,
        )
    else:
        aggregate_metrics["manifest_detection_metrics"] = None
    traces.append(
        _trace(
            5,
            "CoordinatorStage",
            "Integrate the three independent agent results without changing their evidence or metrics.",
            started,
            {
                "scientific_agent_count": 3,
                "agent_issue_counts": {result.agent_name: len(result.issues) for result in agent_results},
            },
            {
                "total_issue_count": len(all_issues),
                "evidence_mode": aggregate_metrics["evidence_mode"],
                "DQI_support": aggregate_metrics["DQI_support"],
                "DQI_validated": aggregate_metrics["DQI_validated"],
                "verdict": aggregate_metrics["verdict"],
            },
        )
    )

    started = _now()
    verification_errors = _verify_issues(df, all_issues)
    if len(agent_results) != 3 or {result.agent_name for result in agent_results} != {
        "CompletenessAgent",
        "AccuracyAgent",
        "ReuseAgent",
    }:
        verification_errors.append("The scientific core must contain exactly the three required specialist agents.")
    traces.append(
        _trace(
            6,
            "VerifierStage",
            "Enforce issue-coordinate integrity, evidence boundaries, and the exactly-three-agent architecture.",
            started,
            {"issue_count": len(all_issues), "scientific_agent_count": len(agent_results)},
            {"verification_errors": verification_errors, "passed": not verification_errors},
            message="All governance checks passed." if not verification_errors else "Verification failed.",
            status="completed" if not verification_errors else "failed",
        )
    )
    if verification_errors:
        raise RuntimeError("Scientific-core verification failed: " + "; ".join(verification_errors))

    semantic_output = {"status": "disabled"}
    semantic_audit = {"enabled": False}
    if semantic_config and semantic_config.enabled:
        try:
            semantic_output, semantic_audit = run_semantic_advisor(
                df,
                contract,
                agent_results,
                aggregate_metrics,
                semantic_config,
            )
        except Exception as exc:
            semantic_output = {"status": "failed", "error": str(exc)}
            semantic_audit = {
                "enabled": True,
                "failed": True,
                "error": str(exc),
                "deterministic_assessment_completed": True,
                "metrics_and_verdict_unchanged": True,
            }

    return {
        "profile": profile,
        "issues": all_issues,
        "agent_results": {result.agent_name: result for result in agent_results},
        "issues_by_agent": {result.agent_name: result.issues for result in agent_results},
        "quality_support_metrics": aggregate_metrics,
        "agent_trace": traces,
        "contract_warnings": warnings,
        "semantic_advisor": semantic_output,
        "semantic_advisor_audit": semantic_audit,
    }
