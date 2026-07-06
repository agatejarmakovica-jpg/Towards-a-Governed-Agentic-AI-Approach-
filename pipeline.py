from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from . import __version__
from .agents import run_agentic_assessment
from .contract import generate_draft_contract, load_contract, save_contract
from .io import read_table, write_table
from .models import RemediationProposal
from .recommend import recommendations_for
from .remediation import (
    apply_reviewed_proposals,
    load_review_decisions,
    proposals_for,
    write_review_queue,
)
from .reporting import write_html_report
from .semantic_advisor import SemanticAdvisorConfig
from .utils import read_json, sha256_file, write_json, zip_directory


def _read_table(path: Path) -> pd.DataFrame:
    return read_table(path)


def _write_issue_files(output_dir: Path, issues: list[Any]) -> None:
    issue_rows: list[dict[str, Any]] = []
    for issue in issues:
        row = issue.to_dict()
        row["evidence"] = json.dumps(row["evidence"], ensure_ascii=False, sort_keys=True, default=str)
        issue_rows.append(row)
    columns = [
        "issue_id",
        "issue_type",
        "scope",
        "column",
        "row_index",
        "value",
        "evidence_status",
        "confidence",
        "severity",
        "evidence",
        "automatic_action_allowed",
    ]
    pd.DataFrame(issue_rows, columns=columns).to_csv(output_dir / "issues.csv", index=False)
    write_json(output_dir / "issues.json", [issue.to_dict() for issue in issues])


def _write_recommendation_files(output_dir: Path, recommendations: list[Any]) -> None:
    rows: list[dict[str, Any]] = []
    for recommendation in recommendations:
        row = recommendation.to_dict()
        row["prerequisites"] = json.dumps(row["prerequisites"], ensure_ascii=False)
        row["issue_ids"] = json.dumps(row["issue_ids"], ensure_ascii=False)
        rows.append(row)
    columns = [
        "issue_group_id",
        "issue_type",
        "column",
        "affected_count",
        "issue_ids",
        "method",
        "rank",
        "rationale",
        "prerequisites",
        "expected_risk",
        "application_mode",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(output_dir / "recommendations.csv", index=False)
    write_json(output_dir / "recommendations.json", [recommendation.to_dict() for recommendation in recommendations])


def _make_run_id(dataset_id: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{dataset_id}-{stamp}-{uuid4().hex[:8]}"


def _write_agent_outputs(output_dir: Path, result: dict[str, Any]) -> None:
    file_prefix = {
        "CompletenessAgent": "completeness",
        "AccuracyAgent": "accuracy",
        "ReuseAgent": "reuse",
    }
    all_availability: list[dict[str, Any]] = []
    validation: dict[str, Any] = {}
    for agent_name, agent_result in result["agent_results"].items():
        prefix = file_prefix[agent_name]
        payload = agent_result.to_dict()
        write_json(output_dir / f"{prefix}_agent_result.json", payload)
        write_json(output_dir / f"{prefix}_metrics.json", agent_result.metrics)
        write_json(
            output_dir / f"{prefix}_recommendations.json",
            [item.to_dict() for item in agent_result.recommendations],
        )
        write_json(output_dir / f"{agent_name.casefold()}_issues.json", [issue.to_dict() for issue in agent_result.issues])
        all_availability.extend(item.to_dict() for item in agent_result.metric_availability)
        if "agent_validation_metrics" in agent_result.metrics:
            validation[agent_name] = agent_result.metrics["agent_validation_metrics"]
    write_json(output_dir / "metric_availability.json", all_availability)
    write_json(output_dir / "agent_validation_metrics.json", validation)


def run_assessment(
    input_path: Path,
    output_dir: Path,
    config_path: Path | None = None,
    reference_path: Path | None = None,
    truth_manifest_path: Path | None = None,
    *,
    require_approved_contract: bool = False,
    create_zip: bool = True,
    semantic_config: SemanticAdvisorConfig | None = None,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(input_path)
    if config_path:
        contract, contract_warnings = load_contract(
            Path(config_path),
            dataset_columns=[str(column) for column in df.columns],
            require_approved=require_approved_contract,
            dataset_id_fallback=input_path.stem,
        )
    else:
        if require_approved_contract:
            raise ValueError("An approved dataset contract JSON is required in strict product mode.")
        contract = generate_draft_contract(df, dataset_id=input_path.stem, title=input_path.stem)
        contract_warnings = ["No contract was supplied. A draft contract was generated and must be completed and approved."]

    # Evidence must be loaded before the scientific agents run.
    source_hash_before = sha256_file(input_path)
    reference_hash = sha256_file(Path(reference_path)) if reference_path else None
    if reference_hash is not None and reference_hash == source_hash_before:
        raise ValueError(
            "The dataset and clean reference are byte-identical. "
            "This would compare the dataset with itself and produce a false validation result. "
            "Upload the messy/input dataset as Dataset and the cleaned/reference dataset as Clean reference."
        )
    reference_df = read_table(Path(reference_path)) if reference_path else None
    truth_manifest = pd.read_csv(Path(truth_manifest_path)) if truth_manifest_path else None

    run_id = _make_run_id(contract.dataset_id)
    created_at = datetime.now(timezone.utc).isoformat()

    snapshot_suffix = input_path.suffix.casefold()
    if snapshot_suffix == ".xls":
        snapshot_suffix = ".xlsx"
    source_snapshot = output_dir / f"input_snapshot{snapshot_suffix}"
    write_table(df, source_snapshot)
    save_contract(output_dir / "contract_effective.json", contract)

    result = run_agentic_assessment(
        df,
        contract,
        contract_warnings=contract_warnings,
        reference_df=reference_df,
        truth_manifest=truth_manifest,
        semantic_config=semantic_config,
    )
    issues = result["issues"]
    metrics = result["quality_support_metrics"]
    traces = result["agent_trace"]
    recommendations = recommendations_for(issues)
    proposals = proposals_for(df, issues, contract)

    _write_issue_files(output_dir, issues)
    _write_recommendation_files(output_dir, recommendations)
    write_review_queue(output_dir, proposals)
    write_json(output_dir / "dataset_profile.json", result["profile"])
    write_json(output_dir / "quality_support_metrics.json", metrics)
    write_json(output_dir / "aggregate_metrics.json", metrics)
    write_json(output_dir / "agent_trace.json", [trace.to_dict() for trace in traces])
    _write_agent_outputs(output_dir, result)

    reference_metrics = result["agent_results"]["AccuracyAgent"].metrics.get("reference_metrics")
    if reference_metrics is not None:
        write_json(output_dir / "reference_exact_metrics.json", reference_metrics)

    benchmark_metrics = metrics.get("manifest_detection_metrics")
    if benchmark_metrics is not None:
        write_json(output_dir / "detection_benchmark_metrics.json", benchmark_metrics)
        write_json(output_dir / "statistical_validation.json", benchmark_metrics)

    write_json(output_dir / "semantic_advisor.json", result["semantic_advisor"])
    write_json(output_dir / "semantic_advisor_audit.json", result["semantic_advisor_audit"])

    source_hash_after = sha256_file(input_path)
    if source_hash_before != source_hash_after:
        raise RuntimeError("Source file hash changed during assessment. Execution stopped.")

    manifest = {
        "software": "C-DQC Scientific Product",
        "version": __version__,
        "run_id": run_id,
        "created_at_utc": created_at,
        "dataset_id": contract.dataset_id,
        "scientific_agent_count": 3,
        "scientific_agents": ["CompletenessAgent", "AccuracyAgent", "ReuseAgent"],
        "evidence_mode": metrics.get("evidence_mode"),
        "contract_status": contract.approval.status,
        "contract_approved_by": contract.approval.approved_by,
        "input_file": str(input_path),
        "input_sha256": source_hash_before,
        "source_unchanged_after_run": True,
        "source_snapshot": str(source_snapshot),
        "source_snapshot_sha256": sha256_file(source_snapshot),
        "config_file": str(config_path) if config_path else None,
        "config_sha256": sha256_file(Path(config_path)) if config_path else None,
        "effective_contract_sha256": sha256_file(output_dir / "contract_effective.json"),
        "reference_file": str(reference_path) if reference_path else None,
        "reference_sha256": reference_hash,
        "truth_manifest_file": str(truth_manifest_path) if truth_manifest_path else None,
        "truth_manifest_sha256": sha256_file(Path(truth_manifest_path)) if truth_manifest_path else None,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "issue_count": len(issues),
        "recommendation_count": len(recommendations),
        "review_proposal_count": len(proposals),
        "verdict": metrics.get("verdict"),
        "DQI_support": metrics.get("DQI_support"),
        "DQI_validated": metrics.get("DQI_validated"),
        "contract_warnings": contract_warnings,
        "semantic_advisor": result["semantic_advisor_audit"],
        "primary_output_contract": {
            "source_snapshot_preserves_original_schema": True,
            "audit_information_is_separate": True,
            "automatic_data_repair": False,
            "curated_output_requires_named_human_approval": True,
            "candidate_findings_never_authorize_changes": True,
        },
        "scientific_claim_boundary": {
            "contract_only_mode_does_not_claim_ground_truth_accuracy": True,
            "candidate_issues_are_not_confirmed_errors": True,
            "reference_or_injection_manifest_required_for_validated_accuracy": True,
            "before_after_comparison_requires_same_contract_policy_and_evidence_mode": True,
            "semantic_advisor_cannot_change_metrics_verdict_or_data": True,
        },
    }
    write_json(output_dir / "run_manifest.json", manifest)

    write_html_report(
        output_dir / "report.html",
        dataset_id=contract.dataset_id,
        metrics=metrics,
        issues=issues,
        recommendations=recommendations,
        proposals=proposals,
        traces=traces,
        contract_status=contract.approval.status,
        contract_warnings=contract_warnings,
        manifest=manifest,
    )

    archive_path = None
    if create_zip:
        archive_path = zip_directory(output_dir, output_dir.parent / f"{output_dir.name}_bundle")

    return {
        "manifest": manifest,
        "quality_support_metrics": metrics,
        "reference_metrics": reference_metrics,
        "benchmark_metrics": benchmark_metrics,
        "semantic_advisor": result["semantic_advisor"],
        "output_dir": str(output_dir),
        "archive_path": str(archive_path) if archive_path else None,
    }


def apply_human_review(
    input_path: Path,
    assessment_dir: Path,
    review_path: Path,
    output_dir: Path,
    *,
    allow_medium_risk: bool = False,
    reference_path: Path | None = None,
    truth_manifest_path: Path | None = None,
    create_zip: bool = True,
) -> dict[str, Any]:
    input_path = Path(input_path)
    assessment_dir = Path(assessment_dir)
    review_path = Path(review_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = read_json(assessment_dir / "run_manifest.json")
    expected_hash = source_manifest.get("input_sha256")
    actual_hash = sha256_file(input_path)
    if expected_hash != actual_hash:
        raise ValueError("The supplied source dataset does not match the assessment run hash.")

    df = read_table(input_path)
    contract, warnings = load_contract(
        assessment_dir / "contract_effective.json",
        dataset_columns=[str(column) for column in df.columns],
        require_approved=False,
        dataset_id_fallback=input_path.stem,
    )
    proposal_payload = json.loads((assessment_dir / "review_queue.json").read_text(encoding="utf-8"))
    proposals = [RemediationProposal(**record) for record in proposal_payload]
    decisions = load_review_decisions(review_path)
    curated, applied, reviewed = apply_reviewed_proposals(
        df,
        proposals,
        decisions,
        allow_medium_risk=allow_medium_risk,
    )

    suffix = input_path.suffix.casefold()
    if suffix == ".xls":
        suffix = ".xlsx"
    curated_path = output_dir / f"curated_dataset{suffix}"
    write_table(curated, curated_path)
    write_json(output_dir / "applied_changes.json", applied)
    pd.DataFrame(applied).to_csv(output_dir / "applied_changes.csv", index=False)
    write_json(output_dir / "reviewed_proposals.json", [proposal.to_dict() for proposal in reviewed])

    reference_df = read_table(Path(reference_path)) if reference_path else None
    truth_manifest = pd.read_csv(Path(truth_manifest_path)) if truth_manifest_path else None
    before_metrics = read_json(assessment_dir / "quality_support_metrics.json")
    after_result = run_agentic_assessment(
        curated,
        contract,
        contract_warnings=warnings,
        reference_df=reference_df,
        truth_manifest=truth_manifest,
    )
    after_metrics = after_result["quality_support_metrics"]
    if before_metrics.get("evidence_mode") != after_metrics.get("evidence_mode"):
        raise ValueError("Before and after metrics must use the same evidence mode.")

    comparison = {
        "same_contract": True,
        "same_evidence_mode": True,
        "evidence_mode": after_metrics.get("evidence_mode"),
        "before": before_metrics,
        "after": after_metrics,
        "C_before": before_metrics.get("C"),
        "C_after": after_metrics.get("C"),
        "delta_C": float(after_metrics.get("C", 0.0)) - float(before_metrics.get("C", 0.0)),
        "A_before": before_metrics.get("A_validated") if before_metrics.get("A_validated") is not None else before_metrics.get("A_support"),
        "A_after": after_metrics.get("A_validated") if after_metrics.get("A_validated") is not None else after_metrics.get("A_support"),
        "delta_A": float(after_metrics.get("accuracy_support", 0.0)) - float(before_metrics.get("accuracy_support", 0.0)),
        "R_before": before_metrics.get("R"),
        "R_after": after_metrics.get("R"),
        "delta_R": float(after_metrics.get("R", 0.0)) - float(before_metrics.get("R", 0.0)),
        "DQI_before": before_metrics.get("quality_support_index"),
        "DQI_after": after_metrics.get("quality_support_index"),
        "delta_DQI": float(after_metrics.get("quality_support_index", 0.0)) - float(before_metrics.get("quality_support_index", 0.0)),
        "delta": {
            "completeness_support": float(after_metrics.get("completeness_support", 0.0)) - float(before_metrics.get("completeness_support", 0.0)),
            "accuracy_support": float(after_metrics.get("accuracy_support", 0.0)) - float(before_metrics.get("accuracy_support", 0.0)),
            "reuse_readiness_support": float(after_metrics.get("reuse_readiness_support", 0.0)) - float(before_metrics.get("reuse_readiness_support", 0.0)),
            "quality_support_index": float(after_metrics.get("quality_support_index", 0.0)) - float(before_metrics.get("quality_support_index", 0.0)),
        },
        "applied_change_count": len(applied),
    }
    write_json(output_dir / "before_after_comparison.json", comparison)
    write_json(output_dir / "before_after_metrics.json", comparison)
    write_json(output_dir / "post_review_agent_trace.json", [trace.to_dict() for trace in after_result["agent_trace"]])
    write_json(output_dir / "post_review_issues.json", [issue.to_dict() for issue in after_result["issues"]])
    for name, agent_result in after_result["agent_results"].items():
        write_json(output_dir / f"post_review_{name.casefold()}_result.json", agent_result.to_dict())

    manifest = {
        "software": "C-DQC Scientific Product",
        "version": __version__,
        "operation": "apply_human_review",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_assessment_run_id": source_manifest.get("run_id"),
        "source_input_sha256": actual_hash,
        "review_file": str(review_path),
        "review_file_sha256": sha256_file(review_path),
        "curated_dataset": str(curated_path),
        "curated_dataset_sha256": sha256_file(curated_path),
        "applied_change_count": len(applied),
        "allow_medium_risk": allow_medium_risk,
        "source_file_modified": False,
        "post_review_verdict": after_metrics.get("verdict"),
        "post_review_DQI": after_metrics.get("quality_support_index"),
    }
    write_json(output_dir / "review_application_manifest.json", manifest)

    archive_path = None
    if create_zip:
        archive_path = zip_directory(output_dir, output_dir.parent / f"{output_dir.name}_bundle")
    return {
        "manifest": manifest,
        "comparison": comparison,
        "reference_comparison": after_result["agent_results"]["AccuracyAgent"].metrics.get("reference_metrics"),
        "output_dir": str(output_dir),
        "archive_path": str(archive_path) if archive_path else None,
    }
