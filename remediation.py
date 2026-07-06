from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .contract import DatasetContract
from .models import Issue, RemediationProposal
from .utils import stable_id, write_json


def _safe_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _proposal(
    issue: Issue,
    *,
    proposed_value: Any,
    action: str,
    risk: str,
    executable: bool,
    rationale: str,
    reversible: bool = True,
) -> RemediationProposal:
    return RemediationProposal(
        proposal_id=stable_id("proposal", issue.issue_id, action, repr(proposed_value), prefix="PROP"),
        issue_id=issue.issue_id,
        issue_type=issue.issue_type,
        scope=issue.scope,
        row_index=issue.row_index,
        column=issue.column,
        current_value=issue.value,
        proposed_value=proposed_value,
        action=action,
        risk=risk,  # type: ignore[arg-type]
        reversible=reversible,
        executable=executable,
        rationale=rationale,
    )


def proposals_for(df: pd.DataFrame, issues: list[Issue], contract: DatasetContract) -> list[RemediationProposal]:
    proposals: list[RemediationProposal] = []
    for issue in issues:
        column_meta = contract.columns.get(issue.column) if issue.column else None
        protected = bool(column_meta and column_meta.protected_from_change)

        if issue.issue_type == "explicit_missing_token":
            proposals.append(
                _proposal(
                    issue,
                    proposed_value=None,
                    action="set_null",
                    risk="low",
                    executable=not protected,
                    rationale="The contract explicitly declares this token as missing.",
                )
            )
        elif issue.issue_type == "confirmed_disguised_missingness":
            proposals.append(
                _proposal(
                    issue,
                    proposed_value=None,
                    action="set_null",
                    risk="low",
                    executable=not protected,
                    rationale="The approved contract declares this sentinel value as missing.",
                )
            )
        elif issue.issue_type == "categorical_representation_inconsistency":
            canonical = issue.evidence.get("canonical")
            if column_meta and column_meta.canonical_mapping:
                raw = str(issue.value)
                canonical = column_meta.canonical_mapping.get(raw, column_meta.canonical_mapping.get(raw.strip(), canonical))
            proposals.append(
                _proposal(
                    issue,
                    proposed_value=canonical,
                    action="replace_value",
                    risk="low",
                    executable=canonical is not None and not protected,
                    rationale="The proposed value is the contract-backed or observed canonical representation.",
                )
            )
        elif issue.issue_type == "date_format_inconsistency":
            target = column_meta.target_format if column_meta else None
            proposed: Any = None
            executable = False
            if target and column_meta and column_meta.allow_format_normalization and not protected:
                # A target format alone is insufficient for safe normalization because values such as
                # 02/01/2024 are ambiguous. At least one approved source format must parse exactly.
                parsed_candidates: list[pd.Timestamp] = []
                for source_format in column_meta.source_formats:
                    try:
                        parsed_candidate = pd.to_datetime(issue.value, errors="raise", format=source_format)
                        if not _safe_missing(parsed_candidate):
                            parsed_candidates.append(parsed_candidate)
                    except (TypeError, ValueError):
                        continue
                unique_candidates = {candidate.isoformat() for candidate in parsed_candidates}
                if len(unique_candidates) == 1:
                    proposed = parsed_candidates[0].strftime(target)
                    executable = True
            proposals.append(
                _proposal(
                    issue,
                    proposed_value=proposed,
                    action="replace_value",
                    risk="low",
                    executable=executable,
                    rationale="Normalize a parseable date to the approved target format after human authorization.",
                )
            )
        elif issue.issue_type == "exact_duplicate_row":
            proposals.append(
                _proposal(
                    issue,
                    proposed_value=None,
                    action="drop_row",
                    risk="medium",
                    executable=True,
                    rationale="Remove the duplicate row only when the reviewer confirms the record-identity policy.",
                )
            )
        elif issue.issue_type in {"range_violation", "categorical_invalid_value", "format_inconsistency", "unparseable_date"}:
            proposals.append(
                _proposal(
                    issue,
                    proposed_value=None,
                    action="manual_source_verification",
                    risk="medium",
                    executable=False,
                    rationale="The issue is confirmed, but no evidence-safe replacement is derivable from the contract.",
                )
            )
        elif issue.issue_type in {"disguised_missingness_candidate", "numeric_extreme_candidate", "suspected_missing_token"}:
            proposals.append(
                _proposal(
                    issue,
                    proposed_value=None,
                    action="manual_evidence_review",
                    risk="high",
                    executable=False,
                    rationale="A candidate finding must not change data without dataset-specific evidence.",
                )
            )
        elif issue.scope in {"column", "dataset"}:
            proposals.append(
                _proposal(
                    issue,
                    proposed_value=None,
                    action="metadata_or_governance_action",
                    risk="low",
                    executable=False,
                    rationale="This finding requires metadata or governance work rather than a cell edit.",
                )
            )

    unique = {proposal.proposal_id: proposal for proposal in proposals}
    return sorted(
        unique.values(),
        key=lambda p: (p.risk, p.issue_type, "" if p.column is None else p.column, -1 if p.row_index is None else p.row_index),
    )


def review_queue_rows(proposals: list[RemediationProposal]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for proposal in proposals:
        row = proposal.to_dict()
        row["current_value"] = json.dumps(row["current_value"], ensure_ascii=False, default=str)
        row["proposed_value"] = json.dumps(row["proposed_value"], ensure_ascii=False, default=str)
        rows.append(row)
    return rows


def write_review_queue(output_dir: Path, proposals: list[RemediationProposal]) -> None:
    write_json(output_dir / "review_queue.json", [proposal.to_dict() for proposal in proposals])
    columns = [
        "proposal_id",
        "issue_id",
        "issue_type",
        "scope",
        "row_index",
        "column",
        "current_value",
        "proposed_value",
        "action",
        "risk",
        "reversible",
        "executable",
        "rationale",
        "decision",
        "reviewer",
        "reviewer_comment",
    ]
    pd.DataFrame(review_queue_rows(proposals), columns=columns).to_csv(output_dir / "review_queue.csv", index=False)


def _decode_json_cell(value: Any) -> Any:
    if value is None or _safe_missing(value):
        return None
    text = str(value)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def load_review_decisions(review_path: Path) -> dict[str, dict[str, Any]]:
    if review_path.suffix.casefold() == ".json":
        payload = json.loads(review_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Review JSON must contain a list of proposal decisions.")
        rows = payload
    else:
        rows = pd.read_csv(review_path, keep_default_na=False).to_dict(orient="records")
    decisions: dict[str, dict[str, Any]] = {}
    for row in rows:
        proposal_id = str(row.get("proposal_id", "")).strip()
        if not proposal_id:
            continue
        decision = str(row.get("decision", "pending")).strip().casefold()
        if decision not in {"pending", "approve", "reject", "defer"}:
            raise ValueError(f"Invalid review decision '{decision}' for {proposal_id}")
        decisions[proposal_id] = {
            "decision": decision,
            "reviewer": str(row.get("reviewer", "")).strip(),
            "reviewer_comment": str(row.get("reviewer_comment", "")).strip(),
        }
    return decisions


def apply_reviewed_proposals(
    df: pd.DataFrame,
    proposals: list[RemediationProposal],
    decisions: dict[str, dict[str, Any]],
    *,
    allow_medium_risk: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[RemediationProposal]]:
    curated = df.copy(deep=True)
    applied: list[dict[str, Any]] = []
    reviewed: list[RemediationProposal] = []
    rows_to_drop: set[int] = set()

    for proposal in proposals:
        review = decisions.get(proposal.proposal_id, {"decision": "pending", "reviewer": "", "reviewer_comment": ""})
        updated = replace(
            proposal,
            decision=review["decision"],
            reviewer=review["reviewer"],
            reviewer_comment=review["reviewer_comment"],
        )
        reviewed.append(updated)
        if updated.decision != "approve":
            continue
        if not updated.reviewer:
            raise ValueError(f"Approved proposal {updated.proposal_id} requires a reviewer name.")
        if not updated.executable:
            raise ValueError(f"Proposal {updated.proposal_id} is not executable and cannot be approved for application.")
        if updated.risk == "high":
            raise ValueError(f"High-risk proposal {updated.proposal_id} cannot be applied by the product.")
        if updated.risk == "medium" and not allow_medium_risk:
            raise ValueError(
                f"Medium-risk proposal {updated.proposal_id} requires allow_medium_risk=True and explicit reviewer approval."
            )

        if updated.action == "set_null":
            assert updated.column is not None and updated.row_index is not None
            old = curated.at[updated.row_index, updated.column]
            curated.at[updated.row_index, updated.column] = pd.NA
            applied.append({**updated.to_dict(), "applied_old_value": old, "applied_new_value": None})
        elif updated.action == "replace_value":
            assert updated.column is not None and updated.row_index is not None
            old = curated.at[updated.row_index, updated.column]
            curated.at[updated.row_index, updated.column] = updated.proposed_value
            applied.append({**updated.to_dict(), "applied_old_value": old, "applied_new_value": updated.proposed_value})
        elif updated.action == "drop_row":
            assert updated.row_index is not None
            rows_to_drop.add(updated.row_index)
            applied.append({**updated.to_dict(), "applied_old_value": "row", "applied_new_value": None})
        else:
            raise ValueError(f"Unsupported executable action: {updated.action}")

    if rows_to_drop:
        curated = curated.drop(index=sorted(rows_to_drop)).reset_index(drop=True)
    return curated, applied, reviewed
