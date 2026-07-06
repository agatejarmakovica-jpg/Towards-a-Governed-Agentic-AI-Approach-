from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
import os
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from .contract import DatasetContract
from .models import AgentResult


@dataclass(frozen=True)
class SemanticAdvisorConfig:
    enabled: bool = False
    api_key: str = ""
    model: str = "gpt-5-mini"
    include_non_sensitive_sample_values: bool = False
    max_sample_values_per_column: int = 3

    def resolved_key(self) -> str:
        return self.api_key.strip() or os.environ.get("OPENAI_API_KEY", "").strip()

    def public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("api_key", None)
        payload["api_key_source"] = "runtime_or_OPENAI_API_KEY"
        return payload


class ContractImprovement(BaseModel):
    column: str | None = None
    field: str
    proposed_value: str
    rationale: str
    evidence_required: str
    confidence: float = Field(ge=0.0, le=1.0)


class CrossColumnCheck(BaseModel):
    name: str
    columns: list[str]
    rule_description: str
    rationale: str
    evidence_required: str
    confidence: float = Field(ge=0.0, le=1.0)


class ReviewPriority(BaseModel):
    issue_type: str
    column: str | None = None
    priority: Literal["high", "medium", "low"]
    rationale: str


class SemanticAdvisorOutput(BaseModel):
    summary: str
    contract_improvements: list[ContractImprovement] = Field(default_factory=list)
    cross_column_checks: list[CrossColumnCheck] = Field(default_factory=list)
    review_priorities: list[ReviewPriority] = Field(default_factory=list)
    cautions: list[str] = Field(default_factory=list)


SENSITIVE_ROLES = {"email", "phone"}
SENSITIVE_LEVELS = {"direct_identifier", "quasi_identifier", "sensitive"}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if pd.isna(value):
        return None
    return str(value)


def build_advisor_payload(
    df: pd.DataFrame,
    contract: DatasetContract,
    agent_results: list[AgentResult],
    aggregate_metrics: dict[str, Any],
    *,
    include_non_sensitive_sample_values: bool = False,
    max_sample_values_per_column: int = 3,
) -> dict[str, Any]:
    issue_by_column: dict[str, Counter[str]] = defaultdict(Counter)
    issue_by_type: Counter[str] = Counter()
    for result in agent_results:
        for issue in result.issues:
            issue_by_type[issue.issue_type] += 1
            if issue.column:
                issue_by_column[issue.column][issue.issue_type] += 1

    columns: list[dict[str, Any]] = []
    for name, meta in contract.columns.items():
        if name not in df.columns:
            continue
        series = df[name]
        sensitive = meta.sensitivity in SENSITIVE_LEVELS or meta.role in SENSITIVE_ROLES
        payload: dict[str, Any] = {
            "name": name,
            "role": meta.role,
            "description": meta.description,
            "required": meta.required,
            "criticality_weight": meta.criticality_weight,
            "valid_range": meta.valid_range,
            "measurement_unit": meta.measurement_unit,
            "approved_allowed_value_count": len(meta.effective_allowed_values()),
            "has_canonical_mapping": bool(meta.canonical_mapping),
            "target_format": meta.target_format,
            "has_regex": bool(meta.regex),
            "sensitivity": meta.sensitivity,
            "non_null_count": int(series.notna().sum()),
            "missing_count": int(series.isna().sum()),
            "unique_count": int(series.nunique(dropna=True)),
            "issue_counts": dict(sorted(issue_by_column.get(name, {}).items())),
            "sample_values_included": False,
        }
        if include_non_sensitive_sample_values and not sensitive:
            values = series.dropna().drop_duplicates().head(max(0, max_sample_values_per_column))
            payload["sample_values"] = [_json_safe(value) for value in values.tolist()]
            payload["sample_values_included"] = True
        columns.append(payload)

    return {
        "purpose": "Advisory semantic review of a governed deterministic three-agent data-quality assessment.",
        "claim_boundary": {
            "advisor_must_not_confirm_errors": True,
            "advisor_must_not_change_metrics_or_verdict": True,
            "advisor_must_not_modify_data": True,
            "outputs_require_human_and_deterministic_validation": True,
        },
        "dataset": {
            "dataset_id": contract.dataset_id,
            "metadata": contract.dataset_metadata.model_dump(mode="json"),
            "primary_key": contract.primary_key,
            "columns": columns,
        },
        "assessment": {
            "evidence_mode": aggregate_metrics.get("evidence_mode"),
            "verdict": aggregate_metrics.get("verdict"),
            "C": aggregate_metrics.get("C"),
            "A_support": aggregate_metrics.get("A_support"),
            "A_validated": aggregate_metrics.get("A_validated"),
            "R": aggregate_metrics.get("R"),
            "issue_counts_by_type": dict(sorted(issue_by_type.items())),
        },
    }


def run_semantic_advisor(
    df: pd.DataFrame,
    contract: DatasetContract,
    agent_results: list[AgentResult],
    aggregate_metrics: dict[str, Any],
    config: SemanticAdvisorConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not config.enabled:
        return {"status": "disabled"}, {"enabled": False}
    api_key = config.resolved_key()
    if not api_key:
        raise ValueError("OPENAI_API_KEY or a runtime API key is required when SemanticAdvisor is enabled.")
    if not config.model.strip():
        raise ValueError("OpenAI model name is required.")

    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("Install the optional 'openai' package to enable SemanticAdvisor.") from exc

    payload = build_advisor_payload(
        df,
        contract,
        agent_results,
        aggregate_metrics,
        include_non_sensitive_sample_values=config.include_non_sensitive_sample_values,
        max_sample_values_per_column=config.max_sample_values_per_column,
    )
    system = (
        "You are SemanticAdvisor in C-DQC. Three deterministic scientific agents have already assessed "
        "completeness, accuracy, and reuse readiness. Use only aggregate metadata and findings. Do not "
        "confirm errors, change metrics, change the verdict, infer clinical truth, or recommend silent data edits. "
        "Return candidate contract improvements, testable cross-column rules, and HITL priorities with explicit "
        "evidence requirements."
    )
    client = OpenAI(api_key=api_key)
    response = client.responses.parse(
        model=config.model.strip(),
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
        ],
        text_format=SemanticAdvisorOutput,
        store=False,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise RuntimeError("OpenAI returned no structured SemanticAdvisor output.")
    result = parsed.model_dump(mode="json")
    usage = getattr(response, "usage", None)
    audit = {
        "enabled": True,
        "model": config.model.strip(),
        "store": False,
        "raw_rows_sent": False,
        "direct_identifier_values_sent": False,
        "include_non_sensitive_sample_values": config.include_non_sensitive_sample_values,
        "payload_column_count": len(payload["dataset"]["columns"]),
        "response_id": getattr(response, "id", None),
        "usage": usage.model_dump(mode="json") if usage is not None and hasattr(usage, "model_dump") else None,
        "api_key_persisted": False,
    }
    return result, audit
