from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .agent_utils import infer_role
from .utils import read_json, write_json

ApprovalStatus = Literal["draft", "approved", "retired"]
RowAlignmentPolicy = Literal["require_key", "position", "position_with_warning"]


class Approval(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ApprovalStatus = "draft"
    approved_by: str | None = None
    approved_at_utc: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def approved_requires_identity(self) -> "Approval":
        if self.status == "approved":
            if not self.approved_by or not str(self.approved_by).strip():
                raise ValueError("approval.approved_by is required when approval.status is 'approved'")
            if not self.approved_at_utc:
                raise ValueError("approval.approved_at_utc is required when approval.status is 'approved'")
        return self


class DatasetMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    title: str = ""
    description: str = ""
    provenance: str = ""
    version: str = ""
    license: str = ""
    intended_reuse: str = ""
    privacy_status: str = ""


class CrossFieldRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str
    description: str
    severity: Literal["confirmed", "supported", "candidate"] = "confirmed"
    left_column: str | None = None
    operator: Literal["<", "<=", "==", "!=", ">=", ">", "in", "not_in"] | None = None
    right_column: str | None = None
    right_value: Any = None
    expression: str | None = None

    @model_validator(mode="after")
    def validate_operands(self) -> "CrossFieldRule":
        if not self.expression and not (self.left_column and self.operator):
            raise ValueError("cross-field rule requires expression or left_column/operator")
        if self.right_column is None and self.right_value is None and not self.expression:
            raise ValueError("cross-field rule requires right_column or right_value")
        return self


class ReusePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    documentation_weight: float = 0.20
    schema_weight: float = 0.20
    standardization_weight: float = 0.20
    privacy_weight: float = 0.20
    machine_readability_weight: float = 0.20

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "ReusePolicy":
        weights = [
            self.documentation_weight,
            self.schema_weight,
            self.standardization_weight,
            self.privacy_weight,
            self.machine_readability_weight,
        ]
        if any(value < 0 for value in weights):
            raise ValueError("reuse-policy weights must be non-negative")
        if abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("reuse-policy weights must sum to 1.0")
        return self


class ColumnContract(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    description: str = ""
    required: bool = False
    criticality_weight: float = 1.0
    valid_range: tuple[float, float] | None = None
    measurement_unit: str | None = None
    numeric_tolerance: float = 0.0
    date_tolerance_days: int = 0
    case_sensitive: bool = False

    # Backward-compatible field. New drafts do not auto-populate it.
    allowed_values: list[Any] = Field(default_factory=list)
    observed_values_sample: list[Any] = Field(default_factory=list)
    suggested_allowed_values: list[Any] = Field(default_factory=list)
    approved_allowed_values: list[Any] = Field(default_factory=list)

    sentinel_values: dict[str, str] = Field(default_factory=dict)
    missing_semantics: dict[str, str] = Field(default_factory=dict)
    canonical_mapping: dict[str, Any] = Field(default_factory=dict)
    target_format: str | None = None
    source_formats: list[str] = Field(default_factory=list)
    allow_format_normalization: bool = False
    regex: str | None = None
    uniqueness_required: bool = False
    standard_or_code_system: str | None = None
    sensitivity: Literal["public", "quasi_identifier", "direct_identifier", "sensitive", "unknown"] = "unknown"
    protected_from_change: bool = False

    @field_validator("role")
    @classmethod
    def normalize_role(cls, value: str) -> str:
        value = str(value).strip().casefold()
        aliases = {
            "number": "numeric",
            "integer": "numeric",
            "float": "numeric",
            "category": "categorical",
            "datetime": "date",
            "string": "text",
        }
        return aliases.get(value, value)

    @field_validator("valid_range")
    @classmethod
    def validate_range(cls, value: tuple[float, float] | None) -> tuple[float, float] | None:
        if value is not None and value[0] > value[1]:
            raise ValueError("valid_range lower bound must not exceed upper bound")
        return value

    @field_validator("regex")
    @classmethod
    def validate_regex(cls, value: str | None) -> str | None:
        if value:
            re.compile(value)
        return value

    @field_validator("criticality_weight", "numeric_tolerance")
    @classmethod
    def non_negative_float(cls, value: float) -> float:
        if value < 0:
            raise ValueError("value must be non-negative")
        return value

    @field_validator("date_tolerance_days")
    @classmethod
    def non_negative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("date_tolerance_days must be non-negative")
        return value

    def effective_allowed_values(self) -> list[Any]:
        return list(self.approved_allowed_values or self.allowed_values)


class DecisionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    completeness_weight: float = 0.30
    accuracy_weight: float = 0.40
    reuse_weight: float = 0.30
    ready_threshold: float = 0.80
    conditional_threshold: float = 0.60
    allow_low_risk_apply_after_approval: bool = True
    require_human_review_for_supported: bool = True
    require_human_review_for_candidates: bool = True
    require_approved_contract_for_ready: bool = True
    max_confirmed_missing_rate_for_ready: float = 0.05
    max_confirmed_accuracy_violation_rate_for_ready: float = 0.02
    max_direct_identifier_columns_for_ready: int = 0

    @model_validator(mode="after")
    def validate_policy(self) -> "DecisionPolicy":
        weights = [self.completeness_weight, self.accuracy_weight, self.reuse_weight]
        if any(weight < 0 for weight in weights):
            raise ValueError("decision-policy weights must be non-negative")
        if abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("decision-policy weights must sum to 1.0")
        if not 0 <= self.conditional_threshold <= self.ready_threshold <= 1:
            raise ValueError("thresholds must satisfy 0 <= conditional <= ready <= 1")
        if not 0 <= self.max_confirmed_missing_rate_for_ready <= 1:
            raise ValueError("max_confirmed_missing_rate_for_ready must be between 0 and 1")
        if not 0 <= self.max_confirmed_accuracy_violation_rate_for_ready <= 1:
            raise ValueError("max_confirmed_accuracy_violation_rate_for_ready must be between 0 and 1")
        if self.max_direct_identifier_columns_for_ready < 0:
            raise ValueError("max_direct_identifier_columns_for_ready must be non-negative")
        return self


class DatasetContract(BaseModel):
    model_config = ConfigDict(extra="allow")

    contract_version: str = "2.0"
    dataset_id: str
    approval: Approval = Field(default_factory=Approval)
    dataset_metadata: DatasetMetadata = Field(default_factory=DatasetMetadata)
    dataset_metadata_required_fields: list[str] = Field(
        default_factory=lambda: [
            "title",
            "description",
            "provenance",
            "version",
            "license",
            "intended_reuse",
            "privacy_status",
        ]
    )
    metadata_required_fields: list[str] = Field(default_factory=lambda: ["role", "description"])
    missing_tokens: list[str] = Field(default_factory=lambda: ["NA", "N/A", "NULL", "?"])
    columns: dict[str, ColumnContract] = Field(default_factory=dict)
    primary_key: list[str] = Field(default_factory=list)
    row_alignment_policy: RowAlignmentPolicy = "position_with_warning"
    cross_field_rules: list[CrossFieldRule] = Field(default_factory=list)
    required_columns_for_reuse: list[str] = Field(default_factory=list)
    reuse_policy: ReusePolicy = Field(default_factory=ReusePolicy)
    decision_policy: DecisionPolicy = Field(default_factory=DecisionPolicy)

    @field_validator("dataset_id")
    @classmethod
    def non_empty_dataset_id(cls, value: str) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError("dataset_id must not be empty")
        return value



def _upgrade_legacy_contract(raw: dict[str, Any], dataset_id_fallback: str = "dataset") -> dict[str, Any]:
    upgraded = dict(raw)
    upgraded.setdefault("contract_version", "2.0")
    upgraded.setdefault("dataset_id", dataset_id_fallback)
    upgraded.setdefault("approval", {"status": "draft"})
    upgraded.setdefault("dataset_metadata", {})
    upgraded.setdefault("decision_policy", {})
    upgraded.setdefault("reuse_policy", {})
    upgraded.setdefault("columns", {})
    upgraded.setdefault("primary_key", [])
    upgraded.setdefault("row_alignment_policy", "position_with_warning")
    upgraded.setdefault("cross_field_rules", [])
    upgraded.setdefault("required_columns_for_reuse", [])
    return upgraded



def validate_contract(
    raw: dict[str, Any],
    *,
    dataset_columns: list[str] | None = None,
    require_approved: bool = False,
    dataset_id_fallback: str = "dataset",
) -> tuple[DatasetContract, list[str]]:
    contract = DatasetContract.model_validate(_upgrade_legacy_contract(raw, dataset_id_fallback))
    warnings: list[str] = []

    if require_approved and contract.approval.status != "approved":
        raise ValueError("The contract must have approval.status='approved' for strict product execution.")

    if dataset_columns is not None:
        observed = set(map(str, dataset_columns))
        declared = set(contract.columns)
        missing_in_contract = sorted(observed - declared)
        missing_in_data = sorted(declared - observed)
        if missing_in_contract:
            warnings.append(f"Columns missing from contract: {missing_in_contract}")
        if missing_in_data:
            warnings.append(f"Contract columns absent from dataset: {missing_in_data}")
        unknown_keys = sorted(set(contract.primary_key) - observed)
        if unknown_keys:
            warnings.append(f"Primary-key columns absent from dataset: {unknown_keys}")

    return contract, warnings



def load_contract(
    path: Path,
    *,
    dataset_columns: list[str] | None = None,
    require_approved: bool = False,
    dataset_id_fallback: str = "dataset",
) -> tuple[DatasetContract, list[str]]:
    try:
        raw = read_json(path)
        return validate_contract(
            raw,
            dataset_columns=dataset_columns,
            require_approved=require_approved,
            dataset_id_fallback=dataset_id_fallback,
        )
    except ValidationError as exc:
        messages = "; ".join(
            f"{'.'.join(map(str, error['loc']))}: {error['msg']}" for error in exc.errors()
        )
        raise ValueError(f"Invalid dataset contract: {messages}") from exc



def generate_draft_contract(df: pd.DataFrame, dataset_id: str, title: str | None = None) -> DatasetContract:
    columns: dict[str, ColumnContract] = {}
    for column in df.columns:
        role = infer_role(str(column), df[column])
        sensitivity: str = "unknown"
        name = str(column).casefold()
        if role in {"email", "phone"} or any(token in name for token in ("name", "address", "person_id", "patient_id")):
            sensitivity = "direct_identifier"
        elif any(token in name for token in ("age", "gender", "sex", "postcode", "zip")):
            sensitivity = "quasi_identifier"

        observed_sample = df[column].dropna().drop_duplicates().head(20).tolist()
        suggested: list[Any] = []
        if role == "categorical":
            suggested = df[column].dropna().value_counts().index.tolist()[:50]

        columns[str(column)] = ColumnContract(
            role=role,
            description="",
            observed_values_sample=observed_sample,
            suggested_allowed_values=suggested,
            approved_allowed_values=[],
            allowed_values=[],
            sensitivity=sensitivity,  # type: ignore[arg-type]
            protected_from_change=role in {"email", "phone"} or "id" in name,
        )

    return DatasetContract(
        dataset_id=dataset_id,
        approval=Approval(status="draft"),
        dataset_metadata=DatasetMetadata(
            title=title or dataset_id,
            description="",
            provenance="",
            version="1.0",
            license="",
            intended_reuse="",
            privacy_status="",
        ),
        columns=columns,
    )



def approve_contract(contract: DatasetContract, approved_by: str, notes: str | None = None) -> DatasetContract:
    payload = contract.model_dump(mode="json")
    payload["approval"] = {
        "status": "approved",
        "approved_by": approved_by.strip(),
        "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
    }
    return DatasetContract.model_validate(payload)



def save_contract(path: Path, contract: DatasetContract) -> None:
    write_json(path, contract.model_dump(mode="json"))
