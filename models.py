from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

IssueScope = Literal["cell", "row", "column", "dataset"]
EvidenceStatus = Literal["confirmed", "supported", "candidate"]
ReviewDecision = Literal["pending", "approve", "reject", "defer"]
EvidenceMode = Literal[
    "CONTRACT_ONLY",
    "CLEAN_REFERENCE",
    "CORRUPTION_MANIFEST",
    "REFERENCE_AND_MANIFEST",
]
MetricState = Literal["COMPUTED", "NOT_COMPUTABLE", "NOT_APPLICABLE"]


@dataclass(frozen=True)
class Issue:
    issue_id: str
    issue_type: str
    scope: IssueScope
    column: str | None
    row_index: int | None
    value: Any
    evidence_status: EvidenceStatus
    confidence: float
    severity: float
    evidence: dict[str, Any] = field(default_factory=dict)
    automatic_action_allowed: bool = False

    def unit_key(self) -> str:
        return f"{self.scope}|{self.row_index}|{self.column}|{self.issue_type}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Recommendation:
    issue_group_id: str
    issue_type: str
    column: str | None
    affected_count: int
    issue_ids: list[str]
    method: str
    rank: int
    rationale: str
    prerequisites: list[str]
    expected_risk: str
    application_mode: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentRecommendation:
    recommendation_id: str
    agent_name: str
    issue_type: str
    column: str | None
    method: str
    rationale: str
    prerequisites: list[str] = field(default_factory=list)
    validation_metrics: dict[str, Any] = field(default_factory=dict)
    risk: Literal["low", "medium", "high"] = "low"
    priority: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetricAvailability:
    metric: str
    state: MetricState
    reason: str = ""
    evidence_mode: EvidenceMode = "CONTRACT_ONLY"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentResult:
    agent_name: str
    evidence_mode: EvidenceMode
    issues: list[Issue]
    metrics: dict[str, Any]
    recommendations: list[AgentRecommendation] = field(default_factory=list)
    remediation_proposals: list[dict[str, Any]] = field(default_factory=list)
    uncertainty: dict[str, Any] = field(default_factory=dict)
    execution_details: dict[str, Any] = field(default_factory=dict)
    metric_availability: list[MetricAvailability] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "evidence_mode": self.evidence_mode,
            "issues": [issue.to_dict() for issue in self.issues],
            "metrics": self.metrics,
            "recommendations": [item.to_dict() for item in self.recommendations],
            "remediation_proposals": self.remediation_proposals,
            "uncertainty": self.uncertainty,
            "execution_details": self.execution_details,
            "metric_availability": [item.to_dict() for item in self.metric_availability],
        }


@dataclass(frozen=True)
class AgentTrace:
    sequence: int
    agent: str
    objective: str
    status: Literal["completed", "blocked", "failed"]
    started_at_utc: str
    finished_at_utc: str
    input_summary: dict[str, Any]
    output_summary: dict[str, Any]
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RemediationProposal:
    proposal_id: str
    issue_id: str
    issue_type: str
    scope: IssueScope
    row_index: int | None
    column: str | None
    current_value: Any
    proposed_value: Any
    action: str
    risk: Literal["low", "medium", "high"]
    reversible: bool
    executable: bool
    rationale: str
    decision: ReviewDecision = "pending"
    reviewer: str = ""
    reviewer_comment: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
