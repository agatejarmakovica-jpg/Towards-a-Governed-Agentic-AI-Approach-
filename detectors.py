from __future__ import annotations

from typing import Any

import pandas as pd

from .agent_utils import infer_role


def detect_issues(df: pd.DataFrame, config: dict[str, Any] | None = None):
    """Backward-compatible deterministic detector entry point.

    The scientific product no longer uses one shared detector as the agentic core.
    This compatibility function validates the supplied contract-like mapping and
    independently executes the three specialist agents, then merges their findings.
    """
    from .contract import DatasetContract
    from .completeness_agent import CompletenessAgent
    from .accuracy_agent import AccuracyAgent
    from .reuse_agent import ReuseAgent
    from .agent_utils import deduplicate_issues

    raw = dict(config or {})
    raw.setdefault("dataset_id", "dataset")
    raw.setdefault("columns", {})
    contract = DatasetContract.model_validate(raw)
    completeness = CompletenessAgent().run(df, contract, evidence_mode="CONTRACT_ONLY")
    accuracy = AccuracyAgent().run(df, contract, evidence_mode="CONTRACT_ONLY")
    reuse = ReuseAgent().run(df, contract, evidence_mode="CONTRACT_ONLY")
    # The legacy detector API historically returned data/schema findings but not
    # dataset-level readiness and privacy governance findings. Preserve that
    # contract for benchmark comparability while the product pipeline retains all
    # ReuseAgent findings.
    legacy_reuse = [issue for issue in reuse.issues if issue.issue_type == "schema_metadata_gap"]
    return deduplicate_issues([*completeness.issues, *accuracy.issues, *legacy_reuse])
