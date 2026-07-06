# C-DQC v5.0 architecture

## Stable product layer

The following external components retain the prior product workflow:

- Streamlit upload and assessment interface
- contract editor and named approval
- CLI commands
- Colab Streamlit launcher
- immutable source snapshot and SHA-256 verification
- HITL review and separate curated output
- downloadable report and result bundle

## Scientific core

```text
INPUT + APPROVED CONTRACT + OPTIONAL EVIDENCE
                       |
                  IntakeStage
                       |
       +---------------+---------------+
       |               |               |
CompletenessAgent AccuracyAgent   ReuseAgent
       |               |               |
       +---------------+---------------+
                       |
                CoordinatorStage
                       |
                 VerifierStage
                       |
       METRICS + ISSUES + RECOMMENDATIONS
                       |
                      HITL
                       |
             SEPARATE CURATED COPY
                       |
              THREE-AGENT REASSESSMENT
```

Only the three specialist components are scientific agents. Intake, coordination, and verification are deterministic governance stages.

## Agent contract

Every specialist returns an `AgentResult` containing:

- evidence mode
- typed findings
- metrics
- metric availability
- recommendations
- remediation information
- uncertainty
- execution details

Agent implementations do not depend on dataset-specific column names or values. Validation behavior is determined by the approved data contract.

## Evidence separation

- contract findings establish structural or rule-based support;
- clean-reference findings establish cell-level comparison evidence;
- corruption manifests establish detection performance;
- statistical anomalies remain candidates unless confirmed by stronger evidence or HITL.

## Optional SemanticAdvisor

OpenAI integration is an advisory service shared by the three agents. It is not a fourth scientific agent. It can propose contract refinements, testable cross-column rules, and review priorities. It cannot modify deterministic results.
