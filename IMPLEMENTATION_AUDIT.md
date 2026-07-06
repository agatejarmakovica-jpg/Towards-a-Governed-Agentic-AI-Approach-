# C-DQC v5.0 implementation audit

## Defects corrected from v4.1

- the shared detector no longer acts as the scientific core;
- CompletenessAgent, AccuracyAgent, and ReuseAgent execute independently;
- clean reference is loaded before agent execution and used inside AccuracyAgent;
- contract role controls parsing rather than Pandas dtype;
- approved categories take priority over generic missing-value heuristics;
- dirty observed categories are not automatically approved in a generated contract;
- ground-truth accuracy is not reported in contract-only mode;
- machine readability is computed from observable indicators;
- manifest metrics exclude reference-comparison findings;
- HITL creates a separate curated copy and reruns all three agents under the same evidence mode;
- OpenAI is optional and advisory only.

## Preserved product components

- Streamlit upload and review workflow;
- CLI entry points;
- Colab Streamlit connection method;
- approved contract workflow;
- result ZIP and HTML report;
- source-file immutability and hashes;
- automated test execution.

## Verified release state

See `VALIDATION_REPORT.json`, `PRODUCT_ACCEPTANCE.md`, and `validation_results/`.
