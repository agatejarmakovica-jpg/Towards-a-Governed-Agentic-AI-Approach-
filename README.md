# C-DQC v5.0 Scientific Product

C-DQC is a governed tabular data-quality assessment product with exactly three independent scientific agents:

1. `CompletenessAgent`
2. `AccuracyAgent`
3. `ReuseAgent`

The Streamlit, CLI, contract approval, HITL review, curated-copy, ZIP export, and Colab connection workflow remain compatible with the prior product structure. The scientific core has been replaced with independent agent execution and evidence-aware metrics.

## Scientific execution model

Each agent independently performs analysis, issue detection, metric calculation, recommendation generation, uncertainty reporting, and execution tracing. `IntakeStage`, `CoordinatorStage`, and `VerifierStage` are governance stages, not scientific agents.

The system supports four evidence modes:

- `CONTRACT_ONLY`
- `CLEAN_REFERENCE`
- `CORRUPTION_MANIFEST`
- `REFERENCE_AND_MANIFEST`

Without a clean reference or corruption manifest, C-DQC does not claim ground-truth accuracy. Unavailable metrics are marked `NOT_COMPUTABLE`.

## Main capabilities

### CompletenessAgent

- explicit nulls, approved missing tokens, sentinel values, and candidate disguised missingness
- overall, per-column, per-row, and criticality-weighted completeness
- missing-pattern frequencies
- masked-value validation for numeric and categorical imputation candidates
- MAE, RMSE, normalized RMSE, categorical accuracy, macro-F1, coverage, and failed-imputation counts

### AccuracyAgent

- contract-role-driven parsing independent of Pandas dtype
- canonical mapping before parsing and validation
- numeric range, approved categories, dates, regex, uniqueness, and generic cross-field rules
- IQR, MAD, Isolation Forest, and LOF anomaly candidates
- clean-reference alignment by approved primary key or explicitly permitted row position
- exact and tolerance-adjusted cell accuracy, per-column accuracy, row exact-match rate, numeric MAE/RMSE, and categorical macro-F1
- manifest-based precision, recall, specificity, false-positive rate, F1, MCC, and per-issue-type metrics

### ReuseAgent

- documentation, schema, roles, units, code systems, standardization, privacy, reproducibility, and machine readability
- computed machine-readability indicators rather than a fixed constant
- contract-defined reuse weights and explicit blocking conditions

## Installation

```bash
python -m pip install -r requirements.txt
```

Or install the wheel:

```bash
python -m pip install dist/cdqc_product-5.0.0-py3-none-any.whl
```

## Streamlit UI

```bash
streamlit run app.py
```

The existing workflow is preserved:

1. upload the dataset;
2. upload or generate and approve a contract;
3. optionally upload a clean reference and corruption manifest;
4. run the three-agent assessment;
5. review grouped findings and cell-level proposals;
6. create a separate curated copy after named HITL approval;
7. rerun all three agents and compare before/after metrics.

## Google Colab

Use the included `COLAB_LAUNCH.py`. It uses the same Streamlit plus Cloudflare tunnel method as the prior working product flow.

```python
from google.colab import files
files.upload()  # upload COLAB_LAUNCH.py
%run COLAB_LAUNCH.py
```

Then upload `C-DQC_v5.0_scientific_product.zip` and open the printed URL.

## CLI

```bash
python -m cdqc_core contract-draft \
  --input data.csv \
  --output contract.json \
  --dataset-id dataset_1 \
  --approve-by "Reviewer"
```

```bash
python -m cdqc_core assess \
  --input data.csv \
  --contract contract.json \
  --reference clean_reference.csv \
  --truth-manifest corruption_manifest.csv \
  --output results/run_1 \
  --require-approved-contract
```

Optional semantic advice:

```bash
export OPENAI_API_KEY="..."
python -m cdqc_core assess \
  --input data.csv \
  --contract contract.json \
  --output results/run_1 \
  --openai-semantic-advisor
```

`SemanticAdvisor` receives schema, metadata, aggregate statistics, and issue summaries. It cannot change data, findings, metrics, or the verdict.

## Scientific outputs

The assessment bundle includes:

- `completeness_agent_result.json`
- `completeness_metrics.json`
- `completeness_recommendations.json`
- `accuracy_agent_result.json`
- `accuracy_metrics.json`
- `accuracy_recommendations.json`
- `reuse_agent_result.json`
- `reuse_metrics.json`
- `reuse_recommendations.json`
- `aggregate_metrics.json`
- `metric_availability.json`
- `agent_validation_metrics.json`
- `reference_exact_metrics.json`, when a clean reference is supplied
- `detection_benchmark_metrics.json`, when a corruption manifest is supplied
- `before_after_metrics.json`, after HITL-approved curation
- `agent_trace.json`
- `run_manifest.json`
- `report.html`

## Validation

```bash
python -m pytest -q
python -m cdqc_core benchmark --output validation_results/benchmark --runs 30 --rows 200
```

See `VALIDATION_REPORT.json` and `validation_results/benchmark/benchmark_report.json` for the executed validation evidence.

Health-themed data are representative tabular data-science benchmarks. C-DQC is not a clinical decision system.

## Functional LLM layer

This build uses an LLM before assessment as `ContractAdvisorLLM`. The LLM drafts semantic contract fields from the dataset profile and optional clean reference. A deterministic verifier blocks unsafe suggestions. The LLM does not compute C, A, R, DQI, F1, MCC, or verdicts.

