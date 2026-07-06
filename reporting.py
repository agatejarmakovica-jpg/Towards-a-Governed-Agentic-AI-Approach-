from __future__ import annotations

from collections import Counter
from html import escape
from pathlib import Path
from typing import Any

from .models import AgentTrace, Issue, Recommendation, RemediationProposal


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{escape(str(header))}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def write_html_report(
    path: Path,
    *,
    dataset_id: str,
    metrics: dict[str, Any],
    issues: list[Issue],
    recommendations: list[Recommendation],
    proposals: list[RemediationProposal],
    traces: list[AgentTrace],
    contract_status: str,
    contract_warnings: list[str],
    manifest: dict[str, Any],
) -> None:
    issue_counts = Counter(issue.issue_type for issue in issues)
    status_counts = Counter(issue.evidence_status for issue in issues)
    issue_rows = [[kind, count] for kind, count in sorted(issue_counts.items(), key=lambda item: (-item[1], item[0]))]
    trace_rows = [
        [trace.sequence, trace.agent, trace.status, trace.output_summary.get("issue_count", ""), trace.message]
        for trace in traces
    ]
    recommendation_rows = [
        [r.issue_type, r.column or "dataset", r.affected_count, r.rank, r.method, r.expected_risk]
        for r in recommendations[:100]
    ]
    warning_html = "".join(f"<li>{escape(warning)}</li>" for warning in contract_warnings) or "<li>None</li>"

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>C-DQC report - {escape(dataset_id)}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2937; line-height: 1.45; }}
h1, h2 {{ color: #111827; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
.card {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 14px; background: #f9fafb; }}
.value {{ font-size: 1.6rem; font-weight: 700; }}
table {{ width: 100%; border-collapse: collapse; margin: 12px 0 24px; font-size: 0.92rem; }}
th, td {{ border: 1px solid #d1d5db; padding: 7px; text-align: left; vertical-align: top; }}
th {{ background: #eef2f7; }}
.notice {{ border-left: 4px solid #6b7280; padding: 10px 14px; background: #f3f4f6; }}
code {{ background: #f3f4f6; padding: 2px 4px; }}
</style>
</head>
<body>
<h1>C-DQC governed data-quality report</h1>
<p><strong>Dataset:</strong> {escape(dataset_id)}<br>
<strong>Run ID:</strong> {escape(str(manifest.get('run_id', '')))}<br>
<strong>Contract status:</strong> {escape(contract_status)}<br>
<strong>Created:</strong> {escape(str(manifest.get('created_at_utc', '')))}</p>

<div class="notice">This report distinguishes confirmed findings, supported findings, and candidates. No candidate is treated as a confirmed error. The source snapshot is immutable. Curated output is created only from explicit human approvals.</div>

<h2>Decision summary</h2>
<div class="grid">
<div class="card"><div>Verdict</div><div class="value">{escape(str(metrics.get('verdict', 'n/a')))}</div></div>
<div class="card"><div>Quality support index</div><div class="value">{_pct(metrics.get('quality_support_index'))}</div></div>
<div class="card"><div>Completeness support</div><div class="value">{_pct(metrics.get('completeness_support'))}</div></div>
<div class="card"><div>Accuracy support</div><div class="value">{_pct(metrics.get('accuracy_support'))}</div></div>
<div class="card"><div>Reuse readiness</div><div class="value">{_pct(metrics.get('reuse_readiness_support'))}</div></div>
<div class="card"><div>Human review items</div><div class="value">{escape(str(metrics.get('human_review_required', 0)))}</div></div>
</div>

<h2>Evidence status</h2>
{_table(['Status', 'Count'], [[key, value] for key, value in sorted(status_counts.items())])}

<h2>Issues by type</h2>
{_table(['Issue type', 'Count'], issue_rows)}

<h2>Agent execution trace</h2>
{_table(['Seq.', 'Agent', 'Status', 'Issues', 'Message'], trace_rows)}

<h2>Top recommendations</h2>
{_table(['Issue type', 'Column', 'Affected', 'Rank', 'Method', 'Risk'], recommendation_rows)}
<p>Full recommendations are available in <code>recommendations.csv</code>.</p>

<h2>Human review</h2>
<p>{len(proposals)} remediation or governance proposals were created. Executable low-risk proposals may be applied after named reviewer approval. Medium-risk actions additionally require an explicit execution override.</p>

<h2>Contract warnings</h2>
<ul>{warning_html}</ul>

<h2>Scientific claim boundary</h2>
<p>The quality-support scores are governed support metrics. They are not ground-truth accuracy. Precision, recall, and F1 are produced only when a corruption manifest is supplied. Exact cell accuracy is produced only when a clean reference dataset is supplied.</p>
</body>
</html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
