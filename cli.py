from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import run_repeated_benchmark
from .contract import approve_contract, generate_draft_contract, load_contract, save_contract
from .inject import write_injected_benchmark
from .io import read_table
from .pipeline import apply_human_review, run_assessment
from .semantic_advisor import SemanticAdvisorConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cdqc", description="C-DQC v5.0 Scientific Product")
    sub = parser.add_subparsers(dest="command", required=True)

    assess = sub.add_parser("assess", help="Run governed agentic assessment without mutating the source dataset")
    assess.add_argument("--input", required=True, type=Path)
    assess.add_argument("--output", required=True, type=Path)
    assess.add_argument("--contract", "--config", dest="contract", type=Path)
    assess.add_argument("--reference", type=Path)
    assess.add_argument("--truth-manifest", type=Path)
    assess.add_argument("--require-approved-contract", action="store_true")
    assess.add_argument("--openai-semantic-advisor", action="store_true")
    assess.add_argument("--openai-model", default="gpt-5-mini")
    assess.add_argument("--include-non-sensitive-samples", action="store_true")
    assess.add_argument("--no-zip", action="store_true")

    draft = sub.add_parser("contract-draft", help="Generate a dataset contract draft from a table")
    draft.add_argument("--input", required=True, type=Path)
    draft.add_argument("--output", required=True, type=Path)
    draft.add_argument("--dataset-id")
    draft.add_argument("--title")
    draft.add_argument("--approve-by", help="Immediately approve the generated contract under the named reviewer")
    draft.add_argument("--approval-notes")

    validate = sub.add_parser("contract-validate", help="Validate a dataset contract against a table")
    validate.add_argument("--input", required=True, type=Path)
    validate.add_argument("--contract", required=True, type=Path)
    validate.add_argument("--require-approved", action="store_true")

    review = sub.add_parser("apply-review", help="Apply named human approvals to a separate curated copy")
    review.add_argument("--input", required=True, type=Path)
    review.add_argument("--assessment-dir", required=True, type=Path)
    review.add_argument("--review", required=True, type=Path)
    review.add_argument("--output", required=True, type=Path)
    review.add_argument("--allow-medium-risk", action="store_true")
    review.add_argument("--reference", type=Path)
    review.add_argument("--truth-manifest", type=Path)
    review.add_argument("--no-zip", action="store_true")

    inject = sub.add_parser("inject", help="Create one controlled synthetic benchmark")
    inject.add_argument("--output", required=True, type=Path)
    inject.add_argument("--seed", type=int, default=1000)
    inject.add_argument("--rows", type=int, default=600)

    benchmark = sub.add_parser("benchmark", help="Run the repeated synthetic benchmark")
    benchmark.add_argument("--output", required=True, type=Path)
    benchmark.add_argument("--runs", type=int, default=30)
    benchmark.add_argument("--rows", type=int, default=600)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "assess":
        result = run_assessment(
            input_path=args.input,
            output_dir=args.output,
            config_path=args.contract,
            reference_path=args.reference,
            truth_manifest_path=args.truth_manifest,
            require_approved_contract=args.require_approved_contract,
            create_zip=not args.no_zip,
            semantic_config=SemanticAdvisorConfig(
                enabled=args.openai_semantic_advisor,
                model=args.openai_model,
                include_non_sensitive_sample_values=args.include_non_sensitive_samples,
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    elif args.command == "contract-draft":
        df = read_table(args.input)
        contract = generate_draft_contract(
            df,
            dataset_id=args.dataset_id or args.input.stem,
            title=args.title or args.input.stem,
        )
        if args.approve_by:
            contract = approve_contract(contract, args.approve_by, args.approval_notes)
        save_contract(args.output, contract)
        print(json.dumps({"contract": str(args.output), "status": contract.approval.status}, indent=2))
    elif args.command == "contract-validate":
        df = read_table(args.input)
        contract, warnings = load_contract(
            args.contract,
            dataset_columns=[str(column) for column in df.columns],
            require_approved=args.require_approved,
            dataset_id_fallback=args.input.stem,
        )
        print(
            json.dumps(
                {
                    "valid": True,
                    "dataset_id": contract.dataset_id,
                    "approval_status": contract.approval.status,
                    "warnings": warnings,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    elif args.command == "apply-review":
        result = apply_human_review(
            input_path=args.input,
            assessment_dir=args.assessment_dir,
            review_path=args.review,
            output_dir=args.output,
            allow_medium_risk=args.allow_medium_risk,
            reference_path=args.reference,
            truth_manifest_path=args.truth_manifest,
            create_zip=not args.no_zip,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    elif args.command == "inject":
        print(json.dumps(write_injected_benchmark(args.output, seed=args.seed, rows=args.rows), indent=2))
    elif args.command == "benchmark":
        report = run_repeated_benchmark(args.output, runs=args.runs, rows=args.rows)
        print(json.dumps(report["benchmark_design"], indent=2))


if __name__ == "__main__":
    main()
