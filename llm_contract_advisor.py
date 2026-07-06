from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
import os
import re
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from .agent_utils import infer_role, normalize_for_comparison
from .contract import ColumnContract, CrossFieldRule, DatasetContract, DatasetMetadata, generate_draft_contract

ALLOWED_ROLES = {"numeric", "categorical", "date", "text", "identifier", "email", "phone", "binary", "free_text"}
ALLOWED_SENSITIVITY = {"public", "quasi_identifier", "direct_identifier", "sensitive", "unknown"}
DIRECT_IDENTIFIER_TOKENS = ("name", "email", "mail", "phone", "mobile", "telephone", "address", "person_id", "patient_id")
DATE_FORMAT_CANDIDATES = ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%d.%m.%Y")


@dataclass(frozen=True)
class LLMContractAdvisorConfig:
    enabled: bool = False
    api_key: str = ""
    model: str = "gpt-5-mini"
    include_sample_values: bool = True
    max_values_per_column: int = 25
    max_pair_examples_per_column: int = 40
    fail_on_llm_error: bool = True

    def resolved_key(self) -> str:
        return self.api_key.strip() or os.environ.get("OPENAI_API_KEY", "").strip()

    def public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("api_key", None)
        payload["api_key_source"] = "runtime_or_OPENAI_API_KEY"
        return payload


class LLMColumnSuggestion(BaseModel):
    column: str
    role: str | None = None
    description: str | None = None
    measurement_unit: str | None = None
    sensitivity: str | None = None
    approved_allowed_values: list[str] = Field(default_factory=list)
    canonical_mapping: dict[str, str] = Field(default_factory=dict)
    target_format: str | None = None
    source_formats: list[str] = Field(default_factory=list)
    regex: str | None = None
    standard_or_code_system: str | None = None
    required: bool | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str | None = None
    warnings: list[str] = Field(default_factory=list)


class LLMDatasetSuggestion(BaseModel):
    description: str | None = None
    intended_reuse: str | None = None
    privacy_status: str | None = None
    license: str | None = None
    provenance: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    questions_for_user: list[str] = Field(default_factory=list)


class LLMCrossFieldRuleSuggestion(BaseModel):
    rule_id: str
    description: str
    left_column: str
    operator: Literal["<", "<=", "==", "!=", ">=", ">"]
    right_column: str | None = None
    right_value: str | float | int | None = None
    severity: Literal["confirmed", "supported", "candidate"] = "candidate"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class LLMContractAdvisorOutput(BaseModel):
    summary: str
    dataset: LLMDatasetSuggestion = Field(default_factory=LLMDatasetSuggestion)
    columns: list[LLMColumnSuggestion] = Field(default_factory=list)
    cross_field_rules: list[LLMCrossFieldRuleSuggestion] = Field(default_factory=list)
    unsafe_mapping_warnings: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Strict wire models (OpenAI structured outputs).
#
# OpenAI strict JSON schemas reject free-form dict fields (additionalProperties
# must be false), so the wire representation uses explicit entry lists instead
# of dict[str, str]. Structural validity is therefore enforced by constrained
# decoding at generation time; the internal models above stay unchanged and the
# wire output is converted deterministically via _wire_to_internal().
# ---------------------------------------------------------------------------


class WireMappingEntry(BaseModel):
    raw_value: str = Field(description="Exact raw value observed in the dirty dataset.")
    canonical_value: str = Field(description="Exact canonical replacement value.")


class WireUnresolvedQuestion(BaseModel):
    column: str | None = Field(description="Column name, or null for dataset-level questions.")
    question: str


class WireDatasetSuggestion(BaseModel):
    description: str | None
    intended_reuse: str | None
    privacy_status: str | None
    license: str | None
    provenance: str | None
    confidence: float
    questions_for_user: list[str]


class WireColumnSuggestion(BaseModel):
    column: str = Field(description="Exact uploaded column name.")
    role: str | None
    description: str | None
    measurement_unit: str | None
    sensitivity: str | None
    approved_allowed_values: list[str]
    canonical_mapping: list[WireMappingEntry] = Field(
        description="Only concrete raw->canonical value pairs. Never booleans, operation names, or nested objects."
    )
    target_format: str | None
    source_formats: list[str]
    regex: str | None
    standard_or_code_system: str | None
    required: bool | None
    confidence: float
    rationale: str | None
    warnings: list[str]


class WireCrossFieldRule(BaseModel):
    rule_id: str
    description: str
    left_column: str
    operator: Literal["<", "<=", "==", "!=", ">=", ">"]
    right_column: str | None
    right_value: str | None
    severity: Literal["confirmed", "supported", "candidate"]
    confidence: float


class WireAdvisorOutput(BaseModel):
    summary: str
    dataset: WireDatasetSuggestion
    columns: list[WireColumnSuggestion]
    cross_field_rules: list[WireCrossFieldRule]
    unsafe_mapping_warnings: list[str]
    unresolved_questions: list[WireUnresolvedQuestion]


def _clamp_confidence(value: Any) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _wire_to_internal(wire: WireAdvisorOutput) -> LLMContractAdvisorOutput:
    """Deterministic conversion from the strict wire shape to internal models."""
    columns: list[LLMColumnSuggestion] = []
    for col in wire.columns:
        mapping: dict[str, str] = {}
        for entry in col.canonical_mapping:
            raw = entry.raw_value.strip()
            canonical = entry.canonical_value.strip()
            if raw and canonical:
                mapping[raw] = canonical
        columns.append(LLMColumnSuggestion(
            column=col.column,
            role=col.role,
            description=col.description,
            measurement_unit=col.measurement_unit,
            sensitivity=col.sensitivity,
            approved_allowed_values=[v for v in col.approved_allowed_values if str(v).strip()],
            canonical_mapping=mapping,
            target_format=col.target_format,
            source_formats=[fmt for fmt in col.source_formats if str(fmt).strip()],
            regex=col.regex,
            standard_or_code_system=col.standard_or_code_system,
            required=col.required,
            confidence=_clamp_confidence(col.confidence),
            rationale=col.rationale,
            warnings=list(col.warnings),
        ))
    rules = [
        LLMCrossFieldRuleSuggestion(
            rule_id=rule.rule_id,
            description=rule.description,
            left_column=rule.left_column,
            operator=rule.operator,
            right_column=rule.right_column,
            right_value=rule.right_value,
            severity=rule.severity,
            confidence=_clamp_confidence(rule.confidence),
        )
        for rule in wire.cross_field_rules
    ]
    questions = [
        (f"Column '{item.column}': {item.question}" if item.column else item.question)
        for item in wire.unresolved_questions
        if item.question.strip()
    ]
    return LLMContractAdvisorOutput(
        summary=wire.summary,
        dataset=LLMDatasetSuggestion(
            description=wire.dataset.description,
            intended_reuse=wire.dataset.intended_reuse,
            privacy_status=wire.dataset.privacy_status,
            license=wire.dataset.license,
            provenance=wire.dataset.provenance,
            confidence=_clamp_confidence(wire.dataset.confidence),
            questions_for_user=list(wire.dataset.questions_for_user),
        ),
        columns=columns,
        cross_field_rules=rules,
        unsafe_mapping_warnings=list(wire.unsafe_mapping_warnings),
        unresolved_questions=questions,
    )


def _json_safe(value: Any) -> Any:
    try:
        if value is None or pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _missing_mask(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype("string").str.strip().isin(["", "NA", "N/A", "NULL", "?", "nan", "None"])


def _looks_direct_identifier(column: str, role: str) -> bool:
    name = column.casefold()
    return role in {"email", "phone"} or any(token in name for token in DIRECT_IDENTIFIER_TOKENS)


def _safe_unique_values(series: pd.Series, limit: int = 50) -> list[Any]:
    observed = series[~_missing_mask(series)].dropna().drop_duplicates()
    if len(observed) > limit:
        return []
    return [_json_safe(v) for v in observed.tolist()]


def _format_match_rate(series: pd.Series, fmt: str) -> float:
    values = series[~_missing_mask(series)].astype("string").str.strip()
    if len(values) == 0:
        return 0.0
    ok = 0
    for value in values.head(5000):
        try:
            parsed = pd.to_datetime(value, errors="raise", format=fmt)
            if parsed.strftime(fmt) == value:
                ok += 1
        except Exception:
            pass
    return ok / min(len(values), 5000)


def _best_date_format(series: pd.Series) -> tuple[str | None, list[str]]:
    rates = [(fmt, _format_match_rate(series, fmt)) for fmt in DATE_FORMAT_CANDIDATES]
    good = [fmt for fmt, rate in rates if rate >= 0.60]
    if not good:
        return None, []
    best = max(rates, key=lambda item: item[1])[0]
    return best, good


def build_contract_advisor_payload(
    df: pd.DataFrame,
    *,
    dataset_id: str,
    title: str,
    reference_df: pd.DataFrame | None,
    config: LLMContractAdvisorConfig,
) -> dict[str, Any]:
    columns: list[dict[str, Any]] = []
    for column in df.columns:
        name = str(column)
        series = df[column]
        reference_series = reference_df[name] if reference_df is not None and name in reference_df.columns else None
        dirty_role = infer_role(name, series)
        reference_role = infer_role(name, reference_series) if reference_series is not None else None
        role = reference_role or dirty_role
        sensitive = _looks_direct_identifier(name, role)
        col_payload: dict[str, Any] = {
            "name": name,
            "dirty_inferred_role": dirty_role,
            "reference_inferred_role": reference_role,
            "selected_technical_role": role,
            "rows": int(len(series)),
            "dirty_missing_count": int(_missing_mask(series).sum()),
            "dirty_unique_count": int(series[~_missing_mask(series)].nunique(dropna=True)),
            "reference_unique_count": int(reference_series[~_missing_mask(reference_series)].nunique(dropna=True)) if reference_series is not None else None,
            "direct_identifier_like": sensitive,
        }
        if config.include_sample_values and not sensitive:
            col_payload["dirty_observed_values"] = _safe_unique_values(series, config.max_values_per_column)
            if reference_series is not None:
                col_payload["reference_observed_values"] = _safe_unique_values(reference_series, config.max_values_per_column)
        if reference_series is not None and not sensitive:
            pairs: Counter[tuple[str, str]] = Counter()
            count = min(len(series), len(reference_series))
            for i in range(count):
                left = series.iloc[i]
                right = reference_series.iloc[i]
                if pd.isna(left) or pd.isna(right):
                    continue
                left_text = str(left).strip()
                right_text = str(right).strip()
                if left_text and right_text and left_text != right_text:
                    pairs[(left_text, right_text)] += 1
            col_payload["dirty_to_reference_examples"] = [
                {"dirty": left, "reference": right, "count": int(count)}
                for (left, right), count in pairs.most_common(config.max_pair_examples_per_column)
            ]
        columns.append(col_payload)
    return {
        "task": "Propose a data contract for C-DQC. Do not compute quality metrics. Do not approve the contract.",
        "strict_rules": [
            "Return only proposed contract fields.",
            "Do not invent license or provenance. Return null when unknown.",
            "Do not map one valid category to another valid category.",
            "For blood type, A+ and A-, B+ and B-, AB+ and AB-, O+ and O- are distinct values and must not be canonicalized to each other.",
            "Use the clean reference only to propose allowed vocabularies and safe normalization candidates.",
            "If uncertain, return null and add a question for the user.",
        ],
        "dataset": {"dataset_id": dataset_id, "title": title, "rows": int(len(df)), "columns": int(len(df.columns))},
        "reference_supplied": reference_df is not None,
        "columns": columns,
    }


def _extract_response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if isinstance(value, str):
                chunks.append(value)
    return "\n".join(chunks).strip()


def _parse_json_object_from_text(text: str) -> dict[str, Any]:
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean)
        clean = re.sub(r"\s*```$", "", clean)
    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, flags=re.S)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("LLM response was valid JSON but not a JSON object.")
    return data


_MAPPING_RAW_KEYS = ("raw", "raw_value", "source", "source_value", "from", "dirty")
_MAPPING_CANONICAL_KEYS = ("canonical", "canonical_value", "target", "target_value", "to", "clean")
_SEVERITY_ALIASES = {"error": "candidate", "warning": "candidate", "warn": "candidate", "info": "candidate", "critical": "candidate"}


def _coerce_canonical_mapping(mapping: Any, dropped: list[dict[str, Any]]) -> dict[str, str]:
    """Coerce arbitrary LLM output into a str->str value mapping.

    Handles the observed drift modes: booleans used as operation flags
    ({"trim_whitespace": true}), nested example dicts ({"examples": {...}}),
    and list-of-records shapes. Anything that is not a concrete raw->canonical
    string pair is dropped and recorded for the audit report — never silently
    reinterpreted as a data transformation.
    """
    converted: dict[str, str] = {}
    if isinstance(mapping, dict):
        for key, value in mapping.items():
            if isinstance(value, str):
                if str(key).strip() and value.strip():
                    converted[str(key).strip()] = value.strip()
            elif isinstance(value, dict):
                # Nested example dicts: hoist only concrete str->str pairs.
                hoisted = False
                for raw, canonical in value.items():
                    if isinstance(raw, str) and isinstance(canonical, str) and raw.strip() and canonical.strip():
                        converted[raw.strip()] = canonical.strip()
                        hoisted = True
                if not hoisted:
                    dropped.append({"key": str(key), "value": str(value), "reason": "Nested object is not a value mapping."})
            else:
                # Booleans / numbers are operation flags, not value mappings.
                dropped.append({"key": str(key), "value": str(value), "reason": "Non-string mapping value; operation flags are not part of the contract action space."})
    elif isinstance(mapping, list):
        for item in mapping:
            if not isinstance(item, dict):
                dropped.append({"value": str(item), "reason": "Mapping list entry is not an object."})
                continue
            raw = next((item[k] for k in _MAPPING_RAW_KEYS if isinstance(item.get(k), str)), None)
            canonical = next((item[k] for k in _MAPPING_CANONICAL_KEYS if isinstance(item.get(k), str)), None)
            if raw and canonical and raw.strip() and canonical.strip():
                converted[raw.strip()] = canonical.strip()
            else:
                dropped.append({"value": str(item), "reason": "Mapping entry lacks raw/canonical string pair."})
    elif mapping not in (None, {}, []):
        dropped.append({"value": str(mapping), "reason": "canonical_mapping has unsupported type."})
    return converted


def _coerce_unresolved_question(item: Any) -> str | None:
    if isinstance(item, str):
        return item.strip() or None
    if isinstance(item, dict):
        question = item.get("question") or item.get("text") or item.get("issue")
        column = item.get("column") or item.get("field") or item.get("scope")
        if isinstance(question, str) and question.strip():
            if isinstance(column, str) and column.strip() and column.strip().casefold() != "general":
                return f"Column '{column.strip()}': {question.strip()}"
            return question.strip()
        return str(item)
    return None


def _coerce_cross_field_rule(rule: Any, index: int, dropped: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not isinstance(rule, dict):
        dropped.append({"value": str(rule), "reason": "Cross-field rule is not an object."})
        return None
    rule = dict(rule)
    if "rule_id" not in rule:
        alias = rule.get("rule") or rule.get("id") or rule.get("name")
        rule["rule_id"] = str(alias) if alias else f"llm_rule_{index}"
    if "description" not in rule:
        rule["description"] = str(rule.get("rule") or rule.get("rule_id"))
    severity = str(rule.get("severity", "candidate")).strip().casefold()
    rule["severity"] = _SEVERITY_ALIASES.get(severity, severity if severity in {"confirmed", "supported", "candidate"} else "candidate")
    # LLM-proposed rules are never auto-confirmed; the human approves the contract.
    if rule["severity"] == "confirmed":
        rule["severity"] = "supported"
    if not isinstance(rule.get("left_column"), str) or not rule.get("left_column") or rule.get("operator") not in {"<", "<=", "==", "!=", ">=", ">"}:
        dropped.append({"rule_id": rule.get("rule_id"), "value": str(rule), "reason": "Rule lacks a machine-checkable left_column/operator form."})
        return None
    rule.setdefault("right_column", None)
    rule.setdefault("right_value", None)
    rule.setdefault("confidence", 0.0)
    rule["confidence"] = _clamp_confidence(rule.get("confidence"))
    allowed_keys = {"rule_id", "description", "left_column", "operator", "right_column", "right_value", "severity", "confidence"}
    return {key: rule[key] for key in allowed_keys if key in rule}


def _normalize_llm_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Fallback-mode normalizer for non-strict JSON output.

    Used only when constrained decoding is unavailable. Makes common LLM
    shape drift compatible with the internal models; everything dropped here
    is recorded under _normalizer_dropped for the audit report.
    """
    data = dict(data)
    dropped: list[dict[str, Any]] = []
    data.setdefault("summary", "LLM contract advisor suggestions.")
    data.setdefault("dataset", {})
    data.setdefault("columns", [])
    data.setdefault("cross_field_rules", [])
    data.setdefault("unsafe_mapping_warnings", [])
    data.setdefault("unresolved_questions", [])
    if isinstance(data.get("dataset"), dict):
        ds = data["dataset"]
        ds.setdefault("questions_for_user", [])
        ds["confidence"] = _clamp_confidence(ds.get("confidence", 0.0))
    normalized_columns = []
    for column in data.get("columns") or []:
        if not isinstance(column, dict):
            continue
        column = dict(column)
        mapping_dropped: list[dict[str, Any]] = []
        column["canonical_mapping"] = _coerce_canonical_mapping(column.get("canonical_mapping"), mapping_dropped)
        for item in mapping_dropped:
            dropped.append({"column": column.get("column"), "field": "canonical_mapping", **item})
        column.setdefault("approved_allowed_values", [])
        column.setdefault("source_formats", [])
        column.setdefault("warnings", [])
        column["confidence"] = _clamp_confidence(column.get("confidence", 0.0))
        normalized_columns.append(column)
    data["columns"] = normalized_columns
    normalized_rules = []
    for index, rule in enumerate(data.get("cross_field_rules") or []):
        coerced = _coerce_cross_field_rule(rule, index, dropped)
        if coerced is not None:
            normalized_rules.append(coerced)
    data["cross_field_rules"] = normalized_rules
    data["unresolved_questions"] = [
        question for question in (_coerce_unresolved_question(item) for item in data.get("unresolved_questions") or [])
        if question
    ]
    data["unsafe_mapping_warnings"] = [str(item) for item in data.get("unsafe_mapping_warnings") or []]
    data["_normalizer_dropped"] = dropped
    return data


def _call_openai_contract_advisor(payload: dict[str, Any], config: LLMContractAdvisorConfig) -> tuple[LLMContractAdvisorOutput | None, dict[str, Any]]:
    key = config.resolved_key()
    if not config.enabled:
        return None, {"enabled": False, "status": "disabled"}
    if not key:
        raise ValueError("OpenAI API key is required for LLM-assisted contract drafting.")
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("The openai package is required for LLM-assisted contract drafting.") from exc

    system = (
        "You are ContractAdvisorLLM for C-DQC. Draft contract fields from the technical profile and optional clean reference. "
        "Do not compute metrics, do not approve contracts, do not change data, and do not invent license or provenance. "
        "Do not map one already valid category to another already valid category. Every suggestion must be conservative and verifiable. "
        "canonical_mapping contains only concrete raw->canonical value pairs observed in the data. "
        "It is never a place for transformation instructions, operation flags, or boolean options: "
        "entries like trim_whitespace, normalize_spaces, or case_fold are forbidden anywhere in canonical_mapping. "
        "If a systematic normalization would be needed, describe it in the column rationale and add an unresolved question instead. "
        "Cross-field rules must be machine-checkable: rule_id, description, left_column, operator (one of < <= == != >= >), "
        "right_column or right_value, and severity of supported or candidate. "
        "Unresolved questions are objects with a column field (null for dataset-level) and a question field."
    )
    user_content = json.dumps({"contract_advisor_input": payload}, ensure_ascii=False, default=str)
    model = config.model.strip()
    client = OpenAI(api_key=key)
    fallback_errors: list[dict[str, str]] = []

    def _audit(response: Any, api_mode: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        usage = getattr(response, "usage", None)
        audit: dict[str, Any] = {
            "enabled": True,
            "status": "completed",
            "model": model,
            "response_id": getattr(response, "id", None),
            "usage": usage.model_dump(mode="json") if usage is not None and hasattr(usage, "model_dump") else None,
            "api_mode": api_mode,
            "structured_output_enforced": api_mode == "responses_parse_strict",
            "fallback_errors": fallback_errors,
            "api_key_persisted": False,
            "raw_dataset_sent": False,
            "direct_identifier_values_sent": False,
            "reference_supplied": bool(payload.get("reference_supplied")),
        }
        if extra:
            audit.update(extra)
        return audit

    # Primary path: constrained decoding against the strict wire schema.
    # Structural validity is guaranteed at generation time, so the entire
    # class of shape-drift validation errors is impossible by construction.
    try:
        response = client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            text_format=WireAdvisorOutput,
            store=False,
        )
        wire = getattr(response, "output_parsed", None)
        if wire is None:
            raise RuntimeError("responses.parse returned no parsed output (possible refusal or incomplete response).")
        return _wire_to_internal(wire), _audit(response, "responses_parse_strict")
    except Exception as exc:
        fallback_errors.append({"stage": "responses_parse_strict", "error": f"{type(exc).__name__}: {exc}"})

    # Fallback 1: JSON-object mode + normalizer (SDK/model combinations
    # where strict structured outputs are unavailable).
    json_system = system + " Return only one valid JSON object with keys: summary, dataset, columns, cross_field_rules, unsafe_mapping_warnings, unresolved_questions."
    try:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": json_system},
                {"role": "user", "content": user_content},
            ],
            text={"format": {"type": "json_object"}},
            store=False,
        )
        api_mode = "responses_json_object"
    except Exception as exc:
        fallback_errors.append({"stage": "responses_json_object", "error": f"{type(exc).__name__}: {exc}"})
        # Fallback 2: plain text, JSON extracted from the response.
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": json_system + " No markdown fences."},
                {"role": "user", "content": user_content},
            ],
            store=False,
        )
        api_mode = "responses_json_text_fallback"

    text = _extract_response_text(response)
    if not text:
        raise RuntimeError("OpenAI returned no text for the contract-advisor output.")
    data = _normalize_llm_payload(_parse_json_object_from_text(text))
    normalizer_dropped = data.pop("_normalizer_dropped", [])
    parsed = LLMContractAdvisorOutput.model_validate(data)
    return parsed, _audit(response, api_mode, {"normalizer_dropped": normalizer_dropped})


def _reference_enhance_contract(contract: DatasetContract, df: pd.DataFrame, reference_df: pd.DataFrame | None, report: dict[str, Any]) -> None:
    if reference_df is None:
        return
    for name, meta in contract.columns.items():
        if name not in df.columns or name not in reference_df.columns:
            continue
        reference_series = reference_df[name]
        role = infer_role(name, reference_series)
        if role in ALLOWED_ROLES:
            meta.role = role  # type: ignore[assignment]
        if role == "numeric":
            numeric = pd.to_numeric(reference_series, errors="coerce").dropna()
            if len(numeric):
                meta.valid_range = (float(numeric.min()), float(numeric.max()))
        if role == "categorical":
            values = _safe_unique_values(reference_series, 50)
            if values:
                meta.approved_allowed_values = values
        if role == "date":
            target, sources = _best_date_format(reference_series)
            if target:
                meta.target_format = target
                meta.source_formats = list(dict.fromkeys([target, *sources, *DATE_FORMAT_CANDIDATES]))
                meta.allow_format_normalization = True
        meta.sensitivity = _infer_sensitivity(name, meta.role)  # type: ignore[assignment]
        report["deterministic_reference_updates"].append({"column": name, "role": meta.role})


def _infer_sensitivity(column: str, role: str) -> str:
    name = column.casefold()
    if role in {"email", "phone"} or any(token in name for token in ("name", "address", "person_id", "patient_id")):
        return "direct_identifier"
    if any(token in name for token in ("age", "gender", "sex", "dob", "postcode", "zip")):
        return "quasi_identifier"
    return "unknown"


def _normalize_key(value: Any, *, case_sensitive: bool = False) -> str | None:
    return normalize_for_comparison(value, case_sensitive=case_sensitive)


def _set_if_text(current: str | None, proposed: str | None) -> str | None:
    proposed = (proposed or "").strip()
    return proposed if proposed else current


def _apply_llm_output(
    contract: DatasetContract,
    output: LLMContractAdvisorOutput | None,
    df: pd.DataFrame,
    reference_df: pd.DataFrame | None,
    report: dict[str, Any],
) -> None:
    if output is None:
        return
    metadata = contract.dataset_metadata
    if output.dataset.description:
        metadata.description = output.dataset.description.strip()
    if output.dataset.intended_reuse:
        metadata.intended_reuse = output.dataset.intended_reuse.strip()
    if output.dataset.privacy_status:
        metadata.privacy_status = output.dataset.privacy_status.strip()
    # License and provenance are intentionally not auto-filled from the LLM.
    if output.dataset.license:
        report["blocked_dataset_suggestions"].append({"field": "license", "reason": "LLM cannot invent or approve license."})
    if output.dataset.provenance:
        report["blocked_dataset_suggestions"].append({"field": "provenance", "reason": "LLM cannot invent or approve provenance."})

    for suggestion in output.columns:
        name = suggestion.column
        meta = contract.columns.get(name)
        if meta is None or name not in df.columns:
            report["blocked_column_suggestions"].append({"column": name, "reason": "Unknown column."})
            continue
        if suggestion.role:
            role = suggestion.role.strip().casefold()
            if role in ALLOWED_ROLES:
                meta.role = role  # type: ignore[assignment]
            else:
                report["blocked_column_suggestions"].append({"column": name, "field": "role", "value": suggestion.role, "reason": "Unsupported role."})
        if suggestion.description:
            meta.description = suggestion.description.strip()
        if suggestion.measurement_unit:
            # Unit is allowed only for numeric/measurement-like fields. Identifiers are not measurements.
            if meta.role == "numeric":
                meta.measurement_unit = suggestion.measurement_unit.strip()
            else:
                report["blocked_column_suggestions"].append({"column": name, "field": "measurement_unit", "reason": "Non-numeric columns do not receive measurement units."})
        if suggestion.sensitivity:
            sens = suggestion.sensitivity.strip().casefold()
            if sens in ALLOWED_SENSITIVITY:
                meta.sensitivity = sens  # type: ignore[assignment]
        if suggestion.approved_allowed_values:
            verified = _verify_allowed_values(name, suggestion.approved_allowed_values, df, reference_df, meta, report)
            if verified:
                meta.approved_allowed_values = verified
        if suggestion.canonical_mapping:
            verified_mapping = _verify_mapping(name, suggestion.canonical_mapping, df, reference_df, meta, report)
            if verified_mapping:
                existing = dict(meta.canonical_mapping)
                existing.update(verified_mapping)
                meta.canonical_mapping = existing
        if suggestion.target_format:
            meta.target_format = suggestion.target_format.strip()
        if suggestion.source_formats:
            meta.source_formats = list(dict.fromkeys([fmt.strip() for fmt in suggestion.source_formats if fmt.strip()]))
        if suggestion.regex:
            try:
                re.compile(suggestion.regex)
                meta.regex = suggestion.regex
            except re.error as exc:
                report["blocked_column_suggestions"].append({"column": name, "field": "regex", "reason": str(exc)})
        if suggestion.standard_or_code_system:
            meta.standard_or_code_system = suggestion.standard_or_code_system.strip()
        if suggestion.required is not None:
            meta.required = bool(suggestion.required)
        report["applied_column_suggestions"].append({"column": name, "confidence": suggestion.confidence, "rationale": suggestion.rationale})

    for rule in output.cross_field_rules:
        if rule.left_column not in df.columns:
            report["blocked_cross_field_rules"].append({"rule_id": rule.rule_id, "reason": "left_column is absent."})
            continue
        if rule.right_column is not None and rule.right_column not in df.columns:
            report["blocked_cross_field_rules"].append({"rule_id": rule.rule_id, "reason": "right_column is absent."})
            continue
        contract.cross_field_rules.append(CrossFieldRule(
            rule_id=rule.rule_id,
            description=rule.description,
            severity=rule.severity,
            left_column=rule.left_column,
            operator=rule.operator,
            right_column=rule.right_column,
            right_value=rule.right_value,
        ))
        report["applied_cross_field_rules"].append(rule.model_dump(mode="json"))


def _verify_allowed_values(
    column: str,
    values: list[str],
    df: pd.DataFrame,
    reference_df: pd.DataFrame | None,
    meta: ColumnContract,
    report: dict[str, Any],
) -> list[Any]:
    cleaned = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        key = _normalize_key(text, case_sensitive=meta.case_sensitive)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    if len(cleaned) > 100:
        report["blocked_column_suggestions"].append({"column": column, "field": "approved_allowed_values", "reason": "Too many values for a controlled vocabulary."})
        return []
    if reference_df is not None and column in reference_df.columns:
        ref_values = _safe_unique_values(reference_df[column], 100)
        if ref_values:
            ref_keys = {_normalize_key(v, case_sensitive=meta.case_sensitive) for v in ref_values}
            suggested_keys = {_normalize_key(v, case_sensitive=meta.case_sensitive) for v in cleaned}
            # In clean-reference mode, allowed values must not contradict the reference vocabulary.
            if not suggested_keys.issubset(ref_keys):
                report["blocked_column_suggestions"].append({
                    "column": column,
                    "field": "approved_allowed_values",
                    "reason": "LLM vocabulary contains values not present in clean reference.",
                    "blocked_values": sorted(str(v) for v in suggested_keys - ref_keys),
                })
                return [v for v in ref_values if _normalize_key(v, case_sensitive=meta.case_sensitive) in ref_keys]
    return cleaned


def _verify_mapping(
    column: str,
    mapping: dict[str, str],
    df: pd.DataFrame,
    reference_df: pd.DataFrame | None,
    meta: ColumnContract,
    report: dict[str, Any],
) -> dict[str, Any]:
    allowed = list(meta.effective_allowed_values())
    allowed_keys = {_normalize_key(v, case_sensitive=meta.case_sensitive): v for v in allowed}
    dirty_values = {_normalize_key(v, case_sensitive=meta.case_sensitive) for v in df[column].dropna().astype(str).unique()}
    verified: dict[str, Any] = {}
    for raw, canonical in mapping.items():
        raw_text = str(raw).strip()
        canonical_text = str(canonical).strip()
        if not raw_text or not canonical_text:
            continue
        raw_key = _normalize_key(raw_text, case_sensitive=meta.case_sensitive)
        canonical_key = _normalize_key(canonical_text, case_sensitive=meta.case_sensitive)
        if raw_key not in dirty_values:
            report["blocked_mappings"].append({"column": column, "raw": raw_text, "canonical": canonical_text, "reason": "Raw value was not observed in dataset."})
            continue
        if allowed and canonical_key not in allowed_keys:
            report["blocked_mappings"].append({"column": column, "raw": raw_text, "canonical": canonical_text, "reason": "Canonical value is not in approved allowed values."})
            continue
        # Critical safety gate: never convert one already-valid category into another valid category.
        if raw_key in allowed_keys and canonical_key != raw_key:
            report["blocked_mappings"].append({"column": column, "raw": raw_text, "canonical": canonical_text, "reason": "Raw value is already an approved valid value; mapping to a different valid value is unsafe."})
            continue
        # If clean reference contains the raw value as valid, do not remap it to another clean value.
        if reference_df is not None and column in reference_df.columns:
            ref_keys = {_normalize_key(v, case_sensitive=meta.case_sensitive) for v in reference_df[column].dropna().astype(str).unique()}
            if raw_key in ref_keys and canonical_key != raw_key:
                report["blocked_mappings"].append({"column": column, "raw": raw_text, "canonical": canonical_text, "reason": "Raw value appears as a valid clean-reference value."})
                continue
        if raw_key == canonical_key:
            continue
        verified[raw_text] = canonical_text
    return verified


def _finalize_contract(contract: DatasetContract, df: pd.DataFrame, reference_df: pd.DataFrame | None, report: dict[str, Any]) -> None:
    # Explicit blank strings must be governed as missing-like values.
    tokens = list(contract.missing_tokens or [])
    if "" not in tokens:
        tokens.append("")
    contract.missing_tokens = tokens

    for name, meta in contract.columns.items():
        if not meta.description.strip():
            meta.description = f"Column '{name}' from the uploaded tabular dataset. Domain-specific description requires confirmation."
        meta.sensitivity = _infer_sensitivity(name, meta.role) if meta.sensitivity == "unknown" else meta.sensitivity  # type: ignore[assignment]
        # If clean reference is present, set safe categorical vocabulary deterministically when feasible.
        if reference_df is not None and name in reference_df.columns and meta.role == "categorical" and not meta.effective_allowed_values():
            ref_values = _safe_unique_values(reference_df[name], 50)
            if ref_values:
                meta.approved_allowed_values = ref_values
        # Remove unsafe mappings introduced by any source.
        safe_mapping = _verify_mapping(name, dict(meta.canonical_mapping), df, reference_df, meta, report)
        if safe_mapping != meta.canonical_mapping:
            meta.canonical_mapping = safe_mapping

    if not contract.dataset_metadata.title:
        contract.dataset_metadata.title = contract.dataset_id
    if not contract.dataset_metadata.description:
        contract.dataset_metadata.description = f"Tabular dataset with {len(df)} rows and {len(df.columns)} columns profiled by C-DQC."
    if not contract.dataset_metadata.intended_reuse:
        contract.dataset_metadata.intended_reuse = "Data-quality assessment and reuse-readiness evaluation."
    if not contract.dataset_metadata.privacy_status:
        direct = [name for name, meta in contract.columns.items() if meta.sensitivity == "direct_identifier"]
        contract.dataset_metadata.privacy_status = "Contains direct identifiers: " + ", ".join(direct) if direct else "No direct identifier column inferred from the structural profile."

    unresolved = []
    if not contract.dataset_metadata.license:
        unresolved.append({"scope": "dataset", "field": "license", "reason": "License cannot be inferred from the data values or by the LLM."})
    if not contract.dataset_metadata.provenance:
        unresolved.append({"scope": "dataset", "field": "provenance", "reason": "Provenance cannot be inferred from the data values or by the LLM."})
    for name, meta in contract.columns.items():
        if meta.role == "numeric" and meta.measurement_unit is None and any(token in name.casefold() for token in ("age", "amount", "cost", "price", "weight", "height")):
            unresolved.append({"scope": "column", "column": name, "field": "measurement_unit", "reason": "Measurement unit requires confirmation."})
    report["unresolved_fields"] = unresolved


def generate_llm_assisted_contract(
    df: pd.DataFrame,
    *,
    dataset_id: str,
    title: str | None = None,
    reference_df: pd.DataFrame | None = None,
    config: LLMContractAdvisorConfig | None = None,
) -> tuple[DatasetContract, dict[str, Any]]:
    config = config or LLMContractAdvisorConfig(enabled=False)
    contract = generate_draft_contract(df, dataset_id=dataset_id, title=title or dataset_id)
    report: dict[str, Any] = {
        "component": "ContractAdvisorLLM",
        "llm_config": config.public_dict(),
        "llm_audit": {"enabled": False, "status": "disabled"},
        "deterministic_reference_updates": [],
        "applied_column_suggestions": [],
        "applied_cross_field_rules": [],
        "blocked_dataset_suggestions": [],
        "blocked_column_suggestions": [],
        "blocked_cross_field_rules": [],
        "blocked_mappings": [],
        "unresolved_fields": [],
        "contract_approval_status": "draft",
        "metrics_changed_by_llm": False,
    }
    _reference_enhance_contract(contract, df, reference_df, report)
    output = None
    if config.enabled:
        payload = build_contract_advisor_payload(
            df,
            dataset_id=dataset_id,
            title=title or dataset_id,
            reference_df=reference_df,
            config=config,
        )
        report["llm_payload_summary"] = {
            "column_count": len(payload["columns"]),
            "reference_supplied": reference_df is not None,
            "direct_identifier_values_sent": False,
            "raw_dataset_sent": False,
        }
        try:
            output, audit = _call_openai_contract_advisor(payload, config)
            report["llm_audit"] = audit
            report["llm_output"] = output.model_dump(mode="json") if output else None
        except Exception as exc:
            report["llm_audit"] = {"enabled": True, "status": "failed", "error": str(exc)}
            if config.fail_on_llm_error:
                raise
    _apply_llm_output(contract, output, df, reference_df, report)
    _finalize_contract(contract, df, reference_df, report)
    return contract, report
