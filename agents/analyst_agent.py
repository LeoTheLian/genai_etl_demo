"""
Analyst Agent — parses requirements and produces a source-to-target mapping.

This agent reads the requirements document and generates the source-to-target
mapping JSON that drives the downstream developer agent.

Data profiling utilities (profile_data, build_data_dictionary) are still
available as standalone functions for ad-hoc use.

Usage:
  python agents/analyst_agent.py
  python agents/analyst_agent.py --requirements data/raw/requirements_document.txt
  python agents/analyst_agent.py --no-llm
"""

import argparse
import json
import re
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agents.llm_client import llm_available, make_llm
from agents.prompt_logger import PromptResponseLogger


# ---------------------------------------------------------------------------
# Data Profiler Functions
# ---------------------------------------------------------------------------


def _as_json_safe(value):
    """Convert pandas types to JSON-serializable values."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _suggest_transformations(series, col_name):
    """Suggest ETL transformations based on column data and name."""
    suggestions = []
    non_null = series.dropna()
    lower_col = col_name.lower()

    if series.isna().any():
        suggestions.append("handle_nulls")

    if pd.api.types.is_numeric_dtype(series):
        if (non_null < 0).any():
            suggestions.append("validate_non_negative")
        if len(non_null) > 0:
            q95 = non_null.quantile(0.95)
            max_v = non_null.max()
            if q95 != 0 and max_v > q95 * 3:
                suggestions.append("flag_outliers")
    else:
        # String cleanup suggestions
        text_values = non_null.astype(str)
        if (text_values.str.strip() != text_values).any():
            suggestions.append("trim_whitespace")
        if (text_values == "").any():
            suggestions.append("replace_blank_with_null_or_default")

    # Column-name-driven semantic transformation hints
    if lower_col == "time":
        suggestions.append("convert_seconds_to_timestamp")
    if lower_col == "amount":
        suggestions.append("rename_to_transaction_amount")
        suggestions.append("cast_to_decimal_10_2")
    if lower_col == "class":
        suggestions.append("rename_to_is_fraud")
        suggestions.append("validate_binary_values")
    if "country" in lower_col:
        suggestions.append("standardize_country_code")
    if "state" in lower_col:
        suggestions.append("uppercase_state_code")
    if "age" in lower_col:
        suggestions.append("validate_reasonable_age_range")
    if "id" in lower_col:
        suggestions.append("validate_identifier_format")

    # Dedupe while preserving order
    deduped = []
    seen = set()
    for s in suggestions:
        if s not in seen:
            deduped.append(s)
            seen.add(s)
    return deduped


def profile_data(file_path):
    """Profile a CSV file to extract column statistics and metadata."""
    df = pd.read_csv(file_path)

    profile = {
        "dataset": str(file_path),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": [],
    }

    for col in df.columns:
        series = df[col]
        non_null = series.dropna()

        col_profile = {
            "name": col,
            "dtype": str(series.dtype),
            "null_count": int(series.isna().sum()),
            "null_pct": float(series.isna().mean()),
            "non_null_count": int(non_null.shape[0]),
            "distinct_count": int(non_null.nunique(dropna=True)),
            "sample_values": [_as_json_safe(v) for v in non_null.head(5).tolist()],
            "expected_transformations": _suggest_transformations(series, col),
        }

        if pd.api.types.is_numeric_dtype(series):
            col_profile["stats"] = {
                "min": _as_json_safe(non_null.min()) if len(non_null) else None,
                "max": _as_json_safe(non_null.max()) if len(non_null) else None,
                "mean": _as_json_safe(non_null.mean()) if len(non_null) else None,
                "median": _as_json_safe(non_null.median()) if len(non_null) else None,
            }
        else:
            top_values = (
                non_null.astype(str).value_counts(dropna=True).head(5).to_dict()
                if len(non_null)
                else {}
            )
            col_profile["top_values"] = {k: int(v) for k, v in top_values.items()}

        profile["columns"].append(col_profile)

    return profile


def _build_profiler_prompt(profile):
    """Build LLM prompt for data profiling enrichment."""
    return (
        "Given this dataset profile JSON, return ONLY valid JSON with this schema:\n"
        "{\n"
        '  "data_dictionary": [\n'
        "    {\n"
        '      "name": "string",\n'
        '      "dtype": "string",\n'
        '      "business_meaning": "string",\n'
        '      "null_pct": number,\n'
        '      "sample_values": ["..."],\n'
        '      "expected_transformations": ["..."]\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Keep one entry per input column.\n"
        "- expected_transformations should be actionable ETL steps.\n"
        "- Do not invent extra columns.\n"
        "- No markdown, no explanation, JSON only.\n\n"
        "Dataset profile:\n"
        f"{json.dumps(profile, indent=2)}"
    )


_PROFILER_SYSTEM_PROMPT = (
    "You are a data profiling assistant that produces concise, practical "
    "JSON data dictionaries for downstream ETL mapping agents."
)

_REQUIREMENTS_SYSTEM_PROMPT = (
    "You are a data engineering assistant that extracts deterministic "
    "source-to-target mappings from requirements documents. "
    "Always return strict JSON only."
)


def _llm_enrich_dictionary(profile):
    """Enrich data profile with LLM-generated business meanings and insights."""
    human_prompt = _build_profiler_prompt(profile)
    prompt = ChatPromptTemplate.from_messages([
        ("system", _PROFILER_SYSTEM_PROMPT),
        ("human", "{user_prompt}"),
    ])
    chain = prompt | make_llm() | StrOutputParser()
    raw_response = chain.invoke({"user_prompt": human_prompt})
    try:
        PromptResponseLogger().log(
            agent="analyst",
            call_type="data_profiling",
            system_prompt=_PROFILER_SYSTEM_PROMPT,
            human_prompt=human_prompt,
            raw_response=raw_response,
        )
    except Exception:
        pass
    result = JsonOutputParser().parse(raw_response)
    if "data_dictionary" not in result:
        raise RuntimeError("LLM response missing data_dictionary")
    return result["data_dictionary"]


def build_data_dictionary(file_path, output_path, use_llm=True):
    """Profile a CSV file and generate data dictionary JSON."""
    profile = profile_data(file_path)

    data_dictionary = profile["columns"]
    if use_llm and llm_available():
        try:
            llm_dictionary = _llm_enrich_dictionary(profile)
            # Accept LLM output if size aligns with source columns
            if isinstance(llm_dictionary, list) and len(llm_dictionary) == len(profile["columns"]):
                data_dictionary = llm_dictionary
        except Exception:
            # Preserve deterministic output if LLM is unavailable/malformed.
            pass

    dictionary = {
        "dataset": profile["dataset"],
        "row_count": profile["row_count"],
        "column_count": profile["column_count"],
        "data_dictionary": data_dictionary,
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(dictionary, indent=2), encoding="utf-8")
    return dictionary


# ---------------------------------------------------------------------------
# Requirements Parser Functions
# ---------------------------------------------------------------------------


def _read_requirements_document(path):
    """Read requirements document text file."""
    return Path(path).read_text(encoding="utf-8")


def _extract_target_table(text):
    """Extract target table name from requirements text."""
    match = re.search(r"Target Table:\s*\n\s*([A-Za-z_][A-Za-z0-9_]*)", text)
    if match:
        return match.group(1).strip()

    alt = re.search(
        r"Target Schema\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s+Table\s*\)",
        text,
        flags=re.IGNORECASE,
    )
    return alt.group(1).strip() if alt else None


def _extract_section(text, start_heading, end_heading_candidates):
    """Extract a section of text between two headings."""
    start_idx = text.find(start_heading)
    if start_idx == -1:
        return ""

    section_text = text[start_idx + len(start_heading):]
    end_indexes = []
    for heading in end_heading_candidates:
        i = section_text.find(heading)
        if i != -1:
            end_indexes.append(i)

    if not end_indexes:
        return section_text

    return section_text[: min(end_indexes)]


def _extract_section_any(text, start_heading_candidates, end_heading_candidates):
    """Extract a section trying multiple start headings."""
    for heading in start_heading_candidates:
        section = _extract_section(text, heading, end_heading_candidates)
        if section.strip():
            return section
    return ""


def _extract_schema_columns(schema_section_text):
    """Extract column definitions from schema section text."""
    columns = []
    for raw_line in schema_section_text.splitlines():
        line = raw_line.rstrip()

        # Expected bullet format:
        # - field_name: datatype - description
        field_match = re.match(r"\s*-\s*([A-Za-z0-9_]+)\s*:\s*([^\n-]+?)(?:\s*[—-]\s*(.*))?$", line)
        if field_match:
            name = field_match.group(1).strip()
            dtype = field_match.group(2).strip()
            description = (field_match.group(3) or "").strip()
            columns.append(
                {
                    "name": name,
                    "type": dtype,
                    "description": description,
                }
            )

    return columns


def _extract_rename_rules(text):
    """Extract explicit rename pairs from the requirements mapping section."""
    rename_rules = {}
    pattern = re.compile(r'\s*-\s*"([A-Za-z0-9_]+)"\s*→\s*([A-Za-z0-9_]+)')
    for m in pattern.finditer(text):
        source, target = m.group(1), m.group(2)
        if source != target:  # skip identity mappings — they pollute rename_rules count
            rename_rules[source] = target
    return rename_rules


def _is_likely_derived(description):
    """Determine if a column is derived/system-generated based on description."""
    desc = description.lower()
    return (
        "derived" in desc
        or "system-generated" in desc
        or "pipeline execution time" in desc
    )


def _build_source_to_target_mappings(source_columns, target_columns, rename_rules):
    """Build source-to-target mapping rules from schema definitions."""
    source_by_name = {c["name"]: c for c in source_columns}
    target_names = {c["name"] for c in target_columns}

    reverse_rename = {target: source for source, target in rename_rules.items()}
    mappings = []

    for target_col in target_columns:
        t_name = target_col["name"]
        t_type = target_col["type"]
        t_desc = target_col["description"]

        # Priority 1: Explicit rename rule
        if t_name in reverse_rename:
            src_name = reverse_rename[t_name]
            mappings.append(
                {
                    "source_column": src_name,
                    "target_column": t_name,
                    "target_type": t_type,
                    "mapping_type": "rename",
                    "transformation": f"Rename {src_name} to {t_name}",
                    "description": t_desc,
                }
            )
            continue

        # Priority 2: Pass-through if same column exists in source
        if t_name in source_by_name:
            mappings.append(
                {
                    "source_column": t_name,
                    "target_column": t_name,
                    "target_type": t_type,
                    "mapping_type": "pass_through",
                    "transformation": "Pass through with optional standardization",
                    "description": t_desc,
                }
            )
            continue

        # Priority 3: Derived/system columns
        mapping_type = "derived" if _is_likely_derived(t_desc) else "target_only"
        mappings.append(
            {
                "source_column": None,
                "target_column": t_name,
                "target_type": t_type,
                "mapping_type": mapping_type,
                "transformation": t_desc if t_desc else "Derived or system-generated",
                "description": t_desc,
            }
        )

    # Optional visibility: source columns not mapped directly to target
    mapped_sources = {
        m["source_column"] for m in mappings if m.get("source_column") is not None
    }
    unmapped_sources = sorted(set(source_by_name) - mapped_sources)

    return mappings, unmapped_sources, sorted(target_names)


def _parse_requirements_rule_based(text):
    """
    Parse requirements text and produce a structured representation including
    source schema, target schema, and source-to-target mappings.
    """
    target_table = _extract_target_table(text)

    source_schema_text = _extract_section_any(
        text,
        [
            "Source Schema (as received from vendor):",
            "Source Schema (As Received from Vendor)",
        ],
        ["---", "Target Table:", "Target Schema:"],
    )
    target_schema_text = _extract_section_any(
        text,
        [
            "Target Schema:",
            "Target Schema (fraud_transactions Table)",
        ],
        ["---", "Transformation Requirements:"],
    )

    source_columns = _extract_schema_columns(source_schema_text)
    target_columns = _extract_schema_columns(target_schema_text)
    rename_rules = _extract_rename_rules(text)

    mappings, unmapped_sources, target_names = _build_source_to_target_mappings(
        source_columns=source_columns,
        target_columns=target_columns,
        rename_rules=rename_rules,
    )

    return {
        "target_table": target_table,
        "source_columns": source_columns,
        "target_columns": target_columns,
        "rename_rules": rename_rules,
        "source_to_target_mapping": mappings,
        "unmapped_source_columns": unmapped_sources,
        "target_columns_expected": target_names,
    }


def _build_requirements_prompt(text):
    """Build LLM prompt for requirements parsing."""
    return (
        "Read the requirements text and return ONLY valid JSON with this schema:\n"
        "{\n"
        '  "target_table": "string",\n'
        '  "source_columns": [{"name": "string", "type": "string", "description": "string"}],\n'
        '  "target_columns": [{"name": "string", "type": "string", "description": "string"}],\n'
        '  "rename_rules": {"source_col": "target_col"},\n'
        '  "source_to_target_mapping": [\n'
        "    {\n"
        '      "source_column": "string or null",\n'
        '      "target_column": "string",\n'
        '      "target_type": "string",\n'
        '      "mapping_type": "rename|pass_through|derived|target_only",\n'
        '      "transformation": "string",\n'
        '      "description": "string"\n'
        "    }\n"
        "  ],\n"
        '  "unmapped_source_columns": ["string"],\n'
        '  "target_columns_expected": ["string"],\n'
        "}\n\n"
        "Rules:\n"
        "- Use explicit rename rules if listed.\n"
        "- Keep pass-through mappings for same-name columns.\n"
        "- Include derived/system columns with source_column = null.\n"
        "- No markdown, no explanation, JSON only.\n\n"
        "Requirements text:\n"
        f"{text}"
    )


def parse_requirements(text, use_llm=True):
    """Parse requirements text using LLM or deterministic fallback."""
    if use_llm and llm_available():
        try:
            human_prompt = _build_requirements_prompt(text)
            prompt = ChatPromptTemplate.from_messages([
                ("system", _REQUIREMENTS_SYSTEM_PROMPT),
                ("human", "{user_prompt}"),
            ])
            chain = prompt | make_llm() | StrOutputParser()
            raw_response = chain.invoke({"user_prompt": human_prompt})
            try:
                PromptResponseLogger().log(
                    agent="analyst",
                    call_type="requirements_parsing",
                    system_prompt=_REQUIREMENTS_SYSTEM_PROMPT,
                    human_prompt=human_prompt,
                    raw_response=raw_response,
                )
            except Exception:
                pass
            parsed = JsonOutputParser().parse(raw_response)
            required_keys = {
                "target_table",
                "source_columns",
                "target_columns",
                "source_to_target_mapping",
            }
            if required_keys.issubset(set(parsed.keys())):
                return parsed
        except Exception:
            pass

    return _parse_requirements_rule_based(text)


def save_source_to_target_mapping(requirements_path, output_path, use_llm=True):
    """Parse requirements document and save source-to-target mapping JSON."""
    requirements_text = _read_requirements_document(requirements_path)
    parsed = parse_requirements(requirements_text, use_llm=use_llm)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    return parsed


# ---------------------------------------------------------------------------
# Consolidated Public Entry Point
# ---------------------------------------------------------------------------


def analyze_requirements_and_data(
    requirements_path,
    output_mapping_path,
    use_llm=True,
):
    """
    Parse requirements document and save source-to-target mapping JSON.

    Args:
        requirements_path: Path to requirements document
        output_mapping_path: Where to write source-to-target mapping JSON
        use_llm: Whether to use LLM for enrichment (falls back to rule-based)

    Returns:
        {
            "mapping": {...mapping dict...},
            "mapping_output_path": str,
        }
    """
    mapping = save_source_to_target_mapping(
        requirements_path=requirements_path,
        output_path=output_mapping_path,
        use_llm=use_llm,
    )

    return {
        "mapping": mapping,
        "mapping_output_path": str(output_mapping_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyst Agent: parse requirements document and produce source-to-target mapping JSON."
    )
    parser.add_argument(
        "--requirements",
        default="data/raw/requirements_document.txt",
        help="Path to requirements document text file",
    )
    parser.add_argument(
        "--output-mapping",
        default="outputs/source_to_target_mapping.json",
        help="Output path for source-to-target mapping JSON",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM calls and use deterministic rule-based parsing only",
    )
    args = parser.parse_args()

    result = analyze_requirements_and_data(
        requirements_path=args.requirements,
        output_mapping_path=args.output_mapping,
        use_llm=not args.no_llm,
    )

    mapping = result["mapping"]
    print(f"Target table  : {mapping.get('target_table')}")
    print(f"Source columns: {len(mapping.get('source_columns', []))}")
    print(f"Target columns: {len(mapping.get('target_columns', []))}")
    print(f"Mappings      : {len(mapping.get('source_to_target_mapping', []))}")
    print(f"Output        : {result['mapping_output_path']}")
