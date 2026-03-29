import argparse
import json
import re
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.llm_client import call_llm_json, llm_available


def _read_requirements_document(path):
    return Path(path).read_text(encoding="utf-8")


def _extract_target_table(text):
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
    for heading in start_heading_candidates:
        section = _extract_section(text, heading, end_heading_candidates)
        if section.strip():
            return section
    return ""


def _extract_schema_columns(schema_section_text):
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
    """Return explicit rename pairs from the requirements mapping section."""
    rename_rules = {}
    pattern = re.compile(r'\s*-\s*"([A-Za-z0-9_]+)"\s*→\s*([A-Za-z0-9_]+)')
    for m in pattern.finditer(text):
        rename_rules[m.group(1)] = m.group(2)
    return rename_rules


def _is_likely_derived(description):
    desc = description.lower()
    return (
        "derived" in desc
        or "system-generated" in desc
        or "pipeline execution time" in desc
    )


def _build_source_to_target_mappings(source_columns, target_columns, rename_rules):
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
        '  "target_columns_expected": ["string"]\n'
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
    if use_llm and llm_available():
        try:
            system_prompt = (
                "You are a data engineering assistant that extracts deterministic "
                "source-to-target mappings from requirements documents. "
                "Always return strict JSON only."
            )
            user_prompt = _build_requirements_prompt(text)
            parsed = call_llm_json(system_prompt=system_prompt, user_prompt=user_prompt)
            # Minimal sanity checks before accepting LLM output
            required_keys = {
                "target_table",
                "source_columns",
                "target_columns",
                "source_to_target_mapping",
            }
            if required_keys.issubset(set(parsed.keys())):
                return parsed
        except Exception:
            # Fall back to deterministic parser when LLM is unavailable or malformed.
            pass

    return _parse_requirements_rule_based(text)


def save_source_to_target_mapping(requirements_path, output_path, use_llm=True):
    requirements_text = _read_requirements_document(requirements_path)
    parsed = parse_requirements(requirements_text, use_llm=use_llm)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    return parsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Read requirements text and generate source-to-target mapping JSON."
        )
    )
    parser.add_argument(
        "--requirements",
        default="data/raw/requirements_document.txt",
        help="Path to requirements document text file",
    )
    parser.add_argument(
        "--output",
        default="outputs/source_to_target_mapping.json",
        help="Path to generated JSON mapping output",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM calls and use deterministic parser only",
    )
    args = parser.parse_args()

    result = save_source_to_target_mapping(
        args.requirements,
        args.output,
        use_llm=not args.no_llm,
    )
    print(f"Generated mapping for target table: {result.get('target_table')}")
    print(f"Source columns parsed: {len(result.get('source_columns', []))}")
    print(f"Target columns parsed: {len(result.get('target_columns', []))}")
    print(f"Output written to: {args.output}")