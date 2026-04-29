import argparse
import json
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from agents.llm_client import call_llm_json, llm_available


def _as_json_safe(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _suggest_transformations(series, col_name):
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


def _llm_enrich_dictionary(profile):
    system_prompt = (
        "You are a data profiling assistant that produces concise, practical "
        "JSON data dictionaries for downstream ETL mapping agents."
    )
    user_prompt = _build_profiler_prompt(profile)
    llm_result = call_llm_json(system_prompt=system_prompt, user_prompt=user_prompt)
    if "data_dictionary" not in llm_result:
        raise RuntimeError("LLM response missing data_dictionary")
    return llm_result["data_dictionary"]


def build_data_dictionary(file_path, output_path, use_llm=True):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile a CSV and generate JSON data dictionary for downstream mapping agent."
    )
    parser.add_argument(
        "--input",
        default="data/raw/sample_transactions.csv",
        help="Input CSV path to profile",
    )
    parser.add_argument(
        "--output",
        default="outputs/source_data_dictionary.json",
        help="Output JSON data dictionary path",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM calls and use deterministic profiling only",
    )
    args = parser.parse_args()

    result = build_data_dictionary(args.input, args.output, use_llm=not args.no_llm)
    print(f"Profiled dataset: {args.input}")
    print(f"Rows: {result['row_count']} | Columns: {result['column_count']}")
    print(f"Output written to: {args.output}")