def generate_test_results(source_df, df, parsed_requirements):
    results = []
    
    # Tier 1 Checks
    # 1. dataset_non_empty
    if source_df.empty or df.empty:
        results.append({
            "name": "dataset_non_empty",
            "passed": False,
            "details": "One or both datasets are empty."
        })
    
    # 2. row_count_sanity
    source_row_count = source_df.shape[0]
    output_row_count = df.shape[0]
    input_row_count_from_requirements = "unknown"  # Placeholder for requirements-based count
    expected_required_removals_estimate = 20 + 30 + 15 + 10 + 8  # Known issues
    observed_removed_count = source_row_count - output_row_count
    row_count_pass = observed_removed_count >= expected_required_removals_estimate
    results.append({
        "name": "row_count_sanity",
        "passed": row_count_pass,
        "details": f"Source: {source_row_count}, Output: {output_row_count}, Expected Removals: {expected_required_removals_estimate}, Observed Removed: {observed_removed_count}"
    })
    
    # 3. expected_columns_present
    expected_columns = [col["name"] for col in parsed_requirements["target_columns"]]
    present_columns = df.columns.tolist()
    missing_target_columns = list(set(expected_columns) - set(present_columns))
    rename_coverage = {k: v for k, v in parsed_requirements.get("rename_rules", {}).items() if v in present_columns}
    legacy_names_present = [name for name in parsed_requirements.get("rename_rules", {}).keys() if name in present_columns]
    results.append({
        "name": "expected_columns_present",
        "passed": not missing_target_columns,
        "details": f"Expected: {len(expected_columns)}, Present: {len(present_columns)}, Missing: {missing_target_columns}, Rename Coverage: {rename_coverage}, Legacy Names Present: {legacy_names_present}"
    })
    
    # 4. required_fields_not_null
    required_fields = ["transaction_id", "transaction_amount", "is_fraud", "account_id", "account_age_days", "cardholder_age", "merchant_name"]
    null_counts = df[required_fields].isnull().sum()
    null_fields = null_counts[null_counts > 0].index.tolist()
    results.append({
        "name": "required_fields_not_null",
        "passed": not null_fields,
        "details": f"Null fields: {null_fields}"
    })
    
    # 5. id_uniqueness
    id_uniqueness_pass = df["transaction_id"].is_unique
    results.append({
        "name": "id_uniqueness",
        "passed": id_uniqueness_pass,
        "details": "Transaction IDs are unique." if id_uniqueness_pass else "Duplicate transaction IDs found."
    })
    
    # 6. duplicate_rows_absent
    duplicate_rows_pass = not df.duplicated().any()
    results.append({
        "name": "duplicate_rows_absent",
        "passed": duplicate_rows_pass,
        "details": "No duplicate rows found." if duplicate_rows_pass else "Duplicate rows exist."
    })
    
    # 7. numeric_type_coercion_success
    numeric_columns = ["transaction_amount", "credit_limit", "account_balance", "avg_amount_30d"]
    numeric_type_pass = all(df[col].dtype in [float, int] for col in numeric_columns)
    results.append({
        "name": "numeric_type_coercion_success",
        "passed": numeric_type_pass,
        "details": "Numeric types are correct." if numeric_type_pass else "Numeric type coercion failed."
    })
    
    # 8. timestamp_parse_success
    timestamp_columns = ["transaction_timestamp", "ingestion_timestamp"]
    timestamp_parse_pass = all(pd.to_datetime(df[col], errors='coerce').notnull().all() for col in timestamp_columns)
    results.append({
        "name": "timestamp_parse_success",
        "passed": timestamp_parse_pass,
        "details": "Timestamps parsed successfully." if timestamp_parse_pass else "Timestamp parsing failed."
    })
    
    return results
