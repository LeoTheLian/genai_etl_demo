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
    expected_required_removals_estimate = 20 + 30 + 15 + 10 + 10 + 8  # Known issues
    observed_removed_count = source_row_count - output_row_count
    row_count_pass = output_row_count > 0 and output_row_count <= source_row_count - expected_required_removals_estimate
    
    results.append({
        "name": "row_count_sanity",
        "passed": row_count_pass,
        "details": f"Source row count: {source_row_count}, Output row count: {output_row_count}, Expected removals: {expected_required_removals_estimate}, Observed removals: {observed_removed_count}."
    })
    
    # 3. expected_columns_present
    expected_columns = [col["name"] for col in parsed_requirements["target_columns"]]
    present_columns = df.columns.tolist()
    missing_columns = [col for col in expected_columns if col not in present_columns]
    rename_coverage = {k: v for k, v in parsed_requirements.get("rename_rules", {}).items() if v in present_columns}
    legacy_names_present = [col for col in parsed_requirements.get("rename_rules", {}).keys() if col in present_columns]
    
    results.append({
        "name": "expected_columns_present",
        "passed": not missing_columns,
        "details": f"Expected count: {len(expected_columns)}, Present count: {len(present_columns)}, Missing: {missing_columns}, Rename coverage: {rename_coverage}, Legacy names present: {legacy_names_present}."
    })
    
    # 4. required_fields_not_null
    required_fields = ["transaction_id", "transaction_timestamp", "transaction_amount", "is_fraud", "account_id", "merchant_id", "channel"]
    null_counts = {field: df[field].isnull().sum() for field in required_fields if field in df.columns}
    null_fields = [field for field, count in null_counts.items() if count > 0]
    
    results.append({
        "name": "required_fields_not_null",
        "passed": not null_fields,
        "details": f"Null counts for required fields: {null_counts}."
    })
    
    # 5. id_uniqueness
    unique_ids = df["transaction_id"].is_unique
    results.append({
        "name": "id_uniqueness",
        "passed": unique_ids,
        "details": "transaction_id uniqueness check."
    })
    
    # 6. duplicate_rows_absent
    duplicate_rows = df.duplicated().sum()
    results.append({
        "name": "duplicate_rows_absent",
        "passed": duplicate_rows == 0,
        "details": f"Duplicate rows found: {duplicate_rows}."
    })
    
    # 7. numeric_type_coercion_success
    numeric_columns = ["transaction_amount", "credit_limit", "account_balance", "avg_amount_30d"]
    type_coercion_pass = all(df[col].dtype in [float, int] for col in numeric_columns if col in df.columns)
    
    results.append({
        "name": "numeric_type_coercion_success",
        "passed": type_coercion_pass,
        "details": "Numeric type coercion check."
    })
    
    # 8. timestamp_parse_success
    timestamp_columns = ["transaction_timestamp"]
    timestamp_parse_pass = all(pd.to_datetime(df[col], errors='coerce').notnull().all() for col in timestamp_columns if col in df.columns)
    
    results.append({
        "name": "timestamp_parse_success",
        "passed": timestamp_parse_pass,
        "details": "Timestamp parsing check."
    })
    
    return results
