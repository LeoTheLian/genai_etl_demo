def generate_test_results(source_df, df, parsed_requirements):
    results = []
    
    # Tier 1 Checks
    # 1. dataset_non_empty
    if source_df.empty or df.empty:
        results.append({
            "name": "dataset_non_empty",
            "passed": False,
            "details": "One of the datasets is empty."
        })
    
    # 2. row_count_sanity
    source_row_count = source_df.shape[0]
    output_row_count = df.shape[0]
    input_row_count_from_requirements = "unknown"  # Placeholder for requirements-based count
    expected_required_removals_estimate = 0  # Placeholder for expected removals
    observed_removed_count = source_row_count - output_row_count
    
    if output_row_count < source_row_count:
        results.append({
            "name": "row_count_sanity",
            "passed": True,
            "details": f"Source row count: {source_row_count}, Output row count: {output_row_count}, Observed removed count: {observed_removed_count}."
        })
    else:
        results.append({
            "name": "row_count_sanity",
            "passed": False,
            "details": f"Output row count {output_row_count} is not less than source row count {source_row_count}."
        })
    
    # 3. expected_columns_present
    expected_columns = [col['name'] for col in parsed_requirements['target_columns']]
    present_columns = df.columns.tolist()
    missing_target_columns = [col for col in expected_columns if col not in present_columns]
    
    results.append({
        "name": "expected_columns_present",
        "passed": len(missing_target_columns) == 0,
        "details": f"Expected target count: {len(expected_columns)}, Present target count: {len(present_columns)}, Missing target columns: {missing_target_columns}."
    })
    
    # 4. required_fields_not_null
    required_fields = ['transaction_id', 'transaction_timestamp', 'transaction_amount', 'is_fraud']
    null_counts = {field: df[field].isnull().sum() for field in required_fields if field in df.columns}
    failed_null_checks = {field: count for field, count in null_counts.items() if count > 0}
    
    results.append({
        "name": "required_fields_not_null",
        "passed": len(failed_null_checks) == 0,
        "details": f"Null counts for required fields: {failed_null_checks}."
    })
    
    # 5. id_uniqueness
    if df['transaction_id'].is_unique:
        results.append({
            "name": "id_uniqueness",
            "passed": True,
            "details": "All transaction IDs are unique."
        })
    else:
        results.append({
            "name": "id_uniqueness",
            "passed": False,
            "details": "Duplicate transaction IDs found."
        })
    
    # 6. duplicate_rows_absent
    if df.duplicated().sum() == 0:
        results.append({
            "name": "duplicate_rows_absent",
            "passed": True,
            "details": "No duplicate rows found."
        })
    else:
        results.append({
            "name": "duplicate_rows_absent",
            "passed": False,
            "details": "Duplicate rows found."
        })
    
    # 7. numeric_type_coercion_success
    numeric_columns = ['transaction_amount', 'credit_limit', 'account_balance', 'avg_amount_30d']
    type_coercion_success = all(df[col].dtype in [float, 'float64'] for col in numeric_columns if col in df.columns)
    
    results.append({
        "name": "numeric_type_coercion_success",
        "passed": type_coercion_success,
        "details": "Numeric type coercion success check."
    })
    
    # 8. timestamp_parse_success
    if pd.to_datetime(df['transaction_timestamp'], errors='coerce').notnull().all():
        results.append({
            "name": "timestamp_parse_success",
            "passed": True,
            "details": "All timestamps parsed successfully."
        })
    else:
        results.append({
            "name": "timestamp_parse_success",
            "passed": False,
            "details": "Some timestamps failed to parse."
        })
    
    return results
