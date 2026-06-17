def generate_test_results(source_df, df, parsed_requirements):
    results = []

    # Test: expected_columns_present
    try:
        expected_columns = {col['name'] for col in parsed_requirements['target_columns']}
        present_columns = set(df.columns)
        missing_columns = expected_columns - present_columns
        legacy_source_names = {col['name'] for col in parsed_requirements['source_columns'] if col['name'] in source_df.columns and col['name'] not in parsed_requirements['rename_rules']}
        results.append({
            'name': 'expected_columns_present',
            'passed': len(missing_columns) == 0,
            'details': f"expected_target_count={len(expected_columns)}, present_target_count={len(present_columns)}, missing_target_columns={list(missing_columns)}, legacy_source_names={list(legacy_source_names)}"
        })
    except Exception as e:
        results.append({
            'name': 'expected_columns_present',
            'passed': False,
            'details': str(e)
        })

    # Test: rename_mapping_applied
    try:
        actual_renames = {src: tgt for src, tgt in parsed_requirements['rename_rules'].items() if src != tgt}
        rename_pass = all(
            tgt in df.columns and src not in df.columns
            for src, tgt in actual_renames.items()
        )
        results.append({
            'name': 'rename_mapping_applied',
            'passed': rename_pass,
            'details': f"actual_renames={actual_renames}"
        })
    except Exception as e:
        results.append({
            'name': 'rename_mapping_applied',
            'passed': False,
            'details': str(e)
        })

    # Test: timestamp_conversion_logic
    try:
        df['transaction_timestamp'] = pd.to_datetime(df['transaction_timestamp'], errors='coerce')
        valid_range = (df['transaction_timestamp'] >= '2020-01-01') & (df['transaction_timestamp'] <= '2030-01-01')
        timestamp_pass = df['transaction_timestamp'].notna().all() and valid_range.all()
        results.append({
            'name': 'timestamp_conversion_logic',
            'passed': timestamp_pass,
            'details': f"invalid_timestamps_count={(~valid_range).sum()}"
        })
    except Exception as e:
        results.append({
            'name': 'timestamp_conversion_logic',
            'passed': False,
            'details': str(e)
        })

    # Test: merchant_country_standardization
    try:
        invalid_countries = df['merchant_country'].isin(['USA', 'us', 'U.S.', 'United States'])
        country_pass = not invalid_countries.any()
        results.append({
            'name': 'merchant_country_standardization',
            'passed': country_pass,
            'details': f"invalid_country_count={invalid_countries.sum()}"
        })
    except Exception as e:
        results.append({
            'name': 'merchant_country_standardization',
            'passed': False,
            'details': str(e)
        })

    # Test: merchant_name_blank_handling
    try:
        blank_names = df['merchant_name'].str.strip() == ''
        name_pass = not blank_names.any()
        results.append({
            'name': 'merchant_name_blank_handling',
            'passed': name_pass,
            'details': f"blank_name_count={blank_names.sum()}"
        })
    except Exception as e:
        results.append({
            'name': 'merchant_name_blank_handling',
            'passed': False,
            'details': str(e)
        })

    # Test: type_standardization_checks
    try:
        type_checks = {
            'transaction_amount': pd.api.types.is_float_dtype,
            'credit_limit': pd.api.types.is_float_dtype,
            'account_balance': pd.api.types.is_float_dtype,
            'avg_amount_30d': pd.api.types.is_float_dtype,
            'distance_from_home_km': pd.api.types.is_float_dtype,
            'utilization_rate': pd.api.types.is_float_dtype,
            'is_fraud': pd.api.types.is_integer_dtype,
            'is_foreign_transaction': pd.api.types.is_integer_dtype,
            'is_weekend': pd.api.types.is_integer_dtype,
            'is_high_risk_context': pd.api.types.is_integer_dtype,
            'cardholder_age': pd.api.types.is_integer_dtype,
            'account_age_days': pd.api.types.is_integer_dtype,
            'num_transactions_24h': pd.api.types.is_integer_dtype,
            'num_declined_7d': pd.api.types.is_integer_dtype
        }
        type_pass = all(check(df[col]) for col, check in type_checks.items())
        results.append({
            'name': 'type_standardization_checks',
            'passed': type_pass,
            'details': f"failed_columns={[col for col, check in type_checks.items() if not check(df[col])]}"
        })
    except Exception as e:
        results.append({
            'name': 'type_standardization_checks',
            'passed': False,
            'details': str(e)
        })

    # Test: output_schema_rowcount_sanity
    try:
        source_row_count = len(source_df)
        output_row_count = len(df)
        removed_count = source_row_count - output_row_count
        results.append({
            'name': 'output_schema_rowcount_sanity',
            'passed': output_row_count > 0,
            'details': f"source_row_count={source_row_count}, output_row_count={output_row_count}, removed_count={removed_count}, rationale=ETL filtering"
        })
    except Exception as e:
        results.append({
            'name': 'output_schema_rowcount_sanity',
            'passed': False,
            'details': str(e)
        })

    return results
