def generate_test_results(source_df, df, parsed_requirements):
    results = []

    try:
        expected_columns = {col['name'] for col in parsed_requirements['target_columns']}
        present_columns = set(df.columns)
        missing_columns = expected_columns - present_columns
        expected_target_count = len(expected_columns)
        present_target_count = len(present_columns)
        rename_coverage = {src: tgt for src, tgt in parsed_requirements['rename_rules'].items() if src != tgt}
        rename_pass = all(tgt in present_columns and src not in present_columns for src, tgt in rename_coverage.items())
        legacy_source_names = {col['name'] for col in parsed_requirements['source_columns'] if col['name'] in present_columns}

        results.append({
            'name': 'expected_columns_present',
            'passed': len(missing_columns) == 0,
            'details': f"Expected target count: {expected_target_count}, Present target count: {present_target_count}, "
                       f"Missing target columns: {missing_columns}, Rename coverage: {rename_pass}, "
                       f"Legacy source names still present: {legacy_source_names}"
        })
    except Exception as e:
        results.append({
            'name': 'expected_columns_present',
            'passed': False,
            'details': str(e)
        })

    try:
        actual_renames = {src: tgt for src, tgt in parsed_requirements['rename_rules'].items() if src != tgt}
        rename_pass = all(tgt in df.columns and src not in df.columns for src, tgt in actual_renames.items())
        results.append({
            'name': 'rename_mapping_applied',
            'passed': rename_pass,
            'details': f"Rename mapping applied: {rename_pass}"
        })
    except Exception as e:
        results.append({
            'name': 'rename_mapping_applied',
            'passed': False,
            'details': str(e)
        })

    try:
        timestamp_valid = pd.to_datetime(df['transaction_timestamp'], errors='coerce')
        valid_range = (timestamp_valid >= '2020-01-01') & (timestamp_valid <= '2030-01-01')
        results.append({
            'name': 'timestamp_conversion_logic',
            'passed': valid_range.all(),
            'details': f"All timestamps valid: {valid_range.all()}"
        })
    except Exception as e:
        results.append({
            'name': 'timestamp_conversion_logic',
            'passed': False,
            'details': str(e)
        })

    try:
        raw_variants = ['USA', 'us', 'U.S.', 'United States']
        contains_variants = df['merchant_country'].isin(raw_variants).any()
        results.append({
            'name': 'merchant_country_standardization',
            'passed': not contains_variants,
            'details': f"Contains raw variants: {contains_variants}"
        })
    except Exception as e:
        results.append({
            'name': 'merchant_country_standardization',
            'passed': False,
            'details': str(e)
        })

    try:
        blank_names = (df['merchant_name'].str.strip() == '').sum()
        results.append({
            'name': 'merchant_name_blank_handling',
            'passed': blank_names == 0,
            'details': f"Blank merchant names count: {blank_names}"
        })
    except Exception as e:
        results.append({
            'name': 'merchant_name_blank_handling',
            'passed': False,
            'details': str(e)
        })

    try:
        type_checks = {
            'transaction_amount': pd.api.types.is_float_dtype(df['transaction_amount']),
            'credit_limit': pd.api.types.is_float_dtype(df['credit_limit']),
            'account_balance': pd.api.types.is_float_dtype(df['account_balance']),
            'avg_amount_30d': pd.api.types.is_float_dtype(df['avg_amount_30d']),
            'distance_from_home_km': pd.api.types.is_float_dtype(df['distance_from_home_km']),
            'is_fraud': pd.api.types.is_integer_dtype(df['is_fraud']),
            'is_foreign_transaction': pd.api.types.is_integer_dtype(df['is_foreign_transaction']),
            'is_weekend': pd.api.types.is_integer_dtype(df['is_weekend']),
            'is_high_risk_context': pd.api.types.is_integer_dtype(df['is_high_risk_context']),
            'cardholder_age': pd.api.types.is_integer_dtype(df['cardholder_age']),
            'account_age_days': pd.api.types.is_integer_dtype(df['account_age_days']),
            'num_transactions_24h': pd.api.types.is_integer_dtype(df['num_transactions_24h']),
            'num_declined_7d': pd.api.types.is_integer_dtype(df['num_declined_7d']),
        }
        type_pass = all(type_checks.values())
        results.append({
            'name': 'type_standardization_checks',
            'passed': type_pass,
            'details': f"Type checks passed: {type_pass}"
        })
    except Exception as e:
        results.append({
            'name': 'type_standardization_checks',
            'passed': False,
            'details': str(e)
        })

    try:
        source_row_count = len(source_df)
        output_row_count = len(df)
        removed_count = source_row_count - output_row_count
        results.append({
            'name': 'output_schema_rowcount_sanity',
            'passed': output_row_count > 0,
            'details': f"Source row count: {source_row_count}, Output row count: {output_row_count}, Removed count: {removed_count}"
        })
    except Exception as e:
        results.append({
            'name': 'output_schema_rowcount_sanity',
            'passed': False,
            'details': str(e)
        })

    return results
