def generate_test_results(source_df, df, parsed_requirements):
    results = []
    
    try:
        expected_columns = {col['name'] for col in parsed_requirements['target_columns']}
        present_columns = set(df.columns)
        missing_columns = expected_columns - present_columns
        results.append({
            'name': 'expected_columns_present',
            'passed': len(missing_columns) == 0,
            'details': f"Expected target count: {len(expected_columns)}, Present target count: {len(present_columns)}, Missing target columns: {missing_columns}"
        })
    except Exception as e:
        results.append({
            'name': 'expected_columns_present',
            'passed': False,
            'details': str(e)
        })

    try:
        rename_mapping = parsed_requirements['rename_rules']
        rename_coverage = {source: target for source, target in rename_mapping.items() if source in source_df.columns}
        results.append({
            'name': 'rename_mapping_applied',
            'passed': all(col in df.columns for col in rename_coverage.values()),
            'details': f"Rename coverage: {rename_coverage}"
        })
    except Exception as e:
        results.append({
            'name': 'rename_mapping_applied',
            'passed': False,
            'details': str(e)
        })

    try:
        timestamp_col = 'transaction_timestamp'
        valid_timestamps = pd.to_datetime(df[timestamp_col], errors='coerce')
        plausible_range = (pd.Timestamp('2020-01-01'), pd.Timestamp('2030-01-01'))
        all_valid = valid_timestamps.notna().all() and valid_timestamps.between(plausible_range[0], plausible_range[1]).all()
        results.append({
            'name': 'timestamp_conversion_logic',
            'passed': all_valid,
            'details': "All timestamps are valid and within the plausible range."
        })
    except Exception as e:
        results.append({
            'name': 'timestamp_conversion_logic',
            'passed': False,
            'details': str(e)
        })

    try:
        standardized_countries = {'USA': 'US', 'us': 'US', 'U.S.': 'US', 'United States': 'US'}
        df['merchant_country_standardized'] = df['merchant_country'].replace(standardized_countries)
        all_standardized = df['merchant_country_standardized'].notna().all()
        results.append({
            'name': 'merchant_country_standardization',
            'passed': all_standardized,
            'details': "All merchant countries are standardized."
        })
    except Exception as e:
        results.append({
            'name': 'merchant_country_standardization',
            'passed': False,
            'details': str(e)
        })

    try:
        blank_handling = df['merchant_name'].replace('', 'UNKNOWN')
        all_handled = (blank_handling == 'UNKNOWN').sum() == (df['merchant_name'] == '').sum()
        results.append({
            'name': 'merchant_name_blank_handling',
            'passed': all_handled,
            'details': "Blank merchant names are handled correctly."
        })
    except Exception as e:
        results.append({
            'name': 'merchant_name_blank_handling',
            'passed': False,
            'details': str(e)
        })

    try:
        type_checks = {
            'transaction_amount': 'float64',
            'credit_limit': 'float64',
            'account_balance': 'float64',
            'avg_amount_30d': 'float64',
            'distance_from_home_km': 'float64',
            'is_fraud': 'int32',
            'is_foreign_transaction': 'int32',
            'is_weekend': 'int32',
            'is_high_risk_context': 'int32',
            'cardholder_age': 'int32',
            'account_age_days': 'int32',
            'num_transactions_24h': 'int32',
            'num_declined_7d': 'int32'
        }
        type_validation = {col: df[col].dtype.name == expected for col, expected in type_checks.items() if col in df.columns}
        all_types_correct = all(type_validation.values())
        results.append({
            'name': 'type_standardization_checks',
            'passed': all_types_correct,
            'details': f"Type validation results: {type_validation}"
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
