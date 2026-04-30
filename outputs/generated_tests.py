def generate_test_results(source_df, df, parsed_requirements):
    results = []
    
    # 1. Target Schema: expected_columns_present
    expected_columns = [col['name'] for col in parsed_requirements['target_columns']]
    present_columns = df.columns.tolist()
    missing_columns = set(expected_columns) - set(present_columns)
    
    results.append({
        'name': 'expected_columns_present',
        'passed': len(missing_columns) == 0,
        'details': f"Expected target count: {len(expected_columns)}, Present target count: {len(present_columns)}, Missing target columns: {missing_columns}"
    })
    
    # 2. Transformation Requirements: rename_mapping_applied
    rename_mapping = parsed_requirements['rename_rules']
    rename_coverage = {source: target for source, target in rename_mapping.items() if source in source_df.columns}
    rename_passed = all(df[target].equals(source_df[source]) for source, target in rename_coverage.items())
    
    results.append({
        'name': 'rename_mapping_applied',
        'passed': rename_passed,
        'details': f"Rename coverage: {rename_coverage}"
    })
    
    # 3. Timestamp Conversion: timestamp_conversion_logic
    reference_start = pd.Timestamp('2020-01-01 00:00:00')
    expected_timestamps = reference_start + pd.to_timedelta(source_df['Time'], unit='s')
    timestamp_passed = df['transaction_timestamp'].equals(expected_timestamps)
    
    results.append({
        'name': 'timestamp_conversion_logic',
        'passed': timestamp_passed,
        'details': "Timestamp conversion logic validated." if timestamp_passed else "Timestamp conversion logic failed."
    })
    
    # 4. Data Cleaning: merchant_country_standardization
    country_standardization = {
        'USA': 'US', 'us': 'US', 'U.S.': 'US', 'United States': 'US'
    }
    df['merchant_country'] = df['merchant_country'].replace(country_standardization)
    country_passed = df['merchant_country'].isin(country_standardization.values()).all()
    
    results.append({
        'name': 'merchant_country_standardization',
        'passed': country_passed,
        'details': "Merchant country standardized." if country_passed else "Merchant country standardization failed."
    })
    
    # 5. Data Cleaning: merchant_name_blank_handling
    blank_handling_passed = df['merchant_name'].isnull().sum() + (df['merchant_name'] == '').sum() == df['merchant_name'].isnull().sum()
    
    results.append({
        'name': 'merchant_name_blank_handling',
        'passed': blank_handling_passed,
        'details': "Blank merchant names handled correctly." if blank_handling_passed else "Blank merchant names handling failed."
    })
    
    # 6. Type Standardization: type_standardization_checks
    type_checks = {
        'transaction_amount': 'float64',
        'credit_limit': 'float64',
        'account_balance': 'float64',
        'avg_amount_30d': 'float64',
        'is_fraud': 'int64',
        'is_foreign_transaction': 'int64',
        'is_weekend': 'int64',
        'is_high_risk_context': 'int64',
        'cardholder_age': 'int32',
        'account_age_days': 'int32',
        'num_transactions_24h': 'int32',
        'num_declined_7d': 'int32',
        'distance_from_home_km': 'float64',
        'utilization_rate': 'float64'
    }
    
    type_passed = all(df[col].dtype == dtype for col, dtype in type_checks.items() if col in df.columns)
    
    results.append({
        'name': 'type_standardization_checks',
        'passed': type_passed,
        'details': "Type standardization checks passed." if type_passed else "Type standardization checks failed."
    })
    
    # 7. Output Requirements: output_schema_rowcount_sanity
    source_row_count = source_df.shape[0]
    output_row_count = df.shape[0]
    removed_count = source_row_count - output_row_count
    
    results.append({
        'name': 'output_schema_rowcount_sanity',
        'passed': output_row_count > 0,
        'details': f"Source row count: {source_row_count}, Output row count: {output_row_count}, Removed count: {removed_count}"
    })
    
    return results
