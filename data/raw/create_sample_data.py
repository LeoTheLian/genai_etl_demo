import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

n_records = 1000

# ----- Reference data pools -----
states = ['CA', 'TX', 'NY', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI',
          'NJ', 'VA', 'WA', 'AZ', 'MA', 'TN', 'IN', 'MO', 'MD', 'CO']

merchant_categories = [
    'grocery', 'gas_station', 'restaurant', 'online_retail', 'travel',
    'healthcare', 'entertainment', 'hotel', 'pharmacy', 'utilities'
]

merchant_names_by_category = {
    'grocery':       ['Whole Foods Market', 'Kroger', 'Safeway', 'Trader Joes', 'Publix'],
    'gas_station':   ['Shell', 'ExxonMobil', 'BP', 'Chevron', 'Valero'],
    'restaurant':    ['McDonalds', 'Starbucks', 'Chipotle', 'Olive Garden', 'Subway'],
    'online_retail': ['Amazon', 'eBay', 'Walmart.com', 'Target.com', 'Best Buy Online'],
    'travel':        ['Delta Airlines', 'United Airlines', 'Expedia', 'Airbnb', 'Hertz'],
    'healthcare':    ['CVS Health', 'Walgreens', 'Quest Diagnostics', 'LabCorp', 'UnitedHealth'],
    'entertainment': ['Netflix', 'AMC Theatres', 'Spotify', 'Ticketmaster', 'Steam'],
    'hotel':         ['Marriott', 'Hilton', 'Hyatt', 'Holiday Inn', 'Best Western'],
    'pharmacy':      ['Rite Aid', 'CVS Pharmacy', 'Walgreens Pharmacy', 'Duane Reade', 'RxCrossroads'],
    'utilities':     ['AT&T', 'Verizon', 'Comcast', 'Duke Energy', 'Pacific Gas and Electric'],
}

channels = ['in-store', 'online', 'atm']
pos_modes = ['chip', 'swipe', 'contactless', 'online', 'manual']

# ----- Core transaction fields -----
transaction_ids = [f'TXN_{i:06d}' for i in range(n_records)]
time_seconds = np.random.randint(0, 172800, n_records)       # seconds elapsed over 2-day window
amounts = np.random.gamma(2, 2, n_records) * 100              # realistic spend distribution
fraud_labels = np.random.choice([0, 1], n_records, p=[0.95, 0.05])

# ----- Account / cardholder fields -----
account_ids = [f'ACC_{np.random.randint(100000, 999999)}' for _ in range(n_records)]
account_age_days = np.random.randint(30, 5000, n_records)     # days since account opened
credit_limits = np.random.choice([1000, 2000, 5000, 10000, 15000, 25000, 50000], n_records)
account_balances = credit_limits * np.random.uniform(0.0, 0.95, n_records)
cardholder_ages = np.random.randint(21, 75, n_records)
cardholder_states = np.random.choice(states, n_records)
cardholder_cities = [f'City_{s}_{np.random.randint(1,10)}' for s in cardholder_states]
cardholder_zips = [f'{np.random.randint(10000, 99999):05d}' for _ in range(n_records)]

# ----- Merchant fields -----
merchant_cats = np.random.choice(merchant_categories, n_records)
merchant_names = [
    np.random.choice(merchant_names_by_category[cat]) for cat in merchant_cats
]
merchant_ids = [f'MER_{np.random.randint(10000, 99999)}' for _ in range(n_records)]
merchant_states = np.random.choice(states, n_records)
merchant_cities = [f'City_{s}_{np.random.randint(1,10)}' for s in merchant_states]

# Mostly domestic, a few foreign — intentionally inconsistent country formatting
# to demonstrate standardization
country_pool_domestic = ['US', 'USA', 'us', 'U.S.', 'United States']  # messy, all mean USA
country_pool_foreign  = ['CA', 'MX', 'GB', 'DE', 'FR']
merchant_countries = np.where(
    np.random.rand(n_records) < 0.07,
    np.random.choice(country_pool_foreign, n_records),
    np.random.choice(country_pool_domestic, n_records)
)

# ----- Transaction context fields -----
txn_channels = np.random.choice(channels, n_records, p=[0.55, 0.40, 0.05])
pos_entry_modes = np.array([
    np.random.choice(['chip', 'swipe', 'contactless']) if ch == 'in-store'
    else ('online' if ch == 'online' else 'manual')
    for ch in txn_channels
])
is_foreign = (np.isin(merchant_countries, country_pool_foreign)).astype(int)
distance_from_home_km = np.random.exponential(25, n_records)  # most transactions near home
num_transactions_24h = np.random.poisson(3, n_records)
num_declined_7d = np.random.poisson(0.3, n_records)
avg_amount_30d = np.random.gamma(2, 1.5, n_records) * 80

# ----- Introduce data quality issues -----

# 1. Negative amounts (invalid — must be cleaned)
negative_idx = np.random.choice(n_records, 20, replace=False)
amounts[negative_idx] *= -1

# 2. Extreme amount outliers (flag for review)
outlier_idx = np.random.choice(n_records, 30, replace=False)
amounts[outlier_idx] = np.random.uniform(10000, 50000, 30)

# 3. NULL account_balance (missing field — must be handled)
null_balance_idx = np.random.choice(n_records, 15, replace=False)
account_balances = account_balances.astype(object)
account_balances[null_balance_idx] = np.nan

# 4. Invalid cardholder_age values (out-of-range — must be cleaned)
invalid_age_idx = np.random.choice(n_records, 10, replace=False)
cardholder_ages = cardholder_ages.astype(object)
cardholder_ages[invalid_age_idx] = np.random.choice([-5, 0, 120, 150, 999], 10)

# 5. Missing merchant_name (blank string — must be handled)
missing_merchant_idx = np.random.choice(n_records, 10, replace=False)
merchant_names = list(merchant_names)
for idx in missing_merchant_idx:
    merchant_names[idx] = ''

# 6. Negative distance_from_home_km (invalid — must be cleaned)
neg_dist_idx = np.random.choice(n_records, 8, replace=False)
distance_from_home_km[neg_dist_idx] *= -1

# ----- Assemble DataFrame -----
df = pd.DataFrame({
    # Core
    'transaction_id':         transaction_ids,
    'Time':                   time_seconds,
    'Amount':                 amounts,
    'Class':                  fraud_labels,
    # Account / cardholder
    'account_id':             account_ids,
    'account_age_days':       account_age_days,
    'credit_limit':           credit_limits,
    'account_balance':        account_balances,
    'cardholder_age':         cardholder_ages,
    'cardholder_city':        cardholder_cities,
    'cardholder_state':       cardholder_states,
    'cardholder_zip':         cardholder_zips,
    # Merchant
    'merchant_id':            merchant_ids,
    'merchant_name':          merchant_names,
    'merchant_category':      merchant_cats,
    'merchant_city':          merchant_cities,
    'merchant_state':         merchant_states,
    'merchant_country':       merchant_countries,
    # Transaction context
    'channel':                txn_channels,
    'pos_entry_mode':         pos_entry_modes,
    'is_foreign_transaction': is_foreign,
    'distance_from_home_km':  distance_from_home_km,
    'num_transactions_24h':   num_transactions_24h,
    'num_declined_7d':        num_declined_7d,
    'avg_amount_30d':         avg_amount_30d,
})

df.to_csv('data/raw/sample_transactions.csv', index=False)

# ----- Summary -----
null_counts = df.isnull().sum()
print(f'✓ Created sample dataset with {n_records} transactions')
print(f'✓ Saved to: data/raw/sample_transactions.csv')
print(f'\nColumns ({len(df.columns)}): {", ".join(df.columns)}')
print(f'\nData quality issues introduced:')
print(f'  - {len(negative_idx)} records with negative Amount')
print(f'  - {len(outlier_idx)} records with extreme Amount outliers ($10K-$50K)')
print(f'  - {len(null_balance_idx)} records with NULL account_balance')
print(f'  - {len(invalid_age_idx)} records with invalid cardholder_age (out-of-range)')
print(f'  - {len(missing_merchant_idx)} records with blank merchant_name')
print(f'  - {len(neg_dist_idx)} records with negative distance_from_home_km')
print(f'  - {len(np.unique(country_pool_domestic))} inconsistent formats for domestic country code')
print(f'\nFraud distribution: {(df["Class"]==0).sum()} legitimate, {(df["Class"]==1).sum()} fraudulent')
print(f'Dataset shape: {df.shape}')
