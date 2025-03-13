import pandas as pd

# Check credit data
df_credit = pd.read_csv('api/data/featured_credit_data.csv')
print('Credit columns:', df_credit.columns.tolist())

# Check fraud data
df_fraud = pd.read_csv('api/data/featured_fraud_data.csv')
print('Fraud columns:', df_fraud.columns.tolist())