import pandas as pd

def load_data(file_path):
    df = pd.read_excel(file_path, sheet_name='Data 1')
    return df

file_path = 'oil_production_forecasting/MCRFPUS2m.xls'
df = load_data(file_path)

# Inspect the first few rows and column names
print(df.head(10))
print(df.columns)


