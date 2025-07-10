import pandas as pd

def load_and_clean_data(file_path):
    df = pd.read_excel(file_path, sheet_name='Data 1', skiprows=2)
    df.columns = ["Date", "Production"]
    df["Date"] = pd.to_datetime(df["Date"])
    df["Production"] = pd.to_numeric(df["Production"], errors='coerce')
    df.dropna(subset=["Production"], inplace=True)
    return df

def feature_engineer(df):
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    # Create lagged features
    for i in range(1, 4):
        df[f"Production_Lag_{i}"] = df["Production"].shift(i)
    df.dropna(inplace=True)
    return df

file_path = 'oil_production_forecasting/MCRFPUS2m.xls'
df = load_and_clean_data(file_path)
df_fe = feature_engineer(df)

df_fe.to_csv('oil_production_forecasting/processed_data.csv', index=False)
print('Processed data saved to oil_production_forecasting/processed_data.csv')
print(df_fe.head())


