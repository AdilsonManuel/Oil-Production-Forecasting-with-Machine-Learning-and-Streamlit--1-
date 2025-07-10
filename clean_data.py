import pandas as pd

def load_and_clean_data(file_path):
    df = pd.read_excel(file_path, sheet_name='Data 1', skiprows=2)
    df.columns = ["Date", "Production"]
    df["Date"] = pd.to_datetime(df["Date"])
    df["Production"] = pd.to_numeric(df["Production"], errors='coerce')
    df.dropna(subset=["Production"], inplace=True)
    return df

file_path = 'oil_production_forecasting/MCRFPUS2m.xls'
df = load_and_clean_data(file_path)

print(df.head())
print(df.info())


