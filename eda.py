import pandas as pd
import matplotlib.pyplot as plt

def load_and_clean_data(file_path):
    df = pd.read_excel(file_path, sheet_name='Data 1', skiprows=2)
    df.columns = ["Date", "Production"]
    df["Date"] = pd.to_datetime(df["Date"])
    df["Production"] = pd.to_numeric(df["Production"], errors='coerce')
    df.dropna(subset=["Production"], inplace=True)
    return df

file_path = 'oil_production_forecasting/MCRFPUS2m.xls'
df = load_and_clean_data(file_path)

plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["Production"])
plt.title('U.S. Crude Oil Production Over Time')
plt.xlabel('Date')
plt.ylabel('Production (Thousand Barrels per Day)')
plt.grid(True)
plt.savefig('oil_production_forecasting/production_trend.png')
print('Plot saved to oil_production_forecasting/production_trend.png')


