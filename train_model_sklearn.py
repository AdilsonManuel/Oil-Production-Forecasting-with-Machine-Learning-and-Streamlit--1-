import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def load_processed_data(file_path):
    df = pd.read_csv(file_path)
    return df

def train_and_evaluate_model(df):
    X = df[['Year', 'Month', 'Production_Lag_1', 'Production_Lag_2', 'Production_Lag_3']]
    y = df['Production']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    print(f'Mean Squared Error: {mse}')
    return model, mse

file_path = 'oil_production_forecasting/processed_data.csv'
df_processed = load_processed_data(file_path)
model, mse = train_and_evaluate_model(df_processed)


