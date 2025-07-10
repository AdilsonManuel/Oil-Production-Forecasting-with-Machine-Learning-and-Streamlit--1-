import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def load_processed_data(file_path):
    df = pd.read_csv(file_path)
    return df

def train_and_evaluate_tf_model(df):
    X = df[["Year", "Month", "Production_Lag_1", "Production_Lag_2", "Production_Lag_3"]]
    y = df["Production"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Dense(64, activation="relu", input_shape=(X_train.shape[1],)))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    print(f"TensorFlow Model Mean Squared Error: {mse}")
    return model, mse

file_path = "oil_production_forecasting/processed_data.csv"
df_processed = load_processed_data(file_path)
tf_model, tf_mse = train_and_evaluate_tf_model(df_processed)


