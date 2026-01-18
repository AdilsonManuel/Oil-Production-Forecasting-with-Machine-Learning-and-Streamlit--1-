
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    TF_AVAILABLE = True
except (ImportError, OSError):
    TF_AVAILABLE = False
    print("TensorFlow not available. Neural Network training will be disabled.")

def load_processed_data(file_path):
    """Loads the processed data from CSV."""
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def feature_engineer(df):
    """Ensures features are prepared (already done in processed_data, but kept for robustness)."""
    # This might be redundant if loading processed_data.csv, but good for safety
    if 'Date' in df.columns and 'Year' not in df.columns:
         df["Year"] = df["Date"].dt.year
         df["Month"] = df["Date"].dt.month
    return df

def train_linear_regression(df):
    """
    Trains a Linear Regression model.
    Returns: model, metrics (dict), X_test, y_test, predictions, scaler (None for LR)
    """
    X = df[['Year', 'Month', 'Production_Lag_1', 'Production_Lag_2', 'Production_Lag_3']]
    y = df['Production']
    
    # Linear Regression doesn't strictly need scaling, but we keep the signature consistent
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    metrics = {"MSE": mse, "MAE": mae, "R2": r2}
    
    return model, metrics, X_test, y_test, predictions, None

def train_neural_network(df, epochs=50, learning_rate=0.001):
    """
    Trains a TensorFlow Neural Network model.
    Returns: model, metrics (dict), X_test_scaled, y_test, predictions, scaler
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is not installed or failed to load.")

    X = df[["Year", "Month", "Production_Lag_1", "Production_Lag_2", "Production_Lag_3"]]
    y = df["Production"]

    # Scaling is important for Neural Networks
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Dense(64, activation="relu", input_shape=(X_train.shape[1],)))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse")
    
    # Silent training
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    metrics = {"MSE": mse, "MAE": mae, "R2": r2}

    return model, metrics, X_test, y_test, predictions, scaler
