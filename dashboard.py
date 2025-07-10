import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

# Set page config
st.set_page_config(page_title="Oil Production Forecasting System", layout="wide")

# Title
st.title("🛢️ Oil Production Forecasting System")
st.markdown("### Machine Learning-based forecasting for U.S. Crude Oil Production")

# Load and cache data
@st.cache_data
def load_data():
    df = pd.read_csv('processed_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Train model and cache
@st.cache_data
def train_model(df):
    X = df[['Year', 'Month', 'Production_Lag_1', 'Production_Lag_2', 'Production_Lag_3']]
    y = df['Production']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    return model, mse, X_test, y_test, predictions

# Load data
df = load_data()

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Overview", "Data Analysis", "Model Training", "Forecasting"])

if page == "Overview":
    st.header("📊 System Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Data Points", len(df))
    
    with col2:
        st.metric("Date Range", f"{df['Date'].min().year} - {df['Date'].max().year}")
    
    with col3:
        st.metric("Latest Production", f"{df['Production'].iloc[-1]:,.0f} thousand barrels/day")
    
    st.subheader("About the System")
    st.write("""
    This Oil Production Forecasting System uses Machine Learning to predict U.S. crude oil production.
    The system includes:
    - Historical data from 1920 to present
    - Multiple ML models (Scikit-learn, TensorFlow)
    - Interactive dashboard for visualization and forecasting
    """)

elif page == "Data Analysis":
    st.header("📈 Data Analysis")
    
    # Time series plot
    st.subheader("Historical Oil Production Trend")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Date'], df['Production'])
    ax.set_title('U.S. Crude Oil Production Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Production (Thousand Barrels per Day)')
    ax.grid(True)
    st.pyplot(fig)
    
    # Statistics
    st.subheader("Production Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Basic Statistics:**")
        st.write(df['Production'].describe())
    
    with col2:
        st.write("**Recent Trends:**")
        recent_data = df.tail(12)
        avg_recent = recent_data['Production'].mean()
        st.metric("Average (Last 12 months)", f"{avg_recent:,.0f}")

elif page == "Model Training":
    st.header("🤖 Model Training & Evaluation")
    
    # Train model
    model, mse, X_test, y_test, predictions = train_model(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Performance")
        st.metric("Mean Squared Error", f"{mse:,.2f}")
        st.metric("RMSE", f"{np.sqrt(mse):,.2f}")
    
    with col2:
        st.subheader("Model Features")
        st.write("- Year")
        st.write("- Month")
        st.write("- Production Lag 1")
        st.write("- Production Lag 2")
        st.write("- Production Lag 3")
    
    # Prediction vs Actual plot
    st.subheader("Predictions vs Actual Values")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, predictions, alpha=0.6)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Production')
    ax.set_ylabel('Predicted Production')
    ax.set_title('Model Predictions vs Actual Values')
    ax.grid(True)
    st.pyplot(fig)

elif page == "Forecasting":
    st.header("🔮 Production Forecasting")
    
    # Train model
    model, mse, _, _, _ = train_model(df)
    
    st.subheader("Forecast Next Months")
    
    # Input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_months = st.slider("Number of months to forecast", 1, 12, 6)
    
    with col2:
        st.write(f"**Model RMSE:** {np.sqrt(mse):,.2f}")
    
    # Generate forecasts
    last_row = df.iloc[-1]
    forecasts = []
    
    for i in range(forecast_months):
        # Create input for next month
        next_date = pd.to_datetime(last_row['Date']) + pd.DateOffset(months=i+1)
        next_year = next_date.year
        next_month = next_date.month
        
        # Use last known production values as lags
        if i == 0:
            lag_1 = last_row['Production']
            lag_2 = last_row['Production_Lag_1']
            lag_3 = last_row['Production_Lag_2']
        else:
            lag_1 = forecasts[i-1] if i > 0 else last_row['Production']
            lag_2 = forecasts[i-2] if i > 1 else last_row['Production_Lag_1']
            lag_3 = forecasts[i-3] if i > 2 else last_row['Production_Lag_2']
        
        # Make prediction
        X_forecast = np.array([[next_year, next_month, lag_1, lag_2, lag_3]])
        forecast = model.predict(X_forecast)[0]
        forecasts.append(forecast)
    
    # Display forecasts
    st.subheader("Forecast Results")
    
    forecast_dates = [pd.to_datetime(last_row['Date']) + pd.DateOffset(months=i+1) for i in range(forecast_months)]
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecasted Production': forecasts
    })
    
    st.dataframe(forecast_df)
    
    # Plot forecasts
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot historical data (last 24 months)
    recent_data = df.tail(24)
    ax.plot(recent_data['Date'], recent_data['Production'], label='Historical', color='blue')
    
    # Plot forecasts
    ax.plot(forecast_dates, forecasts, label='Forecast', color='red', linestyle='--', marker='o')
    
    ax.set_title('Oil Production Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Production (Thousand Barrels per Day)')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("**Oil Production Forecasting System** - Built with Streamlit, Scikit-learn, and TensorFlow")

