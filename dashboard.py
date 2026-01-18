
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from models import load_processed_data, train_linear_regression, train_neural_network
import pickle

# Set page config
st.set_page_config(page_title="Oil Production Forecasting System", layout="wide")

# Title
st.title("🛢️ Oil Production Forecasting System")
st.markdown("### Machine Learning-based forecasting for U.S. Crude Oil Production")

# Load and cache data
@st.cache_data
def get_data():
    return load_processed_data('processed_data.csv')

# Load data
try:
    df = get_data()
except FileNotFoundError:
    st.error("Data file 'processed_data.csv' not found. Please ensure data processing has been run.")
    st.stop()

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Overview", "Data Analysis", "Model Training", "Forecasting"])

st.sidebar.header("Model Configuration")
try:
    from models import TF_AVAILABLE
except ImportError:
    TF_AVAILABLE = False

if TF_AVAILABLE:
    model_options = ["Linear Regression", "Neural Network (TensorFlow)"]
else:
    model_options = ["Linear Regression"]
    st.sidebar.warning("TensorFlow not installed. Neural Network option disabled.")

model_type = st.sidebar.selectbox("Select Model", model_options)

# Hyperparameters
epochs = 50
learning_rate = 0.001

if model_type == "Neural Network (TensorFlow)":
    st.sidebar.subheader("Hyperparameters")
    epochs = st.sidebar.slider("Epochs", 10, 200, 50)
    learning_rate = st.sidebar.select_slider("Learning Rate", options=[0.001, 0.005, 0.01, 0.05, 0.1], value=0.001)

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
    st.write(f"""
    This Oil Production Forecasting System uses Machine Learning to predict U.S. crude oil production.
    
    **Current Configuration:**
    - **Model:** {model_type}
    - **Data Source:** Processed Historical Data (1920-Present)
    
    The system includes:
    - Historical data analysis
    - Interactive visualizations
    - {model_type} for forecasting
    """)

elif page == "Data Analysis":
    st.header("📈 Data Analysis")
    
    # Time series plot with Plotly
    st.subheader("Historical Oil Production Trend")
    
    fig = px.line(df, x='Date', y='Production', title='U.S. Crude Oil Production Over Time')
    fig.update_layout(xaxis_title='Date', yaxis_title='Production (Thousand Barrels per Day)')
    st.plotly_chart(fig, use_container_width=True)
    
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
        
        # Monthly Seasonality
        st.write("**Monthly Average Production:**")
        monthly_avg = df.groupby('Month')['Production'].mean().reset_index()
        fig_month = px.bar(monthly_avg, x='Month', y='Production', title='Average Production by Month')
        st.plotly_chart(fig_month, use_container_width=True)

elif page == "Model Training":
    st.header("🤖 Model Training & Evaluation")
    
    with st.spinner(f"Training {model_type}..."):
        if model_type == "Linear Regression":
            model, metrics, X_test, y_test, predictions, _ = train_linear_regression(df)
        else:
            # Neural Network
            model, metrics, X_test, y_test, predictions, _ = train_neural_network(df, epochs=epochs, learning_rate=learning_rate)
            # Ensure y_test is 1D for plotting
            if len(predictions.shape) > 1:
                predictions = predictions.flatten()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Model Performance")
        st.metric("RMSE", f"{np.sqrt(metrics['MSE']):,.2f}")
    
    with col2:
        st.subheader("Error Metrics")
        st.metric("MAE", f"{metrics['MAE']:,.2f}")
        st.metric("MSE", f"{metrics['MSE']:,.2f}")

    with col3:
        st.subheader("Goodness of Fit")
        st.metric("R² Score", f"{metrics['R2']:.4f}")
    
    st.write("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Features")
        st.write("- Year")
        st.write("- Month")
        st.write("- Production Lag 1")
        st.write("- Production Lag 2")
        st.write("- Production Lag 3")
    
    # Prediction vs Actual plot with Plotly
    st.subheader("Predictions vs Actual Values")
    
    # Create DataFrame for plotting
    # y_test might be a Series or array, ensure consistency
    if hasattr(y_test, 'values'):
        y_test_values = y_test.values
    else:
        y_test_values = y_test
        
    eval_df = pd.DataFrame({
        'Actual': y_test_values,
        'Predicted': predictions
    })
    
    fig = px.scatter(eval_df, x='Actual', y='Predicted', title='Model Predictions vs Actual Values', opacity=0.6)
    
    # Add perfect prediction line
    min_val = min(y_test_values.min(), predictions.min())
    max_val = max(y_test_values.max(), predictions.max())
    fig.add_shape(
        type="line",
        x0=min_val, y0=min_val, x1=max_val, y1=max_val,
        line=dict(color="Red", dash="dash")
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif page == "Forecasting":
    st.header("🔮 Production Forecasting")
    
    # Train model (or load if we were to implement persistence)
    # Re-training for now to ensure we have the latest selected model
    with st.spinner(f"Training {model_type} for forecasting..."):
        if model_type == "Linear Regression":
            model, metrics, _, _, _, _ = train_linear_regression(df)
            scaler = None
        else:
            model, metrics, _, _, _, scaler = train_neural_network(df, epochs=epochs, learning_rate=learning_rate)
    
    st.subheader("Forecast Next Months")
    
    # Input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_months = st.slider("Number of months to forecast", 1, 12, 6)
    
    with col2:
        st.write(f"**Model Used:** {model_type}")
        st.write(f"**Model RMSE:** {np.sqrt(metrics['MSE']):,.2f}")
    
    # Generate forecasts
    last_row = df.iloc[-1]
    forecasts = []
    
    # Initial lags
    lag_1 = last_row['Production']
    lag_2 = last_row['Production_Lag_1']
    lag_3 = last_row['Production_Lag_2']
    
    for i in range(forecast_months):
        # Create input for next month
        next_date = pd.to_datetime(last_row['Date']) + pd.DateOffset(months=i+1)
        next_year = next_date.year
        next_month = next_date.month
        
        # Prepare input vector [Year, Month, Lag1, Lag2, Lag3]
        input_features = np.array([[next_year, next_month, lag_1, lag_2, lag_3]])
        
        # Scale if necessary (Neural Network)
        if scaler:
            input_features = scaler.transform(input_features)
            
        # Make prediction
        forecast = model.predict(input_features)
        
        # Handle shape differences between sklearn and tf
        if isinstance(forecast, np.ndarray):
            forecast_val = forecast.flatten()[0]
        else:
            forecast_val = forecast
            
        forecasts.append(forecast_val)
        
        # Update lags for next iteration
        lag_3 = lag_2
        lag_2 = lag_1
        lag_1 = forecast_val
    
    # Display forecasts
    st.subheader("Forecast Results")
    
    forecast_dates = [pd.to_datetime(last_row['Date']) + pd.DateOffset(months=i+1) for i in range(forecast_months)]
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecasted Production': forecasts
    })
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.dataframe(forecast_df)
    with col2:
        # Export Button
        csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='oil_production_forecast.csv',
            mime='text/csv',
        )
    
    # Plot forecasts with Plotly
    fig = go.Figure()
    
    # Plot historical data (last 24 months)
    recent_data = df.tail(24)
    fig.add_trace(go.Scatter(x=recent_data['Date'], y=recent_data['Production'], mode='lines', name='Historical'))
    
    # Plot forecasts
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecasts, mode='lines+markers', name='Forecast', line=dict(color='red', dash='dash')))
    
    fig.update_layout(title='Oil Production Forecast', xaxis_title='Date', yaxis_title='Production (Thousand Barrels per Day)')
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(f"**Oil Production Forecasting System** - Powered by {model_type}")
