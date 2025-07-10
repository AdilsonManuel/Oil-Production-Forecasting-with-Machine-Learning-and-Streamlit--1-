# 🛢️ Oil Production Forecasting System

**Forecasting U.S. Crude Oil Production Using Machine Learning (1920–2025)**  
📈 Powered by Scikit-learn, TensorFlow & Streamlit  
🧠 Developed by Adilson Zumba Manuel

---

## 📌 Overview

The Oil Production Forecasting System is a robust machine learning platform for predicting monthly U.S. crude oil production. Trained on over a century of historical data, it offers high prediction accuracy, interpretable models, and a user-friendly dashboard for exploration and decision-making.

---

## 🔍 Key Features

- ⛽ **Historical Data Coverage**: 1920–2025, 1,260+ monthly observations
- 📊 **Dual Modeling**: Scikit-learn Linear Regression & TensorFlow Neural Network
- 📈 **Forecast Accuracy**: RMSE ~132.50 (≈1% relative error)
- ⚡ **Fast & Responsive**: Real-time forecast generation and interactive visualizations
- 🌐 **Streamlit Dashboard**: Modular pages for exploration, model evaluation, and forecasting

---

## 🧠 Model Performance

| Model                  | RMSE     | MSE         | Notes                                   |
|-----------------------|----------|-------------|-----------------------------------------|
| Linear Regression      | 132.50   | 17,555.68   | Best performance, interpretable output  |
| Neural Network (TF)    | 1033.64  | 1,068,211.75| Captures non-linearity, less efficient  |

---

## 🖥️ Dashboard Interface

Built with **Streamlit**, the dashboard includes:

1. **Overview Page** – Key system metrics and dataset summary  
2. **Data Analysis Page** – Interactive production trends & EDA charts  
3. **Model Training Page** – Compare model performance & validation metrics  
4. **Forecasting Page** – Generate 1–12 month predictions interactively

---

## 📂 Project Structure

oil_production_forecasting/
├── data/
│ ├── MCRFPUS2m.xls # Raw EIA Excel file
│ └── processed_data.csv # Cleaned and engineered dataset
│
├── clean_data.py # Data cleaning logic
├── feature_engineering.py # Lag features and time-based features
├── train_model_sklearn.py # Linear Regression model
├── train_model_tensorflow.py # Neural Network model (TensorFlow)
├── dashboard.py # Streamlit dashboard app
├── production_trend.png # EDA output image
├── requirements.txt # Python dependencies
└── README.md # Documentation


---

## ⚙️ Setup & Installation

### ✅ Prerequisites

- Python 3.11+
- pip
- 4GB RAM (minimum)

### 📦 Install Dependencies

```bash
# Clone the repo
git clone https://github.com/your-username/oil-production-forecasting.git
cd oil-production-forecasting

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate        # For Linux/Mac
# venv\Scripts\activate         # For Windows

# Install required packages
pip install -r requirements.txt

🚀 Run the Dashboard
streamlit run dashboard.py
Then open your browser and go to http://localhost:8501

🐳 Docker Deployment (Optional)

FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]

Build and Run
docker build -t oil-forecasting .
docker run -p 8501:8501 oil-forecasting

🧪 Testing
python -m unittest discover tests

🔬 Technologies Used
Python 3.11
Scikit-learn – Linear Regression
TensorFlow – Deep Learning (Multi-Layer Perceptron)
Pandas & NumPy – Data analysis
Matplotlib – Visualization
Streamlit – Dashboard UI
EIA Data – Official U.S. oil production statistics

🚧 Future Improvements
🔁 Integrate time-series models: LSTM, ARIMA, Prophet
🌍 Incorporate external features: oil prices, rig count, GDP, policy shifts
📉 Add forecast confidence intervals
🧩 Enable CSV/PDF export of results
📈 Use Plotly for interactive charts


📈 Example Results
Best RMSE: 132.50 (Linear Regression)
Prediction Latency: ~10ms per month
Dashboard Response Time: <500ms
Memory Usage: ~100MB in active mode

📜 License
This project is licensed under the MIT License © 2025 Adilson Zumba Manuel

🤝 Contributing
Contributions are welcome!
If you'd like to suggest improvements, submit a pull request or open an issue.
Fork the repo
Create a new branch (git checkout -b feature-xyz)
Commit your changes
Push to your fork
Open a pull request

📬 Contact
Author: Adilson Zumba Manuel
📧 Email: adilsonzumba@hotmail.com
🔗 LinkedIn: (https://www.linkedin.com/in/adilson-manuel-1039181aa/)

Built with ❤️ to forecast the future of energy.









