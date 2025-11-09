import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
import os

# ------------------ Page Configuration ------------------
st.set_page_config(page_title="Gold Price Predictor", layout="wide")
st.title("💰 Gold Price Prediction using ML & DL (in ₹ INR)")

# ------------------ Sidebar ------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2936/2936757.png", width=100)
    st.markdown("### 📤 Upload Your Data")
    
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
st.sidebar.markdown(
    "[📥 Download Sample CSV](https://raw.githubusercontent.com/datasets/gold-prices/master/data/monthly.csv)"
)

# ------------------ Helper Functions ------------------
def load_data(file):
    """Load CSV, detect INR column, or convert USD to INR automatically."""
    try:
        df = pd.read_csv(file, sep=None, engine="python")
    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")
        st.stop()

    df.columns = df.columns.str.strip()

    # 🧩 If 'Price_INR' already exists, skip conversion
    if 'Price_INR' in df.columns:
        st.info("💡 Detected 'Price_INR' column — skipping USD to INR conversion.")
        return df

    # Otherwise, ensure 'Price' column exists
    if 'Price' not in df.columns:
        st.error("❌ The CSV must have a 'Price' column or 'Price_INR'. Example: Date,Price")
        st.write("Detected columns:", list(df.columns))
        st.stop()

    # ✅ Convert USD/oz → INR/g
    USD_TO_INR = 83.0
    OZ_TO_GRAM = 31.1035
    df['Price_INR'] = df['Price'] * USD_TO_INR / OZ_TO_GRAM

    st.success("✅ Price converted from USD/oz to INR/g automatically.")
    return df


def preprocess_data(df):
    df.fillna(method='ffill', inplace=True)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.set_index('Date', inplace=True)
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    return df, df_scaled, scaler


def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i - n_steps:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ------------------ Main App Logic ------------------
if uploaded_file is not None:
    st.subheader("📂 Uploaded Data Preview")
    df = load_data(uploaded_file)
else:
    st.warning("")
    try:
        df = load_data("data/gold_prices.csv")
    except FileNotFoundError:
        st.error("❌ Default dataset not found! Please upload a CSV manually.")
        st.stop()

# ✅ Show detected columns
st.success("✅ Data successfully loaded and columns verified.")

# ✅ Display dataset
st.subheader("📊 Dataset Preview")
st.dataframe(df.head(25000))

# ✅ Data Preprocessing
df_original, df_scaled, scaler = preprocess_data(df[['Price_INR']])
n_steps = 60
X, y = create_sequences(df_scaled, n_steps)
X = X.reshape((X.shape[0], X.shape[1], 1))

# ✅ Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ✅ Model Selection
st.sidebar.title("Model Selection")
model_type = st.sidebar.selectbox(
    "Choose a model", ["LSTM", "Linear Regression", "Random Forest", "XGBoost"]
)

# ------------------ Model Training ------------------
if model_type == "LSTM":
    st.subheader("Training LSTM Model (Deep Learning)")
    model = build_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    predictions = model.predict(X_test)
else:
    X_flat = X.reshape(X.shape[0], X.shape[1])
    X_train_f, X_test_f = X_flat[:split], X_flat[split:]

    if model_type == "Linear Regression":
        model = LinearRegression()
    elif model_type == "Random Forest":
        model = RandomForestRegressor()
    elif model_type == "XGBoost":
        model = XGBRegressor(objective='reg:squarederror')

    st.subheader(f"🧮 Training {model_type} Model (Machine Learning)")
    model.fit(X_train_f, y_train)
    predictions = model.predict(X_test_f)

# ✅ Reverse scaling
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
pred_inv = scaler.inverse_transform(predictions.reshape(-1, 1))

# ✅ Interactive Visualization (Plotly)
st.subheader("📈 Actual vs Predicted Gold Prices (₹/gram)")

if 'Date' in df.index:
    dates = df.index[-len(y_test_inv):]
else:
    dates = np.arange(len(y_test_inv))

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=dates,
    y=y_test_inv.flatten(),
    mode='lines',
    name='Actual Price (₹)',
    line=dict(color='blue')
))
fig.add_trace(go.Scatter(
    x=dates,
    y=pred_inv.flatten(),
    mode='lines',
    name='Predicted Price (₹)',
    line=dict(color='red')
))

fig.update_layout(
    title="Gold Price Prediction in Indian Rupees (₹/gram)",
    xaxis_title="Date",
    yaxis_title="Price (₹ per gram)",
    hovermode="x unified",
    template="plotly_white",
    legend=dict(x=0, y=1)
)

st.plotly_chart(fig, use_container_width=True)

# ✅ Evaluation Metrics
rmse = np.sqrt(mean_squared_error(y_test_inv, pred_inv))
mae = mean_absolute_error(y_test_inv, pred_inv)

st.metric(label="RMSE (₹)", value=f"{rmse:,.2f}")
st.metric(label="MAE (₹)", value=f"{mae:,.2f}")

st.success("✅ Prediction completed successfully! Prices are displayed in Indian Rupees per gram.")
