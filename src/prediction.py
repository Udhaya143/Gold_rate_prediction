# prediction.py - Modular code for Gold Price Prediction Models

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# -------------------- Data Loader Function --------------------
def load_and_prepare_data(filepath):
    """
    Loads CSV file and prepares the data by filling missing values and setting Date as index.
    """
    df = pd.read_csv(filepath)
    df.fillna(method='ffill', inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df[['Price']]

# -------------------- Scaling Function --------------------
def scale_data(data):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler

# -------------------- Create time-series sequences --------------------
def create_sequences(data, time_step):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i - time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# -------------------- Train-test split --------------------
def split_data(X, y, train_ratio=0.8):
    split = int(len(X) * train_ratio)
    return X[:split], X[split:], y[:split], y[split:]

# -------------------- Linear Regression Model --------------------
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# -------------------- Random Forest Model --------------------
def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    return model

# -------------------- XGBoost Model --------------------
def train_xgboost(X_train, y_train):
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)
    return model

# -------------------- LSTM Model --------------------
def build_and_train_lstm(X_train, y_train, input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(64))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    return model

# -------------------- Inverse transform predictions --------------------
def inverse_transform(scaler, data):
    return scaler.inverse_transform(data.reshape(-1, 1))

# -------------------- Evaluate performance --------------------
def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

# -------------------- Visualize actual vs predicted --------------------
def plot_predictions(y_true, y_pred, model_name="Model"):
    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label="Actual", linewidth=2)
    plt.plot(y_pred, label="Predicted", linewidth=2)
    plt.title(f"{model_name} - Actual vs Predicted")
    plt.xlabel("Time")
    plt.ylabel("Gold Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -------------------- Main Prediction Function --------------------
def run_prediction_pipeline(filepath, model_type="LSTM"):
    df = load_and_prepare_data(filepath)
    scaled_data, scaler = scale_data(df)
    time_step = 60
    X, y = create_sequences(scaled_data, time_step)

    if model_type == "LSTM":
        X = X.reshape((X.shape[0], X.shape[1], 1))
    else:
        X = X.reshape(X.shape[0], X.shape[1])

    X_train, X_test, y_train, y_test = split_data(X, y)

    if model_type == "Linear Regression":
        model = train_linear_regression(X_train, y_train)
        predictions = model.predict(X_test)
    elif model_type == "Random Forest":
        model = train_random_forest(X_train, y_train)
        predictions = model.predict(X_test)
    elif model_type == "XGBoost":
        model = train_xgboost(X_train, y_train)
        predictions = model.predict(X_test)
    elif model_type == "LSTM":
        model = build_and_train_lstm(X_train, y_train, (X_train.shape[1], 1))
        predictions = model.predict(X_test)
        predictions = predictions.reshape(-1)

    # Inverse transform results
    y_test_inv = inverse_transform(scaler, y_test)
    pred_inv = inverse_transform(scaler, predictions)

    # Evaluate
    rmse, mae, r2 = evaluate_model(y_test_inv, pred_inv)
    print(f"Model: {model_type}")
    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2 Score: {r2:.2f}")

    # Plot
    plot_predictions(y_test_inv, pred_inv, model_type)


# -------------------- Run directly for testing --------------------
if __name__ == "__main__":
    run_prediction_pipeline("data/gold_prices.csv", model_type="LSTM")
