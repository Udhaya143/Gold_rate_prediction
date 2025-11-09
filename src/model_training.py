# model_training.py

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ---------------- Linear Regression ----------------
def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# ---------------- XGBoost ----------------
def train_xgboost(X_train, y_train):
    """
    Train an XGBoost Regressor model
    """
    model = XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train)
    return model


# ---------------- LSTM (Long Short-Term Memory) ----------------
def train_lstm(X_train, y_train):
    """
    Train an LSTM model for time series forecasting
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    return model
