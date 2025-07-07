import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
from datetime import datetime, timedelta

from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense

st.set_page_config(layout="wide", page_title="Stock Dashboard")

TICKERS = ["NVDA", "TSM"]
DAYS = 100
END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=150)

@st.cache_data
def fetch_data(ticker):
    df = yf.download(ticker, start=START_DATE, end=END_DATE)
    df = df.tail(DAYS).copy()

    required_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"❌ '{col}' column is missing in data for {ticker}")
            return pd.DataFrame()

    for col in required_cols:
        if col not in df.columns:
            st.error(f"❌ '{col}' column is missing in data for {ticker}")
            return pd.DataFrame()
        series = df[col]
        if isinstance(series, pd.DataFrame):  # someone did df[['Close']] by mistake
            series = series.squeeze()
        if not isinstance(series, pd.Series):
            st.error(f"❌ '{col}' is not a 1D Series in {ticker} data")
            return pd.DataFrame()
        df[col] = pd.to_numeric(series, errors='coerce')


    close_series = df['Close']
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.squeeze()

    rsi = ta.momentum.RSIIndicator(close_series, window=14)
    df['RSI'] = rsi.rsi()


    df['SMA_20'] = close_series.rolling(20).mean()
    df['SMA_50'] = close_series.rolling(50).mean()

    macd = ta.trend.MACD(close_series)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(close_series, window=20)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()

    df.dropna(inplace=True)
    return df

def plot_chart(df, ticker):
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f"{ticker} - Last {DAYS} Trading Days", fontsize=16)

    axes[0].plot(df.index, df['Close'], label='Close', color='black')
    axes[0].plot(df['SMA_20'], label='SMA 20', color='blue')
    axes[0].plot(df['SMA_50'], label='SMA 50', color='purple')
    axes[0].fill_between(df.index, df['BB_Upper'], df['BB_Lower'], color='gray', alpha=0.2)
    axes[0].legend()
    axes[0].set_ylabel("Price")

    axes[1].plot(df['RSI'], label='RSI', color='orange')
    axes[1].axhline(70, linestyle='--', color='red')
    axes[1].axhline(30, linestyle='--', color='green')
    axes[1].legend()
    axes[1].set_ylabel("RSI")

    axes[2].plot(df['MACD'], label='MACD', color='red')
    axes[2].plot(df['MACD_Signal'], label='Signal', color='green')
    axes[2].bar(df.index, df['MACD_Hist'], label='Histogram', color='gray', alpha=0.3)
    axes[2].legend()
    axes[2].set_ylabel("MACD")
    volume_data = df['Volume']
    if isinstance(volume_data, pd.DataFrame):
        volume_data = volume_data.squeeze()

    volume_data = pd.to_numeric(volume_data, errors='coerce')

    axes[3].bar(df.index, volume_data, label='Volume', color='blue', alpha=0.4)
    axes[3].legend()
    axes[3].set_ylabel("Volume")

    st.pyplot(fig)
    plt.close()

def predict_with_xgboost(df):
    features = ["SMA_20", "SMA_50", "MACD", "RSI", "Volume"]
    df = df[features + ["Close"]].dropna()

    X, y = df[features], df["Close"]
    model = XGBRegressor(n_estimators=100)
    model.fit(X, y)

    preds = []
    future_input = X.iloc[-1]

    for _ in range(4):
        pred = model.predict([future_input])[0]
        preds.append(pred)
        future_input["Volume"] *= 1 + np.random.normal(0, 0.01)

    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=5), periods=4, freq='W')
    return pd.DataFrame({"XGBoost": preds}, index=future_dates)

def predict_with_lstm(df):
    df = df[['Close']].copy()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(20, len(scaled)):
        X.append(scaled[i-20:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=20, verbose=0)

    future_input = scaled[-20:].reshape(1, 20, 1)
    preds = []
    for _ in range(4):
        pred = model.predict(future_input)[0][0]
        preds.append(scaler.inverse_transform([[pred]])[0][0])
        future_input = np.append(future_input[:, 1:, :], [[[pred]]], axis=1)

    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=5), periods=4, freq='W')
    return pd.DataFrame({"LSTM": preds}, index=future_dates)

# --- Streamlit App ---
st.title("📈 NVDA & TSM Stock Technicals + ML Predictions")

for ticker in TICKERS:
    st.subheader(f"📊 {ticker}")
    with st.spinner(f"Fetching & analyzing {ticker}..."):
        df = fetch_data(ticker)
        if df.empty:
            continue
        plot_chart(df, ticker)
        st.dataframe(df.tail(5).style.format("{:.2f}"))

        st.markdown("#### 🔮 Predictions for Next 4 Weeks")
        xgb_df = predict_with_xgboost(df)
        lstm_df = predict_with_lstm(df)
        forecast = pd.concat([xgb_df, lstm_df], axis=1)

        st.line_chart(forecast)
        st.dataframe(forecast.style.format("{:.2f}"))

st.info("This dashboard is for educational purposes only.")