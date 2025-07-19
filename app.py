import streamlit as st
import yfinance as yf
import pandas as pd
import mplfinance as mpf
import ta
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

st.set_page_config(layout="wide")
st.title("📈 AI-Powered Indian Stock Market Assistant")

def fetch_stock_data(ticker, period='1y', interval='1d'):
    data = yf.download(ticker, period=period, interval=interval)
    data.reset_index(inplace=True)
    return data

def add_indicators(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['macd'] = ta.trend.MACD(df['Close']).macd()
    df['sma_50'] = df['Close'].rolling(window=50).mean()
    df['sma_200'] = df['Close'].rolling(window=200).mean()
    return df

def lstm_prediction(df):
    data = df[['Close']].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(60, len(data_scaled)):
        X.append(data_scaled[i-60:i])
        y.append(data_scaled[i])
    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(50, input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=16, verbose=0)

    last_60 = data_scaled[-60:]
    pred_input = np.expand_dims(last_60, axis=0)
    pred = model.predict(pred_input)
    pred_price = scaler.inverse_transform(pred)
    price = pred_price[0][0]
    return {
        'buy_price': round(price * 0.98, 2),
        'target_price': round(price * 1.10, 2),
        'stop_loss': round(price * 0.95, 2)
    }

def get_moneycontrol_headlines():
    url = "https://www.moneycontrol.com/news/business/markets/"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.content, "html.parser")
    headlines = []
    for item in soup.select("li.clearfix a"):
        title = item.get_text(strip=True)
        link = item['href']
        if title and "video" not in link:
            headlines.append((title, link))
        if len(headlines) >= 5:
            break
    return headlines

ticker = st.text_input("Enter NSE stock ticker (e.g., TCS.NS):", "TCS.NS")
if st.button("Analyze"):
    df = fetch_stock_data(ticker)
    df = add_indicators(df)

    st.subheader("📊 Technical Chart")
    st.line_chart(df[['Close', 'sma_50', 'sma_200']])

    st.subheader("🕯️ Candlestick Chart")
    df_candle = df.set_index('Date')
    mpf_plot = mpf.plot(df_candle, type='candle', style='charles', volume=True, returnfig=True)
    st.pyplot(mpf_plot[0])

    st.subheader("🧠 AI Prediction")
    pred = lstm_prediction(df)
    st.write(f"**Buy at:** ₹{pred['buy_price']}")
    st.write(f"**Target:** ₹{pred['target_price']}")
    st.write(f"**Stop Loss:** ₹{pred['stop_loss']}")

    st.subheader("📰 News (Moneycontrol)")
    for title, link in get_moneycontrol_headlines():
        st.markdown(f"- [{title}]({link})")
