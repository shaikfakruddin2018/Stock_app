import streamlit as st
import pandas as pd
import joblib
import os
import datetime
import plotly.graph_objects as go
import plotly.express as px
import requests\import yfinance as yf
import ta  # for RSI, MACD, Bollinger Bands

# ✅ MODEL DOWNLOAD FROM HUGGING FACE
MODEL_PATH = "rf_model.joblib.joblib"
MODEL_URL = "https://huggingface.co/shaikfakruddin18/stock-predictor-model/resolve/main/rf_model.joblib.joblib"

if not os.path.exists(MODEL_PATH):
    st.write("📥 Downloading model from Hugging Face Hub...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    st.write(f"✅ Model downloaded successfully! Size: {len(response.content)} bytes")

# ✅ Load model
model = joblib.load(MODEL_PATH)

# ✅ PAGE CONFIG
st.set_page_config(page_title="AI Stock Predictor Dashboard", page_icon="📈", layout="wide")

# ✅ FUNCTIONS
def fetch_live_data(ticker, period="6mo"):
    df = yf.download(ticker, period=period, interval="1d")
    df.reset_index(inplace=True)
    return df

def add_technical_indicators(df):
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()
    return df

def plot_candlestick(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"))
    fig.update_layout(height=500, template="plotly_dark")
    return fig

def plot_rsi_macd(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], name="RSI"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], name="MACD"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD_Signal"], name="MACD Signal"))
    fig.update_layout(height=300, template="plotly_dark")
    return fig

def save_prediction(stock, prediction, confidence):
    history_file = "prediction_history.csv"
    entry = pd.DataFrame([[datetime.datetime.now(), stock, prediction, confidence]], columns=["Time", "Stock", "Prediction", "Confidence"])
    if os.path.exists(history_file):
        entry.to_csv(history_file, mode='a', header=False, index=False)
    else:
        entry.to_csv(history_file, index=False)

def load_prediction_history():
    if os.path.exists("prediction_history.csv"):
        return pd.read_csv("prediction_history.csv")
    return pd.DataFrame(columns=["Time", "Stock", "Prediction", "Confidence"])

# ✅ SIDEBAR
st.sidebar.title("📊 Navigation")
menu = st.sidebar.radio("Go to:", ["📈 Stock Predictor", "🧠 Focus Tasks"])

if menu == "🧠 Focus Tasks":
    st.title("🧠 Focus Task Assistant")
    task = st.text_input("✍️ Add a task")
    if st.button("➕ Add"):
        st.session_state.setdefault("tasks", []).append({"task": task, "added": str(datetime.datetime.now())})
        st.success("✅ Task added!")

    st.subheader("📋 Your Tasks")
    for t in st.session_state.get("tasks", []):
        st.markdown(f"✅ {t['task']} *(added {t['added'].split('.')[0]})*")

    if st.button("🎯 Start Focus Timer"):
        st.success("⏱️ Focus mode started for 25 mins!")

else:
    st.title("📈 AI Stock Predictor Dashboard")

    # ✅ Choose Data Source
    source = st.sidebar.radio("📂 Select Data Source", ["Local CSV", "Live Yahoo Finance"])

    if source == "Live Yahoo Finance":
        ticker = st.sidebar.text_input("Enter Stock Ticker (e.g. AAPL, TSLA)", "AAPL")
        period = st.sidebar.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y"])
        df = fetch_live_data(ticker, period)
    else:
        data_dir = "stock_data_with_indicators"
        files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
        selected_file = st.sidebar.selectbox("Select Stock CSV", files)
        df = pd.read_csv(os.path.join(data_dir, selected_file))

    if not df.empty:
        df = add_technical_indicators(df)

        # ✅ Tabs for multiple timeframes
        tab1, tab2 = st.tabs(["📊 Price Chart", "📉 Indicators"])
        with tab1:
            st.plotly_chart(plot_candlestick(df), use_container_width=True)
        with tab2:
            st.plotly_chart(plot_rsi_macd(df), use_container_width=True)

        # ✅ Prediction Section
        latest_features = df[["MACD", "MACD_Signal", "BB_High", "BB_Low", "Close"]].iloc[[-1]]
        pred = model.predict(latest_features)[0]
        prob = model.predict_proba(latest_features)[0]
        direction = "📈 UP" if pred == 1 else "📉 DOWN"
        confidence = f"{prob[pred]*100:.2f}%"

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediction", direction)
        with col2:
            st.metric("Confidence", confidence)

        # ✅ Save prediction
        stock_name = ticker if source == "Live Yahoo Finance" else selected_file
        save_prediction(stock_name, direction, confidence)

        # ✅ Show Prediction History
        st.subheader("📝 Prediction History")
        history_df = load_prediction_history()
        st.dataframe(history_df)

        # ✅ Model Feature Importance
        st.subheader("📊 Model Feature Importance")
        if hasattr(model, "feature_importances_"):
            features = ["MACD", "MACD_Signal", "BB_High", "BB_Low", "Close"]
            fi_df = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_})
            st.plotly_chart(px.bar(fi_df, x="Feature", y="Importance", title="Feature Importance"), use_container_width=True)
    else:
        st.warning("⚠️ No data available")

# ✅ FOOTER
st.markdown("---")
st.caption("🚀 Built with Streamlit | AI Stock Predictor Dashboard")


