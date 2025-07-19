import streamlit as st
import pandas as pd
import joblib
import os
import datetime
import plotly.graph_objects as go
import plotly.express as px
import requests
import yfinance as yf
import ta

# ✅ Alpha Vantage API Key
ALPHA_VANTAGE_KEY = "IY2HMVXFHXE83LB5"

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

# ✅ Fetch Yahoo Finance Data
def fetch_yahoo_data(ticker, period="6mo"):
    df = yf.download(ticker, period=period, interval="1d")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    expected_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in expected_cols if c in df.columns]]
    return df

# ✅ Fetch Alpha Vantage Intraday Data
def fetch_alpha_intraday(symbol, interval="15min"):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={ALPHA_VANTAGE_KEY}"
    r = requests.get(url).json()
    key = f"Time Series ({interval})"
    if key not in r:
        st.error("❌ Alpha Vantage API limit reached or invalid symbol")
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(r[key], orient="index")
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    df = df.astype(float).reset_index().rename(columns={"index": "Datetime"})
    df = df.sort_values("Datetime")
    # Convert to datetime
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    # Rename to match model expectations
    df.rename(columns={"Datetime": "Date"}, inplace=True)
    return df

# ✅ Safe ADX & Stoch Wrappers
def safe_adx(df):
    try:
        adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
        return adx.adx()
    except Exception:
        st.warning("⚠️ Could not compute ADX (too few valid rows).")
        return pd.Series([None] * len(df))

def safe_stoch(df):
    try:
        stoch = ta.momentum.StochRSIIndicator(df["Close"], window=14)
        return stoch.stochrsi_k(), stoch.stochrsi_d()
    except Exception:
        st.warning("⚠️ Could not compute Stochastic RSI.")
        return pd.Series([None] * len(df)), pd.Series([None] * len(df))

# ✅ Add Technical Indicators
def add_technical_indicators(df):
    if df is None or df.empty:
        st.warning("⚠️ No valid data to calculate indicators.")
        return df

    if "Close" not in df.columns or df["Close"].ndim != 1:
        st.error("❌ Invalid data format. Try another ticker.")
        return pd.DataFrame()

    for col in ["Close", "High", "Low"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Close", "High", "Low"])

    if len(df) < 20:
        st.warning(f"⚠️ Too few rows ({len(df)}) for stable indicators. Try a longer period.")
        return df

    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()

    bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()
    df["BB_Width"] = df["BB_High"] - df["BB_Low"]

    df["ADX"] = safe_adx(df)
    df["Stoch_K"], df["Stoch_D"] = safe_stoch(df)

    df["Lag_1"] = df["Close"].shift(1)
    df["Lag_3"] = df["Close"].shift(3)
    df["Lag_5"] = df["Close"].shift(5)

    return df

# ✅ Plotting
def plot_candlestick(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"))
    fig.update_layout(height=500, template="plotly_dark")
    return fig

def plot_rsi_macd(df):
    fig = go.Figure()
    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], name="RSI"))
    if "MACD" in df.columns:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], name="MACD"))
    if "MACD_Signal" in df.columns:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD_Signal"], name="MACD Signal"))
    fig.update_layout(height=300, template="plotly_dark")
    return fig

# ✅ Prediction History
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

# ✅ Sidebar
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
    data_source = st.sidebar.radio("📂 Select Data Source", ["Yahoo Finance (Daily)", "Alpha Vantage (Intraday)"])

    # ✅ Stock symbol input
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g. RELIANCE.BSE, AAPL)", "RELIANCE.BSE")

    if data_source == "Yahoo Finance (Daily)":
        period = st.sidebar.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y"])
        df = fetch_yahoo_data(ticker, period)
    else:
        interval = st.sidebar.selectbox("Intraday Interval", ["5min", "15min", "30min", "60min"], index=1)
        df = fetch_alpha_intraday(ticker, interval)

    # ✅ Normalize Date
    if df is not None and not df.empty:
        if "Date" not in df.columns:
            if df.index.name in ["Date", "Datetime", None]:
                df = df.reset_index()
            df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    if df is None or df.empty:
        st.error("❌ No data returned. Check ticker or date range.")
    else:
        df = add_technical_indicators(df)

        if not df.empty:
            tab1, tab2 = st.tabs(["📊 Price Chart", "📉 Indicators"])
            with tab1:
                st.plotly_chart(plot_candlestick(df), use_container_width=True)
            with tab2:
                st.plotly_chart(plot_rsi_macd(df), use_container_width=True)

            required_features = [
                "MACD", "MACD_Signal", "BB_High", "BB_Low", "BB_Width",
                "ADX", "Stoch_K", "Stoch_D", "Lag_1", "Lag_3", "Lag_5"
            ]
            df = df.dropna(subset=required_features)

            if not df.empty:
                latest_features = df[required_features].iloc[[-1]]
                pred = model.predict(latest_features)[0]
                prob = model.predict_proba(latest_features)[0]
                direction = "📈 UP" if pred == 1 else "📉 DOWN"
                confidence = f"{prob[pred]*100:.2f}%"

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Prediction", direction)
                with col2:
                    st.metric("Confidence", confidence)

                stock_name = ticker
                save_prediction(stock_name, direction, confidence)

                st.subheader("📝 Prediction History")
                history_df = load_prediction_history()
                st.dataframe(history_df)

                st.subheader("📊 Model Feature Importance")
                if hasattr(model, "feature_importances_"):
                    features = required_features
                    fi_df = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_})
                    st.plotly_chart(px.bar(fi_df, x="Feature", y="Importance", title="Feature Importance"), use_container_width=True)
            else:
                st.warning("⚠️ Not enough valid rows for prediction after computing indicators.")
        else:
            st.warning("⚠️ Data invalid or too short for indicators.")

st.markdown("---")
st.caption("🚀 Built with Streamlit | AI Stock Predictor Dashboard (Yahoo + Alpha Vantage Intraday)")






