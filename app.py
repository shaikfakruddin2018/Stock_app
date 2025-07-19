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
from supabase import create_client

# ✅ SUPABASE CONFIG
SUPABASE_URL = "https://rrvsbizwikocatkdhyfs.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJydnNiaXp3aWtvY2F0a2RoeWZzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI5NjExNDAsImV4cCI6MjA2ODUzNzE0MH0.YWP65KQvwna1yQlhjksyT9Rhpyn5bBw5MDlMVHTF62Q"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ✅ SAVE Prediction to Supabase Cloud
def save_prediction_to_supabase(stock, prediction, confidence, source="Yahoo"):
    try:
        data = {
            "stock": stock,
            "prediction": prediction,
            "confidence": confidence,
            "source": source,
            "created_at": datetime.datetime.now().isoformat()
        }
        response = supabase.table("predictions").insert(data).execute()
        if response.data:
            st.success("✅ Prediction saved to cloud!")
    except Exception as e:
        st.error(f"❌ Failed to save prediction: {e}")

# ✅ LOAD Prediction History from Supabase
def load_prediction_history_supabase(limit=20):
    try:
        response = supabase.table("predictions").select("*").order("created_at", desc=True).limit(limit).execute()
        if response.data:
            df = pd.DataFrame(response.data)
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Failed to fetch history: {e}")
        return pd.DataFrame()

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

    for col in ["Close", "High", "Low"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Close", "High", "Low"])

    if len(df) < 20:
        st.warning(f"⚠️ Too few rows ({len(df)}) for stable indicators.")
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

    # ✅ Stock input
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g. RELIANCE.BSE, AAPL)", "AAPL")
    period = st.sidebar.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y"])

    df = fetch_yahoo_data(ticker, period)

    if df is None or df.empty:
        st.error("❌ No data returned. Check ticker or date range.")
    else:
        df.reset_index(inplace=True)
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

                # ✅ Save to Supabase instead of CSV
                save_prediction_to_supabase(ticker, direction, confidence, source="YahooFinance")

                st.subheader("📝 Prediction History (Cloud)")
                history_df = load_prediction_history_supabase()
                st.dataframe(history_df)
            else:
                st.warning("⚠️ Not enough valid rows for prediction after computing indicators.")

st.markdown("---")
st.caption("🚀 Built with Streamlit | AI Stock Predictor Dashboard + Supabase Cloud")






