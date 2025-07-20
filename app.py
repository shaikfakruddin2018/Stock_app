import os
import io
import requests
import joblib
import random
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
from pathlib import Path
from datetime import datetime
from supabase import create_client, Client

# ============================
# ✅ CONFIG
# ============================

# Supabase Storage (for historical CSVs)
SUPABASE_STORAGE_BASE = "https://rrvsbizwikocatkdhyfs.supabase.co/storage/v1/object/public/prediction/stock_data"

# Supabase DB
SUPABASE_URL = "https://rrvsbizwikocatkdhyfs.supabase.co"
SUPABASE_KEY = "YOUR_SUPABASE_ANON_KEY"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Alpha Vantage API Key
ALPHA_VANTAGE_API_KEY = "IY2HMVXFHXE83LB5"

# Hugging Face model (joblib)
HUGGINGFACE_MODEL_URL = "https://huggingface.co/shaikfakruddin18/stock-predictor-model/resolve/main/rf_model.joblib.joblib"
MODEL_PATH = "rf_model.joblib"  # cached model locally

# API request timeout
TIMEOUT_SEC = 5

# ============================
# ✅ MODEL LOADER
# ============================

@st.cache_resource
def load_model():
    """Download model from Hugging Face if not cached"""
    if not Path(MODEL_PATH).exists():
        st.write("📥 Downloading ML model from Hugging Face...")
        res = requests.get(HUGGINGFACE_MODEL_URL)
        if res.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(res.content)
            st.success("✅ Model downloaded!")
        else:
            st.error(f"❌ Could not fetch model (HTTP {res.status_code})")
            return None
    
    try:
        model = joblib.load(MODEL_PATH)
        st.success("✅ ML model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

ml_model = load_model()

# ============================
# ✅ STOCK DATA FETCHING
# ============================

def load_csv_from_supabase(stock_file: str):
    """Load CSV from Supabase public storage"""
    url = f"{SUPABASE_STORAGE_BASE}/{stock_file}"
    st.write(f"📂 Fetching Supabase CSV: {url}")
    try:
        res = requests.get(url, timeout=TIMEOUT_SEC)
        if res.status_code == 200:
            return pd.read_csv(io.StringIO(res.text))
        else:
            st.warning(f"⚠ No Supabase file found for {stock_file} (HTTP {res.status_code})")
            return pd.DataFrame()
    except Exception as e:
        st.warning(f"⚠ Supabase CSV fetch failed: {e}")
        return pd.DataFrame()

def fetch_yahoo_data(ticker, period="3mo"):
    """Fetch daily data from Yahoo Finance"""
    st.write("🔍 Trying Yahoo Finance...")
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False)
        if df.empty:
            return pd.DataFrame()
        df.reset_index(inplace=True)
        st.success("✅ Yahoo Finance data fetched!")
        return df
    except Exception as e:
        st.warning(f"⚠ Yahoo Finance failed: {e}")
        return pd.DataFrame()

def fetch_alpha_vantage_intraday(ticker, interval="5min"):
    """Fetch intraday data from Alpha Vantage"""
    st.write("🔍 Trying Alpha Vantage...")
    url = (
        f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY"
        f"&symbol={ticker}&interval={interval}&apikey={ALPHA_VANTAGE_API_KEY}&datatype=json"
    )
    try:
        r = requests.get(url, timeout=TIMEOUT_SEC).json()
        key = f"Time Series ({interval})"
        if key not in r:
            st.warning("❌ Alpha Vantage returned no data.")
            return pd.DataFrame()
        df = pd.DataFrame(r[key]).T
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df.reset_index(inplace=True)
        df.rename(columns={"index": "Datetime"}, inplace=True)
        df[["Open", "High", "Low", "Close", "Volume"]] = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        st.success("✅ Alpha Vantage intraday data fetched!")
        return df
    except Exception as e:
        st.warning(f"⚠ Alpha Vantage failed: {e}")
        return pd.DataFrame()

def unified_fetch_stock_data(ticker, period="3mo", interval="5min"):
    """
    Priority:
    1️⃣ Supabase Storage (historical)
    2️⃣ Yahoo Finance (daily)
    3️⃣ Alpha Vantage (intraday)
    """
    # Supabase
    df = load_csv_from_supabase(f"{ticker}.csv")
    if not df.empty:
        return df, "Supabase Storage"

    # Yahoo Finance
    df = fetch_yahoo_data(ticker, period)
    if not df.empty:
        return df, "Yahoo Finance"

    # Alpha Vantage
    df = fetch_alpha_vantage_intraday(ticker, interval)
    if not df.empty:
        return df, "Alpha Vantage Intraday"

    return pd.DataFrame(), "None"

# ============================
# ✅ VISUALIZATION
# ============================

def plot_candlestick(df, title="Stock Price"):
    """Candlestick Chart"""
    first_col = df.columns[0]  # Date/Datetime
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df[first_col],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
            )
        ]
    )
    fig.update_layout(title=title, xaxis_rangeslider_visible=False, height=400)
    return fig

# ============================
# ✅ REAL PREDICTION
# ============================

def predict_stock_movement(model, df):
    """Take last row & predict UP/DOWN"""
    if df.empty:
        return None, 0.0

    # Only numeric columns for ML model
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if numeric_df.empty:
        st.warning("⚠ No numeric data for prediction")
        return None, 0.0
    
    latest_features = numeric_df.tail(1).values
    try:
        pred = model.predict(latest_features)[0]
        # Confidence
        if hasattr(model, "predict_proba"):
            conf = max(model.predict_proba(latest_features)[0]) * 100
        else:
            conf = 60.0  # fallback confidence
        return pred, conf
    except Exception as e:
        st.warning(f"⚠ Prediction error: {e}")
        return None, 0.0

# ============================
# ✅ SUPABASE DB LOGGING
# ============================

def save_prediction_to_supabase(stock, prediction, confidence, source):
    """Save prediction result to Supabase DB"""
    try:
        created_at = datetime.utcnow().isoformat()
        data = {
            "created_at": created_at,
            "stock": stock,
            "prediction": prediction,
            "confidence": f"{confidence:.2f}%",
            "source": source
        }
        response = supabase.table("predictions").insert(data).execute()
        if response.data:
            st.success("✅ Prediction saved to Supabase!")
        else:
            st.error(f"❌ Failed to save prediction: {response}")
    except Exception as e:
        st.error(f"❌ Supabase DB error: {e}")

def load_prediction_history_supabase():
    """Load past predictions"""
    try:
        response = supabase.table("predictions").select("*").order("created_at", desc=True).execute()
        return pd.DataFrame(response.data) if response.data else pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Failed to load history: {e}")
        return pd.DataFrame()

# ============================
# ✅ STREAMLIT DASHBOARD
# ============================

st.set_page_config(page_title="AI Stock Predictor", layout="wide")
st.sidebar.title("📊 Stock Predictor")

ticker = st.sidebar.text_input("Enter Stock Ticker", "ADANIENT")
period = st.sidebar.selectbox("Yahoo Finance Period", ["1mo", "3mo", "6mo", "1y"], index=1)
interval = st.sidebar.selectbox("Intraday Interval (Alpha Vantage)", ["1min", "5min", "15min"], index=1)

st.title("📈 AI Stock Predictor Dashboard")

if st.sidebar.button("Fetch & Predict"):
    with st.spinner("Fetching stock data..."):
        df, source_name = unified_fetch_stock_data(ticker, period, interval)

    if df.empty:
        st.error("❌ No data available from Supabase/Yahoo/Alpha")
    else:
        # Preview & chart
        st.subheader(f"Stock Data: {ticker} ({source_name})")
        st.write("📊 Data Preview:")
        st.dataframe(df.head())
        st.plotly_chart(plot_candlestick(df, f"{ticker} Price Chart"), use_container_width=True)

        # ML Prediction
        if ml_model:
            prediction, confidence = predict_stock_movement(ml_model, df)
            if prediction is None:
                st.warning("⚠ Model failed, using random fallback")
                prediction = random.choice(["UP", "DOWN"])
                confidence = random.uniform(55, 75)
        else:
            st.warning("⚠ ML model not available, using random prediction")
            prediction = random.choice(["UP", "DOWN"])
            confidence = random.uniform(55, 75)

        st.markdown(f"### 🔮 Prediction: **{prediction}**")
        st.markdown(f"### ✅ Confidence: **{confidence:.2f}%**")

        # Save to Supabase DB
        save_prediction_to_supabase(ticker, prediction, confidence, source_name)

# Show Prediction History
st.subheader("📜 Prediction History")
history_df = load_prediction_history_supabase()
if not history_df.empty:
    st.dataframe(history_df[["created_at", "stock", "prediction", "confidence", "source"]])
else:
    st.info("No prediction history yet.")
