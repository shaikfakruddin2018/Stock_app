import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import plotly.graph_objects as go
from datetime import datetime
from supabase import create_client, Client
import random

# ============================
# ✅ CONFIG
# ============================
ALPHA_VANTAGE_API_KEY = "IY2HMVXFHXE83LB5"  # Replace with your key
HUGGINGFACE_BASE_URL = "https://huggingface.co/datasets/shaikfakruddin18/stock-predictor-model/resolve/main/"

SUPABASE_URL = "https://rrvsbizwikocatkdhyfs.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJydnNiaXp3aWtvY2F0a2RoeWZzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI5NjExNDAsImV4cCI6MjA2ODUzNzE0MH0.YWP65KQvwna1yQlhjksyT9Rhpyn5bBw5MDlMVHTF62Q"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ============================
# ✅ FETCH FUNCTIONS
# ============================

def fetch_from_huggingface(ticker):
    """Fetch pre-downloaded CSV from Hugging Face"""
    csv_url = f"{HUGGINGFACE_BASE_URL}{ticker}.csv"
    try:
        df = pd.read_csv(csv_url)
        if not df.empty:
            st.success(f"✅ Loaded from Hugging Face: {ticker}.csv")
            # Normalize columns
            if "Date" not in df.columns:
                df.rename(columns={df.columns[0]: "Date"}, inplace=True)
            df["Date"] = pd.to_datetime(df["Date"])
            return df
    except Exception:
        st.warning(f"⚠ No Hugging Face CSV found for {ticker}")
    return pd.DataFrame()

def fetch_yahoo_data(ticker, period="3mo"):
    """Fetch historical daily data from Yahoo Finance"""
    df = yf.download(ticker, period=period, interval="1d")
    if df.empty:
        return pd.DataFrame()
    df.reset_index(inplace=True)
    st.success("✅ Yahoo Finance data fetched successfully!")
    return df

def fetch_alpha_vantage_intraday(ticker, interval="5min"):
    """Fetch intraday data from Alpha Vantage"""
    url = (
        f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY"
        f"&symbol={ticker}&interval={interval}&apikey={ALPHA_VANTAGE_API_KEY}&datatype=json"
    )
    r = requests.get(url).json()
    key = f"Time Series ({interval})"
    if key not in r:
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

def unified_fetch_stock_data(ticker, period="3mo", interval="5min"):
    """Try Hugging Face → Yahoo → Alpha Vantage"""
    # 1️⃣ Hugging Face first
    df = fetch_from_huggingface(ticker)
    if not df.empty:
        return df, "HuggingFace"
    
    # 2️⃣ Yahoo Finance next
    df = fetch_yahoo_data(ticker, period)
    if not df.empty:
        return df, "YahooFinance"
    
    # 3️⃣ Alpha Vantage fallback
    df = fetch_alpha_vantage_intraday(ticker, interval)
    if not df.empty:
        return df, "AlphaVantage"
    
    return pd.DataFrame(), "None"

# ============================
# ✅ VISUALIZATION
# ============================

def plot_candlestick(df, title="Stock Price"):
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df[df.columns[0]],
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
# ✅ SUPABASE LOGGING
# ============================

def save_prediction_to_supabase(stock, prediction, confidence, source):
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
        st.error(f"❌ Supabase error: {e}")

def load_prediction_history_supabase():
    try:
        response = supabase.table("predictions").select("*").order("created_at", desc=True).execute()
        return pd.DataFrame(response.data) if response.data else pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Failed to load history: {e}")
        return pd.DataFrame()

# ============================
# ✅ STREAMLIT UI
# ============================

st.set_page_config(page_title="AI Stock Predictor", layout="wide")
st.sidebar.title("📊 Navigation")

st.sidebar.subheader("Select Data Source")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g. RELIANCE.NS, AAPL, HDFCBANK.NS)", "RELIANCE.NS")
period = st.sidebar.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=1)
interval = st.sidebar.selectbox("Intraday Interval (for Alpha Vantage fallback)", ["1min", "5min", "15min", "30min", "60min"], index=1)

st.title("📈 AI Stock Predictor Dashboard")

if st.sidebar.button("Fetch Data"):
    with st.spinner("Fetching stock data..."):
        df, source_name = unified_fetch_stock_data(ticker, period, interval)

    if df is None or df.empty:
        st.error("❌ No data returned. Check ticker or date range.")
    else:
        st.subheader(f"Stock Data: {ticker} ({source_name})")
        st.plotly_chart(plot_candlestick(df, f"{ticker} Price Chart"), use_container_width=True)

        # Simulated AI Prediction (later integrate your real model)
        prediction = random.choice(["UP", "DOWN"])
        confidence = random.uniform(51, 70)

        st.markdown(f"### Prediction: **{prediction}**")
        st.markdown(f"### Confidence: **{confidence:.2f}%**")

        # Save to Supabase
        save_prediction_to_supabase(ticker, prediction, confidence, source_name)

# ✅ Show Prediction History
st.subheader("📜 Prediction History (Cloud)")
history_df = load_prediction_history_supabase()
if not history_df.empty:
    st.dataframe(history_df[["created_at", "stock", "prediction", "confidence", "source"]])
else:
    st.info("No prediction history yet.")








