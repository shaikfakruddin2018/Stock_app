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

# 🔗 Your GitHub repo raw URL base (change this to your repo!)
GITHUB_BASE_URL = "https://raw.githubusercontent.com/<your-username>/<your-repo>/main/data/"

SUPABASE_URL = "https://rrvsbizwikocatkdhyfs.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJydnNiaXp3aWtvY2F0a2RoeWZzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI5NjExNDAsImV4cCI6MjA2ODUzNzE0MH0.YWP65KQvwna1yQlhjksyT9Rhpyn5bBw5MDlMVHTF62Q"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ============================
# ✅ FETCH FUNCTIONS
# ============================

def fetch_from_github(ticker):
    """Fetch pre-downloaded CSV from GitHub repo"""
    csv_url = f"{GITHUB_BASE_URL}{ticker}.csv"
    st.write(f"🔍 Trying GitHub CSV: {csv_url}")
    try:
        response = requests.get(csv_url, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(csv_url)
            if not df.empty:
                st.success(f"✅ Loaded from GitHub CSV: {ticker}.csv")
                if "Date" not in df.columns:
                    df.rename(columns={df.columns[0]: "Date"}, inplace=True)
                df["Date"] = pd.to_datetime(df["Date"])
                return df
        else:
            st.warning(f"⚠ GitHub returned {response.status_code} for {ticker}")
    except Exception as e:
        st.warning(f"⚠ GitHub CSV failed: {e}")
    return pd.DataFrame()

def fetch_yahoo_data(ticker, period="3mo"):
    """Fetch historical daily data from Yahoo Finance"""
    st.write("🔍 Trying Yahoo Finance...")
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False)
        if df.empty:
            return pd.DataFrame()
        df.reset_index(inplace=True)
        st.success("✅ Yahoo Finance data fetched successfully!")
        return df
    except Exception as e:
        st.warning(f"⚠ Yahoo Finance error: {e}")
        return pd.DataFrame()

def fetch_alpha_vantage_intraday(ticker, interval="5min"):
    """Fetch intraday data from Alpha Vantage"""
    st.write("🔍 Trying Alpha Vantage...")
    url = (
        f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY"
        f"&symbol={ticker}&interval={interval}&apikey={ALPHA_VANTAGE_API_KEY}&datatype=json"
    )
    try:
        r = requests.get(url, timeout=10).json()
        key = f"Time Series ({interval})"
        if key not in r:
            st.error("❌ Alpha Vantage returned no data.")
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
        st.error(f"❌ Alpha Vantage fetch error: {e}")
        return pd.DataFrame()

def unified_fetch_stock_data(ticker, period="3mo", interval="5min"):
    """Try GitHub CSV → Yahoo → Alpha Vantage"""
    # 1️⃣ GitHub CSV first
    df = fetch_from_github(ticker)
    if not df.empty:
        return df, "GitHub CSV"
    
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

st.sidebar.subheader("Stock Selection")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g. RELIANCE.NS, AAPL, HDFCBANK.NS)", "RELIANCE.NS")
period = st.sidebar.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=1)
interval = st.sidebar.selectbox("Intraday Interval (for Alpha Vantage fallback)", ["1min", "5min", "15min", "30min", "60min"], index=1)

st.title("📈 AI Stock Predictor Dashboard")

if st.sidebar.button("Fetch Data"):
    with st.spinner("Fetching stock data..."):
        df, source_name = unified_fetch_stock_data(ticker, period, interval)

    if df is None or df.empty:
        st.error("❌ No data returned from any source. Try a different ticker.")
    else:
        st.subheader(f"Stock Data: {ticker} ({source_name})")
        
        # ✅ Debug preview
        st.write("🔍 Data preview (first 5 rows):")
        st.dataframe(df.head())

        # ✅ Plot chart
        st.plotly_chart(plot_candlestick(df, f"{ticker} Price Chart"), use_container_width=True)

        # ✅ Simulated AI Prediction (replace with your ML model later)
        prediction = random.choice(["UP", "DOWN"])
        confidence = random.uniform(51, 75)

        st.markdown(f"### Prediction: **{prediction}**")
        st.markdown(f"### Confidence: **{confidence:.2f}%**")

        # ✅ Save to Supabase
        save_prediction_to_supabase(ticker, prediction, confidence, source_name)

# ✅ Show Prediction History
st.subheader("📜 Prediction History (Cloud)")
history_df = load_prediction_history_supabase()
if not history_df.empty:
    st.dataframe(history_df[["created_at", "stock", "prediction", "confidence", "source"]])
else:
    st.info("No prediction history yet.")










