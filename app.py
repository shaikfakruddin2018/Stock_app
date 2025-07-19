import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import plotly.graph_objects as go
from datetime import datetime
from supabase import create_client, Client
import os

# ============================
# ✅ CONFIG
# ============================
ALPHA_VANTAGE_API_KEY = "IY2HMVXFHXE83LB5"  # Replace with your key

SUPABASE_URL = "https://rrvsbizwikocatkdhyfs.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJydnNiaXp3aWtvY2F0a2RoeWZzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI5NjExNDAsImV4cCI6MjA2ODUzNzE0MH0.YWP65KQvwna1yQlhjksyT9Rhpyn5bBw5MDlMVHTF62Q"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ============================
# ✅ FUNCTIONS
# ============================

def fetch_yahoo_data(ticker, period="3mo"):
    """Fetch historical daily data from Yahoo Finance"""
    df = yf.download(ticker, period=period, interval="1d")
    if df.empty:
        return None
    df.reset_index(inplace=True)
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
        return None

    df = pd.DataFrame(r[key]).T
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Datetime"}, inplace=True)
    df[["Open", "High", "Low", "Close", "Volume"]] = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    return df

def plot_candlestick(df, title="Stock Price"):
    """Plot candlestick chart"""
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

def save_prediction_to_supabase(stock, prediction, confidence, source):
    """Save prediction to Supabase"""
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
    """Load previous predictions"""
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

# Data source selection
st.sidebar.subheader("Select Data Source")
data_source = st.sidebar.radio(
    "Fetch data from:",
    ["Yahoo Finance (Daily)", "Alpha Vantage (Intraday)"]
)

# Stock input
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g. AAPL, RELIANCE.BSE)", "AAPL")

if data_source == "Yahoo Finance (Daily)":
    period = st.sidebar.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=1)
else:
    interval = st.sidebar.selectbox("Intraday Interval", ["1min", "5min", "15min", "30min", "60min"], index=1)

st.title("📈 AI Stock Predictor Dashboard")

# Fetch data
if st.sidebar.button("Fetch Data"):
    if data_source == "Yahoo Finance (Daily)":
        df = fetch_yahoo_data(ticker, period)
        source_name = "YahooFinance"
    else:
        df = fetch_alpha_vantage_intraday(ticker, interval)
        source_name = "AlphaVantage"

    if df is None or df.empty:
        st.error("❌ No data returned. Check ticker or date range.")
    else:
        # Show candlestick chart
        st.subheader(f"Stock Data: {ticker} ({source_name})")
        st.plotly_chart(plot_candlestick(df, f"{ticker} Price Chart"), use_container_width=True)

        # Simulate AI Prediction
        import random
        prediction = random.choice(["UP", "DOWN"])
        confidence = random.uniform(51, 70)

        st.markdown(f"### Prediction: **{prediction}**")
        st.markdown(f"### Confidence: **{confidence:.2f}%**")

        # Save prediction to Supabase
        save_prediction_to_supabase(ticker, prediction, confidence, source_name)

# Show Prediction History
st.subheader("📜 Prediction History (Cloud)")
history_df = load_prediction_history_supabase()

if not history_df.empty:
    st.dataframe(history_df[["created_at", "stock", "prediction", "confidence", "source"]])
else:
    st.info("No prediction history yet.")







