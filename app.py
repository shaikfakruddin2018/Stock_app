import streamlit as st
import pandas as pd
import requests, io

# ✅ Supabase Public Bucket Base URL
SUPABASE_PUBLIC_BUCKET = "https://rrvsbizwikocatkdhyfs.supabase.co/storage/v1/object/public/prediction/stock_data"

def load_csv_from_supabase(stock_file: str):
    """Load CSV directly from Supabase public storage"""
    url = f"{SUPABASE_PUBLIC_BUCKET}/{stock_file}"
    st.write(f"📂 Fetching: {url}")  # Debug info
    
    res = requests.get(url)
    if res.status_code == 200:
        df = pd.read_csv(io.StringIO(res.text))
        return df
    else:
        st.error(f"❌ Could not load {stock_file} (HTTP {res.status_code})")
        return None

# ✅ Streamlit UI
st.title("📊 Stock Data Viewer (Supabase)")

# ✅ List of stock files you uploaded (add more here)
available_stocks = [
    "ADANIENT.csv",
    "RELIANCE.csv",
    "ICICIBANK.csv",
    "TCS.csv",
    "HDFCBANK.csv"
]

# ✅ Dropdown for stock selection
selected_stock = st.selectbox("Select Stock CSV", available_stocks)

# ✅ Fetch & show
if st.button("Load Stock Data"):
    df = load_csv_from_supabase(selected_stock)
    if df is not None:
        st.success(f"✅ Loaded {selected_stock}")
        st.write("Preview of Stock Data:")
        st.dataframe(df.head())  # Show first 5 rows





