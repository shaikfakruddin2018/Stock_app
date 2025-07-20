import streamlit as st
import yfinance as yf

st.title("✅ Quick Test")

ticker = st.text_input("Ticker", "AAPL")
if st.button("Fetch"):
    st.write("Fetching...")
    df = yf.download(ticker, period="1mo")
    st.write(df.head())




