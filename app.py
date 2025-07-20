import streamlit as st

try:
    import joblib
    st.success("✅ joblib installed successfully!")
except Exception as e:
    st.error(f"❌ Import error: {e}")
