import streamlit as st
import pandas as pd
import joblib
import os
import datetime
import plotly.graph_objects as go
import requests

# ---- MODEL AUTO-DOWNLOAD ----
MODEL_PATH = "rf_model.joblib"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1TGCp1gcj9b4Ju7Oein-lghBQ3b4jHorF"

if not os.path.exists(MODEL_PATH):
    st.write("📥 Downloading model from Google Drive...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    st.write("✅ Model downloaded successfully!")

# Load model
model = joblib.load(MODEL_PATH)

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="AI Stock Predictor Dashboard",
    page_icon="📈",
    layout="wide",
)

# ---- SIDEBAR ----
st.sidebar.title("📊 Navigation")
menu = st.sidebar.radio("Go to:", ["📈 Stock Predictor", "🧠 Focus Tasks"])

# ---- SESSION STATE ----
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# ---- FOCUS TASK PAGE ----
if menu == "🧠 Focus Tasks":
    st.title("🧠 Focus Task Assistant")

    # Task input
    task = st.text_input("✍️ Add a task")
    if st.button("➕ Add"):
        if task:
            st.session_state.setdefault("tasks", []).append({"task": task, "added": str(datetime.datetime.now())})
            st.success("Task added!")

    # Task List
    st.subheader("📋 Your Tasks")
    for i, t in enumerate(st.session_state.get("tasks", [])):
        st.markdown(f"✅ {t['task']} *(added {t['added'].split('.')[0]})*")

    # Focus Mode
    if st.button("🎯 Start Focus Timer"):
        st.success("⏱️ Focus mode started for 25 mins! Stay focused.")

# ---- STOCK PREDICTOR PAGE ----
else:
    st.title("📈 AI Stock Predictor Dashboard")

    data_dir = "stock_data_with_indicators"
    if not os.path.exists(data_dir):
        st.error(f"Missing folder: {data_dir}")
    else:
        files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
        selected_file = st.sidebar.selectbox("📂 Select Stock", files)

        if selected_file:
            df = pd.read_csv(os.path.join(data_dir, selected_file), index_col=0, parse_dates=True)
            df = df.dropna(subset=[
                'MACD', 'MACD_Signal', 'BB_High', 'BB_Low', 'BB_Width',
                'ADX', 'Stoch_K', 'Stoch_D', 'Lag_1', 'Lag_3', 'Lag_5'
            ])

            if not df.empty:
                # ---- CHART ----
                st.subheader("📊 Stock Price & Indicators")
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Price'
                ))
                fig.update_layout(height=500, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

                # ---- PREDICTION ----
                latest = df[[
                    'MACD', 'MACD_Signal', 'BB_High', 'BB_Low', 'BB_Width',
                    'ADX', 'Stoch_K', 'Stoch_D', 'Lag_1', 'Lag_3', 'Lag_5'
                ]].iloc[[-1]]

                pred = model.predict(latest)[0]
                prob = model.predict_proba(latest)[0]
                direction = "📈 UP" if pred == 1 else "📉 DOWN"
                confidence = f"{prob[pred]*100:.2f}%"

                # Prediction Result Cards
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Prediction", direction)
                with col2:
                    st.metric("Confidence", confidence)

                # Save to history
                st.session_state.prediction_history.append({
                    "Stock": selected_file,
                    "Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "Prediction": direction,
                    "Confidence": confidence
                })

                # ---- HISTORY ----
                st.subheader("📝 Prediction History")
                history_df = pd.DataFrame(st.session_state.prediction_history)
                st.dataframe(history_df)

            else:
                st.warning("⚠️ Not enough data after cleaning.")

# ---- FOOTER ----
st.markdown("---")
st.caption("🚀 Built with Streamlit | AI Stock Predictor Dashboard")