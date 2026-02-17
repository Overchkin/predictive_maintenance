import sys
import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- Fix chemin pour imports Windows ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
from models.predict import Predictor  # import après ajout du path

# Initialisation du prédicteur
predictor = Predictor()

# Config Streamlit
st.set_page_config(
    page_title="Predictive Maintenance",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Header ---
st.title("Industrial Predictive Maintenance Dashboard")
st.markdown("Predict machine failures, visualize feature importance, and track sensor data")

# --- KPI Metrics ---
col1, col2, col3 = st.columns(3)
latest_prob = st.session_state.history[-1]["Failure Probability"] if st.session_state.get("history") else 0
total_preds = len(st.session_state.history) if st.session_state.get("history") else 0
avg_tool_wear = (pd.DataFrame(st.session_state.history)["Tool wear [min]"].mean()
                 if st.session_state.get("history") else 0)
col1.metric("Latest Failure Risk", f"{latest_prob:.2%}")
col2.metric("Total Predictions", total_preds)
col3.metric("Average Tool Wear [min]", f"{avg_tool_wear:.1f}")

# --- Sidebar Inputs ---
st.sidebar.header("Input Sensor Values")
with st.sidebar.form("sensor_form"):
    air_temp = st.number_input("Air temperature [K]", value=300.0)
    process_temp = st.number_input("Process temperature [K]", value=310.0)
    rot_speed = st.number_input("Rotational speed [rpm]", value=1500.0)
    torque = st.number_input("Torque [Nm]", value=40.0)
    tool_wear = st.number_input("Tool wear [min]", value=0.0)
    type_product = st.selectbox("Product Type", ["L", "M", "H"])
    submitted = st.form_submit_button("Predict Failure")
    reset_history = st.form_submit_button("Reset History")

# --- Initialisation historique ---
if "history" not in st.session_state:
    st.session_state.history = []

if reset_history:
    st.session_state.history = []

# --- Bouton prédiction ---
if submitted:
    input_data = {
        "Type": type_product,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rot_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
    }

    result = predictor.predict(input_data)
    pred = result["prediction"]
    prob = result["failure_probability"]

    # Ajouter au historique
    st.session_state.history.append({
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **input_data,
        "Prediction": pred,
        "Failure Probability": prob
    })

    # --- Layout principal ---
    col1, col2 = st.columns([1, 2])

    # --- Gauge et résultat ---
    with col1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob*100,
            title={'text': "Failure Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red" if pred == 1 else "green"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "red"}
                ]
            }
        ))
        st.plotly_chart(fig_gauge, width='stretch')

        st.subheader("Prediction Result")
        if pred == 1:
            st.error("Machine Failure Risk Detected ❌")
        else:
            st.success("Machine Operating Normally ✅")
        st.write(f"Failure Probability: {prob:.2%}")

    # --- Feature importance ---
    with col2:
        importances = predictor.model.feature_importances_
        feature_names = ["Air temperature [K]", "Process temperature [K]",
                         "Rotational speed [rpm]", "Torque [Nm]",
                         "Tool wear [min]"]
        # Ajuster la longueur si nécessaire
        if len(importances) > len(feature_names):
            feature_names += [f"Feature_{i}" for i in range(len(importances) - len(feature_names))]
        fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=True)
        st.subheader("Feature Importance")
        fig_bar = px.bar(
            fi_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_bar, width='stretch')

# --- Historique des prédictions ---
if st.session_state.history:
    with st.expander("Prediction History & Sensor Data"):
        hist_df = pd.DataFrame(st.session_state.history)
        st.subheader("Prediction History")
        st.dataframe(hist_df)

        st.subheader("Sensor Data Overview")
        fig_sensors = px.histogram(
            hist_df,
            x=["Air temperature [K]", "Process temperature [K]",
               "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"],
            barmode="overlay",
            nbins=20,
            title="Distribution of Sensor Values",
            marginal="box"
        )
        st.plotly_chart(fig_sensors, width='stretch')
