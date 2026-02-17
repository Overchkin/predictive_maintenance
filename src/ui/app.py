import sys
import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from src.models.predict import Predictor

# Fix chemin Windows
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Initialisation
predictor = Predictor()

# Config Streamlit
st.set_page_config(page_title="Predictive Maintenance", layout="wide", initial_sidebar_state="expanded")
st.title("Industrial Predictive Maintenance Dashboard")
st.markdown("Predict machine failures, visualize feature importance, and track sensor data")

# Sidebar pour inputs
st.sidebar.header("Input Sensor Values")
air_temp = st.sidebar.number_input("Air temperature [K]", value=300.0)
process_temp = st.sidebar.number_input("Process temperature [K]", value=310.0)
rot_speed = st.sidebar.number_input("Rotational speed [rpm]", value=1500.0)
torque = st.sidebar.number_input("Torque [Nm]", value=40.0)
tool_wear = st.sidebar.number_input("Tool wear [min]", value=0.0)
type_product = st.sidebar.selectbox("Product Type", ["L", "M", "H"])

# Initialisation historique
if "history" not in st.session_state:
    st.session_state.history = []

# Bouton réinitialiser historique
if st.sidebar.button("Réinitialiser l'historique"):
    st.session_state.history = []

# Bouton prédiction
if st.sidebar.button("Predict Failure"):
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
                'bar': {'color': "red" if pred==1 else "green"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "red"}
                ]
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.subheader("Prediction Result")
        if pred == 1:
            st.error(" Machine Failure Risk Detected")
        else:
            st.success("Machine Operating Normally")
        st.write(f"Failure Probability: {prob:.2%}")

    # --- Feature importance ---
    with col2:
        importances = predictor.model.feature_importances_
        feature_names = predictor.scaler.feature_names_in_
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
        st.plotly_chart(fig_bar, use_container_width=True)

# --- Historique des prédictions ---
if st.session_state.history:
    st.subheader("Prediction History")
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df)

    # --- Dashboard des capteurs ---
    st.subheader("Sensor Data Overview")
    fig_sensors = px.histogram(
        hist_df,
        x=["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"],
        barmode="overlay",
        nbins=20,
        title="Distribution of Sensor Values",
        marginal="box"
    )
    st.plotly_chart(fig_sensors, use_container_width=True)
