import streamlit as st
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
import statsmodels.api as sm

# -------------------------------
# Load trained model
# -------------------------------
model = joblib.load("malaria_rf_model.joblib")
st.set_page_config(page_title="Malaria Prediction App", page_icon="🦟", layout="centered")

# -------------------------------
# App Header
# -------------------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1;'>🌍 Malaria Cases Prediction App</h1>
    <p style='text-align: center; font-size: 18px;'>
        Predict malaria cases (number of persons who tested positive by RDT) using rainfall, temperature, lagged features and fever-related features.  
        <br>Use this tool for planning and deployment of targeted interventions, not for medical diagnosis.
    </p>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# User Inputs
# -------------------------------
st.subheader("📥 Enter Input Features")

col1, col2 = st.columns(2)

with col1:
    person_fever = st.number_input("Persons with Fever", min_value=0.0, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=0.1)
    rainfall_lag1 = st.number_input("Rainfall Lag 1", min_value=0.0, step=0.1)
    temperature_lag5 = st.number_input("Temperature Lag 5", min_value=0.0, step=0.1)

with col2:
    temperature = st.number_input("Temperature (°C)", min_value=0.0, step=0.1)
    rainfall_lag2 = st.number_input("Rainfall Lag 2", min_value=0.0, step=0.1)
    temperature_lag6 = st.number_input("Temperature Lag 6", min_value=0.0, step=0.1)

# -------------------------------
# Prediction (Basic)
# -------------------------------
if st.button("🔮 Predict Malaria Cases", key="basic_prediction"):
    features = np.array([[
        person_fever,
        rainfall,
        temperature,
        rainfall_lag1,
        rainfall_lag2,
        temperature_lag5,
        temperature_lag6
    ]])

    prediction = model.predict(features)[0]
    st.success(f"✅ Predicted Malaria Cases: {int(prediction)}")

# -------------------------------
# Prediction (with Uncertainty)
# -------------------------------
if st.button("🔮 Predict Malaria Cases (with Uncertainty)", key="uncertainty_prediction"):
    features_input = np.array([[
        person_fever,
        rainfall,
        temperature,
        rainfall_lag1,
        rainfall_lag2,
        temperature_lag5,
        temperature_lag6
    ]])

    individual_predictions = []
    for tree in model.estimators_:
        individual_predictions.append(tree.predict(features_input))
    individual_predictions = np.array(individual_predictions)

    mean_prediction = np.mean(individual_predictions, axis=0)[0]
    lower_bound = np.percentile(individual_predictions, 2.5, axis=0)[0]
    upper_bound = np.percentile(individual_predictions, 97.5, axis=0)[0]
    std_dev = np.std(individual_predictions, axis=0)[0]

    st.success(f"✅ Predicted Malaria Cases: {int(mean_prediction)}")
    st.info(
        f"**Uncertainty Level (95% Prediction Interval):** "
        f"The number of cases is likely to be between **{int(lower_bound)}** and **{int(upper_bound)}**."
    )
    st.warning(f"**Error Margin (Standard Deviation):** The prediction's variability is approximately **±{std_dev:.2f}**.")

# -------------------------------
# Disclaimer
# -------------------------------
st.markdown(
    """
    <hr>
    <p style='font-size: 14px; color: gray; text-align: center;'>
    ⚠️ <b>Disclaimer:</b> This tool is intended for <b>research and public health planning</b> only.
    It does <b>not provide medical diagnosis</b>. For personal health concerns, please consult a qualified healthcare provider.
    </p>
    """,
    unsafe_allow_html=True
)
