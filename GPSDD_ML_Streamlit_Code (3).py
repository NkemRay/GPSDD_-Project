import streamlit as st
import joblib
import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

# -------------------------------
# Load trained model
# -------------------------------
MODEL_PATH = "malaria_rf_model.joblib"
model = joblib.load(MODEL_PATH)

# Check that model is fitted
try:
    check_is_fitted(model)
except NotFittedError:
    st.error("❌ The loaded model is not fitted. Please retrain and re-save the model.")
    st.stop()

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
    persons_fever = st.number_input("Persons with Fever", min_value=0.0, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=0.1)
    rainfall_lag1 = st.number_input("Rainfall Lag 1", min_value=0.0, step=0.1)
    temp_lag5 = st.number_input("Temperature Lag 5", min_value=0.0, step=0.1)

with col2:
    temperature = st.number_input("Temperature (°C)", min_value=0.0, step=0.1)
    rainfall_lag2 = st.number_input("Rainfall Lag 2", min_value=0.0, step=0.1)
    temp_lag6 = st.number_input("Temperature Lag 6", min_value=0.0, step=0.1)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔮 Predict Malaria Cases"):
    # IMPORTANT: match feature order with training!
    features = np.array([[
        temperature,       # "Temperature (celsuis)"
        rainfall,          # "Rainfall (mm)"
        persons_fever,     # "Persons_fever"
        rainfall_lag1,     # "Rainfall_lag1"
        rainfall_lag2,     # "Rainfall_lag2"
        temp_lag5,         # "Temp_lag5"
        temp_lag6          # "Temp_lag6"
    ]])

    # Prediction from ensemble
    individual_predictions = np.array([tree.predict(features) for tree in model.estimators_])
    mean_prediction = np.mean(individual_predictions)
    lower_bound = np.percentile(individual_predictions, 2.5)
    upper_bound = np.percentile(individual_predictions, 97.5)
    std_dev = np.std(individual_predictions)

    st.success(f"✅ Predicted Malaria Cases: {int(mean_prediction)}")
    st.info(f"**95% Prediction Interval:** {int(lower_bound)} – {int(upper_bound)}")
    st.warning(f"**Error Margin (±1σ):** ~ {std_dev:.2f}")

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
