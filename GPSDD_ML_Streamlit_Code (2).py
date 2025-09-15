import streamlit as st
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

# -------------------------------
# Load trained model
# -------------------------------
# Ensure 'Random_Regressor.pkl' is in the same directory as your app
model = joblib.load("Random_Regressor.pkl")
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
    <p style='text-align: center; font-size: 18px;'>
        Lagged features are past values of climate variables used to understand their delayed effect. For example, Rainfall lag_1 means the rainfall from one month ago, while Temperature lag_5 means the temperature from five months ago
    </p>
    <p style='text-align: center; font-size: 18px;'>
        The predictive model is applied at a general level and does not provide separate results for each LGA. However, the LGAs are only used for linking and displaying the distribution in the visualizations
    </p>
    <p style='text-align: center; font-size: 18px;'>
        The R-squared (R2) score for the model is <b>86%</b>. This score measures how much of the variation in your output variable (malaria cases) can be explained by your input variables.
    </p>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# File Uploader
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload your Malaria dataset", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("✅ Dataset loaded successfully.")
    except Exception as e:
        st.error(f"⚠️ Error reading file: {e}")
        st.stop()
else:
    st.warning("⚠️ Please upload your dataset (.csv or .xlsx) to proceed.")
    st.stop()

# -------------------------------
# Check for a single required column and prepare data for visuals
# -------------------------------
required_column = "LGA"
if required_column not in df.columns:
    st.error(f"⚠️ Your dataset does not contain the required column: '{required_column}'. Please check column names.")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

lgas = df["LGA"].dropna().unique().tolist()
selected_lga = st.selectbox("🏘 Select an LGA", options=lgas)
filtered_df = df[df["LGA"] == selected_lga]

st.write(f"📊 Showing available data for **{selected_lga}**")
st.dataframe(filtered_df.head())

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
# Prediction (with Uncertainty)
# -------------------------------
if st.button("🔮 Predict Malaria Cases"):
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
        f"**Uncertainty Level (95% Prediction Interval):**"
        f" The number of cases is likely to be between **{int(lower_bound)}** and **{int(upper_bound)}**."
    )
    st.warning(f"**Error Margin (Standard Deviation):** The prediction's variability is approximately **±{std_dev:.2f}**.")

# -------------------------------
# Visualization by LGA
# -------------------------------
st.subheader(f"📊 Visualization for {selected_lga}")

# Try to find a suitable column for visualization
col_name = "positive_by RDT"
if col_name in filtered_df.columns:
    fig, ax = plt.subplots()
    sns.histplot(filtered_df[col_name], kde=True, ax=ax)
    st.pyplot(fig)
else:
    st.warning("⚠️ No suitable column found for visualization. Columns with 'RDT' or 'fever' are preferred.")
    st.write("Available columns:", filtered_df.columns.tolist())

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