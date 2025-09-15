import streamlit as st
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import os

# -------------------------------
# Load trained model
# -------------------------------
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
        The R-squared (R2) score is 86%; The R2 score measures how much of the variation in your output variable (malaria cases) can be explained by your input variables.
    </p>
    """,
    unsafe_allow_html=True
)


# -------------------------------
# Load dataset (for LGA selection and visualization)
# -------------------------------
# Example: load malaria dataset
df = pd.read_excel(r"C:\Users\USER\Downloads\Benue_Malaria.xlsx")   # <-- replace with your actual dataset file

# Extract unique LGAs
lgas = df["LGA"].dropna().unique().tolist()

# Let user select an LGA
selected_lga = st.selectbox("🏘 Select an LGA", options=lgas)

# Filter dataset for the selected LGA
filtered_df = df[df["LGA"] == selected_lga]

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
# Prediction
# -------------------------------
if st.button("🔮 Predict Malaria Cases"):
    features = np.array([[
        person_fever,
        rainfall,
        temperature,
        rainfall_lag1,
        rainfall_lag2,
        temperature_lag5,
        temperature_lag6
    ]])

    # Get predictions from each individual tree
    individual_predictions = []
    for tree in model.estimators_:
        individual_predictions.append(tree.predict(features))
    individual_predictions = np.array(individual_predictions)

    # Calculate the mean prediction (the final forecast)
    mean_prediction = np.mean(individual_predictions, axis=0)[0]
    
    # Calculate the 95% prediction interval (the uncertainty level)
    # The interval represents the range where the true value is expected to fall
    lower_bound = np.percentile(individual_predictions, 2.5, axis=0)[0]
    upper_bound = np.percentile(individual_predictions, 97.5, axis=0)[0]
    
    # Calculate the standard deviation as a measure of the spread or "error margin"
    std_dev = np.std(individual_predictions, axis=0)[0]

    st.success(f"✅ Predicted Malaria Cases in {selected_lga}: {int(mean_prediction)}")
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
possible_cols = ["positive_by RDT", "Persons with Fever", "RDT_Positive"]
col_name = next((c for c in possible_cols if c in filtered_df.columns), None)

if col_name:
    fig, ax = plt.subplots()
    sns.histplot(filtered_df[col_name], kde=True, ax=ax)
    st.pyplot(fig)
else:
    st.warning("⚠️ No suitable column found for visualization.")
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
