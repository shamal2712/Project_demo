
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("model.pkl")  # Make sure model.pkl is in the same directory

st.title("Medical Insurance Cost Predictor")

st.markdown("### Enter Patient Details")

# Example input fields - customize based on your model
age = st.number_input("Age", min_value=1, max_value=100, value=30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.slider("Number of Children", 0, 5, 1)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Prepare input DataFrame
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker],
    'region': [region]
})

# Preprocessing if needed (e.g., one-hot encoding) must match training phase

# Predict button
if st.button("Predict Insurance Cost"):
    try:
        prediction = model.predict(input_data)
        st.success(f"Predicted Insurance Cost: ₹ {prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Error: {e}")
