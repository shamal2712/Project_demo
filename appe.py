
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Sample model training inside Streamlit for demonstration (replace with joblib.load if using saved model)
@st.cache_resource
def train_model():
    df = pd.read_csv("MED INS.csv")
    X = df.drop("charges", axis=1)
    y = df["charges"]

    categorical_cols = ["sex", "smoker", "region"]
    numeric_cols = ["age", "bmi", "children"]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ], remainder="passthrough")

    X_processed = preprocessor.fit_transform(X)
    model = LinearRegression()
    model.fit(X_processed, y)

    return model, preprocessor

model, preprocessor = train_model()

st.title("Medical Insurance Cost Predictor")

# User Inputs
age = st.number_input("Age", min_value=1, max_value=100, value=30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.slider("Number of Children", 0, 5, 1)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# Prepare input
input_df = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker],
    'region': [region]
})

# Prediction
if st.button("Predict Insurance Cost"):
    input_processed = preprocessor.transform(input_df)
    prediction = model.predict(input_processed)
    st.success(f"Estimated Insurance Cost: ₹ {prediction[0]:,.2f}")
