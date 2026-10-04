# STEP 13: Write the App to a file
import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_resource
def load_model():
    return joblib.load("model.joblib"), joblib.load("scaler.joblib")

model, scaler = load_model()

# Page configuration
st.set_page_config(page_title="Breast Tumor Diagnosis Predictor", page_icon="🩺", layout="centered")

st.title("Breast Tumor Diagnosis Predictor 🩺🎗️")
st.write("Enter the patient's clinical measurements below to predict whether the tumor is Benign or Malignant.")

# Let's create a simple input interface using 3 of our most influential features
worst_concavity = st.number_input("Worst Concavity", value=0.1)
mean_radius = st.number_input("Mean Radius", value=14.0)
texture_error = st.number_input("Texture Error", value=1.0)

if st.button("Predict"):
    # Map features back to the required 30 inputs expected by the scaler and model
    # Using mean values as a placeholder for the remaining features
    input_data = np.zeros((1, 30))

    # Feature index mapping from data.feature_names:
    # mean radius is index 0
    input_data[0, 0] = mean_radius
    # texture error is index 11
    input_data[0, 11] = texture_error
    # worst concavity is index 26
    input_data[0, 26] = worst_concavity

    # Scale features and predict
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    
    confidence = max(probabilities) * 100

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"🟢 **Benign (non-cancerous)**")
        st.info(f"Confidence Level: **{confidence:.2f}%**")
    else:
        st.error(f"🔴 **Malignant (cancerous)**")
        st.warning(f"Confidence Level: **{confidence:.2f}%**")
   

st.caption("⚠️ Disclaimer: This is a machine learning demo application for educational purposes only. It is not intended to be a real medical diagnostic tool.")
