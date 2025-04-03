import streamlit as st
import mlflow.pyfunc
import numpy as np
import requests
import pandas as pd


API_url = "to modify"

st.title("Breast Cancer Prediction App")

# Define feature names (ensure they match the trained model)
features = ["Perimeter_Worst","ConcavePoints_Worst","Area_Worst","Radius_Worst","Texture_Worst","Smoothness_Worst","Texture_Mean","Concavity_Worst"]

# User input form
form1 = st.form(key="features")
user_inputs = {}


for feature in features:
    user_inputs[feature] = form1.number_input(f"Enter value for {feature}", min_value=0.0, format="%.5f")

# Prediction when form is submitted
if form1.form_submit_button("Predict"):
    full_input = np.zeros((1, 30))

    # Fill the corresponding feature values from user input
    for i, feature in enumerate(features):
        feature_index = features.index(feature)  # Get the original index of the feature
        full_input[0, feature_index] = user_inputs[feature]  # Assign user input value

    
    feature_request = requests.post(API_url, json=user_inputs)
    prediction = feature_request.json().get("prediction", None)
    
    st.write(f"### Prediction: {prediction}")


# to run the code: python -m streamlit run user_interface.py