import streamlit as st
import mlflow.pyfunc
import numpy as np
import pandas as pd

# MLflow model URI
logged_model = 'runs:/76119866719d4b18a5666dbeb7e57e00/RandomForestClassifier'

# Load the model as a PyFuncModel
loaded_model = mlflow.pyfunc.load_model(logged_model)

st.title("Breast Cancer Prediction App")

# Define feature names (ensure they match the trained model)
features = ["perimeter3", "concave_points3", "area3", "radius3", "texture3", "smoothness3", "texture1", "concavity3"]

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

    
    prediction = loaded_model.predict(full_input)
    
    result = "Malignant" if prediction[0] == 1 else "Benign"
    st.write(f"### Prediction: {result}")


# to run the code: python -m streamlit run user_interface.py