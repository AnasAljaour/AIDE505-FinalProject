import streamlit as st

import requests



API_URL = "http://Backend-Node:3000/cancer-diagnosis"

st.title("Breast Cancer Prediction App")

# Define feature names (ensure they match the trained model)
features = ["Perimeter_Worst","ConcavePoints_Worst","Area_Worst","Radius_Worst","Texture_Worst","Smoothness_Worst","Texture_Mean","Concavity_Worst"]

# User input form
form1 = st.form(key="features")
user_inputs = {}
invalid_inputs = False  # Flag to check for invalid inputs
for feature in features:
    label = feature.replace("_", " ")  # Format label by removing underscores
    user_input = form1.text_input(label, value="")  # Text input starts empty
    try:
        if user_input.strip() == "":
            user_inputs[feature] = None  # Mark empty inputs
        else:
            user_inputs[feature] = float(user_input)  # Convert to float
    except ValueError:
        invalid_inputs = True
        st.error(f"Invalid input for {label}: Please enter a valid number.")

if form1.form_submit_button("Predict"):
    if invalid_inputs:
        st.error("Fix invalid inputs before predicting.")
    elif None in user_inputs.values():
        st.error("Please fill in all the feature values before predicting.")
    else:
        input_data = {feature: user_inputs[feature] for feature in features}
        
        feature_request = requests.post(API_URL, json=input_data)

        try:
            response_json = feature_request.json()
            prediction = response_json.get("prediction", None)
        except Exception as e:
            st.error(f"Error decoding JSON: {e}")
            prediction = None
        
        st.write(f"### Prediction: {prediction}")

