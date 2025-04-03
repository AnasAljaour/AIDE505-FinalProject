from flask import Flask, request, jsonify
import mlflow
import numpy as np
import dagshub
import os
import pickle
app = Flask(__name__)

def load_model():
    logged_model = 'runs:/e695ea0aa4344469907ac6085ce382f7/SVM (Linear)'
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    return loaded_model

FEATURE_NAMES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

DEFAULT_VALUES = {feature: 0.0 for feature in FEATURE_NAMES}

mapping = {0: "Benign", 1: "Malignant"}

mlflow.set_tracking_uri("https://dagshub.com/AnasAljaour/AIDE505-FinalProject.mlflow")


# dagshub.init(repo_owner='AnasAljaour', repo_name='AIDE505-FinalProject', mlflow=True)
model = load_model()
os.makedirs("models", exist_ok=True)
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)


@app.route("/predict", methods=["POST"])
def predict():
    global mapping, model
    try:
       
        data = request.get_json()

        if not isinstance(data, dict):
            return jsonify({"error": "Invalid JSON format. Expected a dictionary"}), 400

        
        processed_data = [data.get(feature, DEFAULT_VALUES[feature]) for feature in FEATURE_NAMES]
        input_data = np.array(processed_data).reshape(1, -1)
        with open('models/model.pkl', 'rb') as f:
            model = pickle.load(f)
        prediction = model.predict(input_data)
        prediction_value = int(prediction[0])
        return jsonify({"prediction": mapping.get(prediction_value, "Unknown")}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

