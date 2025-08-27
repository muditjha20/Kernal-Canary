from flask import Flask, request, jsonify
import numpy as np
import joblib
import json
from pathlib import Path

from flask_cors import CORS

app = Flask(__name__)

CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    allow_headers=["Content-Type"],
    methods=["GET", "POST", "OPTIONS"]
)

# === Load model and params
ROOT = Path(__file__).resolve().parent
model = joblib.load(ROOT / "model/model.pkl")

with open(ROOT / "artifacts/best_model_params.json") as f:
    params = json.load(f)

# Compute threshold
X = np.load(ROOT / "data/X.npy")
scores = model.decision_function(X)
threshold = np.percentile(scores, params["contamination"] * 100)

@app.route("/score", methods=["POST"])
def score():
    data = request.get_json()
    print("Received JSON:", data)

    if not data or "window" not in data:
        return jsonify({"error": "Missing 'window' in request"}), 400

    window = np.array(data["window"]).reshape(1, -1)
    prediction = model.predict(window)[0]  # -1 = anomaly, 1 = normal

    is_anomaly = prediction == -1

    return jsonify({
        "is_anomaly": bool(is_anomaly)
    })


@app.route("/")
def index():
    return "Kernel Canary++ API is running."

if __name__ == "__main__":
    app.run(debug=True, port=5000)
