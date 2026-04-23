# ============================================================
# app.py
# COMP 247 – Deliverable 5: Model Deployment
# Section 402  |  Group 5
#
# Flask API that loads best_model.pkl and serves predictions.
# Run with: python app.py
# Then open: http://127.0.0.1:5000
# ============================================================

import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# ============================================================
# LOAD MODEL
# ============================================================
MODEL_PATH = Path(__file__).resolve().parent / "best_model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print(f"[app.py] Model loaded from: {MODEL_PATH}")

# ============================================================
# FEATURE DEFINITIONS
# Match exactly what ksi_shared.py sends to the model
# ============================================================
CATEGORICAL_FEATURES = [
    "OCC_MONTH", "OCC_DOW", "DIVISION", "HOOD_158",
    "AUTOMOBILE", "MOTORCYCLE", "PASSENGER", "BICYCLE", "PEDESTRIAN",
]
NUMERICAL_FEATURES = [
    "OCC_YEAR", "OCC_HOUR", "LONG_WGS84", "LAT_WGS84", "IS_WEEKEND",
]

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DIVISIONS = [
    "D11", "D12", "D13", "D14", "D22", "D23", "D31", "D32",
    "D33", "D41", "D42", "D43", "D51", "D52", "D53", "D55", "NSA"
]
YES_NO = ["YES", "NO"]


# ============================================================
# ROUTES
# ============================================================
@app.route("/")
def index():
    """Render the prediction form."""
    return render_template(
        "index.html",
        months=MONTHS,
        days=DAYS,
        divisions=DIVISIONS,
        yes_no=YES_NO,
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept form data or JSON, run model prediction, return result.
    Supports both HTML form submission and Postman/API JSON calls.
    """
    try:
        # Support both JSON (Postman/API) and form (HTML frontend)
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        # Build a single-row DataFrame matching the training features
        row = {
            # Categorical
            "OCC_MONTH":  data.get("OCC_MONTH", "January"),
            "OCC_DOW":    data.get("OCC_DOW",   "Monday"),
            "DIVISION":   data.get("DIVISION",  "D11"),
            "HOOD_158":   data.get("HOOD_158",  "1"),
            "AUTOMOBILE": data.get("AUTOMOBILE","YES"),
            "MOTORCYCLE": data.get("MOTORCYCLE","NO"),
            "PASSENGER":  data.get("PASSENGER", "NO"),
            "BICYCLE":    data.get("BICYCLE",   "NO"),
            "PEDESTRIAN": data.get("PEDESTRIAN","NO"),
            # Numerical
            "OCC_YEAR":   float(data.get("OCC_YEAR",  2024)),
            "OCC_HOUR":   float(data.get("OCC_HOUR",  12)),
            "LONG_WGS84": float(data.get("LONG_WGS84",-79.3832)),
            "LAT_WGS84":  float(data.get("LAT_WGS84",  43.6532)),
            "IS_WEEKEND":  1 if data.get("OCC_DOW", "Monday") in ["Saturday", "Sunday"] else 0,
        }

        input_df = pd.DataFrame([row])

        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])

        result = {
            "prediction": prediction,
            "label": "FATAL" if prediction == 1 else "NON-FATAL",
            "probability": round(probability * 100, 2),
            "message": (
                "⚠️ High risk — this collision is predicted to be FATAL."
                if prediction == 1
                else "✅ This collision is predicted to be NON-FATAL."
            ),
        }

        # Return JSON for API calls, render result page for form submissions
        if request.is_json:
            return jsonify(result)
        else:
            return render_template(
                "index.html",
                months=MONTHS,
                days=DAYS,
                divisions=DIVISIONS,
                yes_no=YES_NO,
                result=result,
                form_data=data,
            )

    except Exception as e:
        error = {"error": str(e)}
        if request.is_json:
            return jsonify(error), 400
        return render_template(
            "index.html",
            months=MONTHS,
            days=DAYS,
            divisions=DIVISIONS,
            yes_no=YES_NO,
            error=str(e),
        )


@app.route("/health")
def health():
    """Health check endpoint — useful for Postman testing."""
    return jsonify({"status": "ok", "model": str(MODEL_PATH.name)})


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("  KSI Collision Predictor — Flask API")
    print("  Open: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
