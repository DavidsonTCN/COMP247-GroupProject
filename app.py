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
# DEMO THRESHOLD
# Lower this if you want FATAL to appear more easily
# 0.20 = 20%
# ============================================================
FATAL_THRESHOLD = 0.20

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


def safe_float(value, default):
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


# ============================================================
# ROUTES
# ============================================================
@app.route("/")
def index():
    return render_template(
        "index.html",
        months=MONTHS,
        days=DAYS,
        divisions=DIVISIONS,
        yes_no=YES_NO,
        threshold=round(FATAL_THRESHOLD * 100, 2),
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.is_json:
            data = request.get_json() or {}
        else:
            data = request.form.to_dict()

        occ_dow = data.get("OCC_DOW", "Monday")

        row = {
            "OCC_MONTH":  data.get("OCC_MONTH", "January"),
            "OCC_DOW":    occ_dow,
            "DIVISION":   data.get("DIVISION", "D11"),
            "HOOD_158":   str(data.get("HOOD_158", "1")),
            "AUTOMOBILE": data.get("AUTOMOBILE", "YES"),
            "MOTORCYCLE": data.get("MOTORCYCLE", "NO"),
            "PASSENGER":  data.get("PASSENGER", "NO"),
            "BICYCLE":    data.get("BICYCLE", "NO"),
            "PEDESTRIAN": data.get("PEDESTRIAN", "NO"),
            "OCC_YEAR":   safe_float(data.get("OCC_YEAR", 2024), 2024),
            "OCC_HOUR":   safe_float(data.get("OCC_HOUR", 12), 12),
            "LONG_WGS84": safe_float(data.get("LONG_WGS84", -79.3832), -79.3832),
            "LAT_WGS84":  safe_float(data.get("LAT_WGS84", 43.6532), 43.6532),
            "IS_WEEKEND": 1 if occ_dow in ["Saturday", "Sunday"] else 0,
        }

        input_df = pd.DataFrame([row])

        probability = float(model.predict_proba(input_df)[0][1])
        prediction = 1 if probability >= FATAL_THRESHOLD else 0

        result = {
            "prediction": prediction,
            "label": "FATAL" if prediction == 1 else "NON-FATAL",
            "probability": round(probability * 100, 2),
            "threshold_used": round(FATAL_THRESHOLD * 100, 2),
            "message": (
                "⚠️ High risk — this collision is predicted to be FATAL."
                if prediction == 1
                else "✅ This collision is predicted to be NON-FATAL."
            ),
        }

        if request.is_json:
            return jsonify(result)
        else:
            return render_template(
                "index.html",
                months=MONTHS,
                days=DAYS,
                divisions=DIVISIONS,
                yes_no=YES_NO,
                threshold=round(FATAL_THRESHOLD * 100, 2),
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
            threshold=round(FATAL_THRESHOLD * 100, 2),
            error=str(e),
        )


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": str(MODEL_PATH.name),
        "fatal_threshold_percent": round(FATAL_THRESHOLD * 100, 2),
    })


if __name__ == "__main__":
    print("=" * 50)
    print("  KSI Collision Predictor — Flask API")
    print("  Open: http://127.0.0.1:5000")
    print(f"  Fatal threshold: {FATAL_THRESHOLD:.2f}")
    print("=" * 50)
    app.run(debug=True, port=5000)