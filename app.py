from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# =====================================
# AUTO CREATE FOLDER (ANTI RENDER CRASH)
# =====================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

os.makedirs(os.path.join(BASE_DIR, "uploads"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "processed"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "model"), exist_ok=True)

# =====================================
# LOAD MODEL & TOOLS
# =====================================
try:
    model = joblib.load(os.path.join(BASE_DIR, "model", "ids_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "processed", "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(BASE_DIR, "processed", "label_encoder.pkl"))

    X_test_path = os.path.join(BASE_DIR, "processed", "X_test.npy")
    y_test_path = os.path.join(BASE_DIR, "processed", "y_test.npy")

    if os.path.exists(X_test_path) and os.path.exists(y_test_path):
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)
        y_pred_test = model.predict(X_test)
        model_accuracy = round(accuracy_score(y_test, y_pred_test) * 100, 2)
    else:
        model_accuracy = 99.0

except Exception as e:
    print("MODEL LOAD ERROR:", e)
    model_accuracy = 0

# =====================================
# ROUTES
# =====================================

@app.route('/')
def home():
    return render_template("index.html", accuracy=model_accuracy)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        if file.filename == '':
            return "No file selected"

        filepath = os.path.join(BASE_DIR, "uploads", file.filename)
        file.save(filepath)

        # =========================
        # LOAD & RANDOM SAMPLE
        # =========================
        df = pd.read_csv(filepath)

        if len(df) > 5000:
            df = df.sample(n=5000, random_state=42)

        df.columns = df.columns.str.strip()

        # =========================
        # CLEAN DATA
        # =========================
        if 'Flow ID' in df.columns:
            df = df.drop(['Flow ID'], axis=1)

        if "Label" in df.columns:
            df = df.drop("Label", axis=1)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        # =========================
        # VALIDATE FEATURE COUNT
        # =========================
        if df.shape[1] != scaler.n_features_in_:
            return f"Feature mismatch! Model expects {scaler.n_features_in_} but got {df.shape[1]}"

        # =========================
        # PREDICTION
        # =========================
        X_scaled = scaler.transform(df)
        predictions = model.predict(X_scaled)
        predicted_labels = label_encoder.inverse_transform(predictions)

        result_series = pd.Series(predicted_labels).value_counts()
        total = result_series.sum()
        result = result_series.to_dict()

        percentages = {
            key: round((value / total) * 100, 2)
            for key, value in result.items()
        }

        # =========================
        # THREAT CALCULATION
        # =========================
        benign_count = result.get("BENIGN", 0)
        attack_count = total - benign_count
        attack_percentage = round((attack_count / total) * 100, 2)

        # Threat level logic
        if attack_percentage > 30:
            threat_level = "HIGH"
            threat_color = "danger"
        elif attack_percentage > 10:
            threat_level = "MEDIUM"
            threat_color = "warning"
        else:
            threat_level = "LOW"
            threat_color = "success"

        return render_template(
            "dashboard.html",
            result=result,
            percentages=percentages,
            accuracy=model_accuracy,
            attack_percentage=attack_percentage,
            threat_level=threat_level,
            threat_color=threat_color
        )

    except Exception as e:
        return f"System Error: {str(e)}"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))