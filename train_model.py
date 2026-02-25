import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
X_train = np.load("processed/X_train.npy")
X_test = np.load("processed/X_test.npy")
y_train = np.load("processed/y_train.npy")
y_test = np.load("processed/y_test.npy")

print("Data loaded successfully!")

# MODEL RINGAN (Production Friendly)
model = RandomForestClassifier(
    n_estimators=30,      # kecil
    max_depth=8,          # batasi kedalaman
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("Training model...")
model.fit(X_train, y_train)
print("Training selesai!")

# Evaluasi
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save compressed model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/ids_model.pkl", compress=3)

print("Model berhasil disimpan!")