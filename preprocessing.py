import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

# 1️⃣ Load semua file CSV dalam folder dataset
path = "dataset/*.csv"
files = glob.glob(path)

df_list = []

for file in files:
    print("Loading:", file)
    data = pd.read_csv(file, nrows=200000)
    df_list.append(data)

# Gabungkan semua file menjadi satu dataframe
df = pd.concat(df_list, ignore_index=True)

print("Total data sebelum cleaning:", df.shape)

# 2️⃣ Hapus spasi pada nama kolom
df.columns = df.columns.str.strip()

# 3️⃣ Hapus kolom yang tidak diperlukan
if 'Flow ID' in df.columns:
    df = df.drop(['Flow ID'], axis=1)

# 4️⃣ Ganti infinite value
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# 5️⃣ Hapus baris yang memiliki NaN
df.dropna(inplace=True)

print("Total data setelah cleaning:", df.shape)

# 6️⃣ Pisahkan fitur dan label
X = df.drop("Label", axis=1)
y = df["Label"]

# 7️⃣ Encode label (serangan jadi angka)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 8️⃣ Normalisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 9️⃣ Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# 🔟 Simpan hasil preprocessing
os.makedirs("processed", exist_ok=True)

np.save("processed/X_train.npy", X_train)
np.save("processed/X_test.npy", X_test)
np.save("processed/y_train.npy", y_train)
np.save("processed/y_test.npy", y_test)

# Simpan label encoder & scaler
import joblib
joblib.dump(le, "processed/label_encoder.pkl")
joblib.dump(scaler, "processed/scaler.pkl")

print("Preprocessing selesai dan data tersimpan!")