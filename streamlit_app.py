import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import shutil
import time
from datetime import datetime
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Secure Cloud IDS", layout="wide")
st.title("🔐 Secure Cloud Intrusion Detection System")
st.caption("Real File-Based Client → Cloud Transmission")

# --------------------------------------------------
# DIRECTORY SETUP (REAL)
# --------------------------------------------------
os.makedirs("client_side", exist_ok=True)
os.makedirs("cloud_side", exist_ok=True)
os.makedirs("model", exist_ok=True)

# --------------------------------------------------
# LOAD MODEL ARTIFACTS
# --------------------------------------------------
MODEL = tf.keras.models.load_model("ids_model.keras")
SCALER = joblib.load("scaler.pkl")
THRESHOLD = float(np.load("best_threshold.npy"))

# --------------------------------------------------
# DATASET UPLOAD (CLIENT SIDE)
# --------------------------------------------------
st.subheader("📂 Client Side: Dataset Upload")

file = st.file_uploader("Upload CICIDS Dataset (Binary Class)", type=["csv"])

if not file:
    st.stop()

client_csv = "client_side/raw_logs.csv"
with open(client_csv, "wb") as f:
    f.write(file.getbuffer())

st.success("✅ Dataset saved to client_side/raw_logs.csv")

# --------------------------------------------------
# CLIENT SIDE: PREPROCESSING
# --------------------------------------------------
st.subheader("⚙️ Client Side: Preprocessing")

df = pd.read_csv(client_csv)
df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

drop_cols = ["Flow ID", "Source IP", "Destination IP", "Timestamp", "Protocol"]
df.drop(columns=drop_cols, errors="ignore", inplace=True)

original_ports = df["Destination Port"].values

X = df.drop("Label", axis=1)
y = LabelEncoder().fit_transform(df["Label"])

X_scaled = SCALER.transform(X)

np.savez(
    "client_side/preprocessed_logs.npz",
    X=X_scaled,
    y=y,
    original_ports=original_ports
)

st.success("✅ Preprocessed data saved (client_side/preprocessed_logs.npz)")

# --------------------------------------------------
# CLIENT SIDE: ENCRYPTION (REAL FILES)
# --------------------------------------------------
st.subheader("🔐 Client Side: Encryption")

data_bytes = open("client_side/preprocessed_logs.npz", "rb").read()

key = os.urandom(32)   # AES-256
iv = os.urandom(16)

cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
encryptor = cipher.encryptor()
encrypted = encryptor.update(data_bytes) + encryptor.finalize()

open("client_side/encrypted_logs.bin", "wb").write(encrypted)
open("client_side/secret.key", "wb").write(key)
open("client_side/iv.bin", "wb").write(iv)

st.success("🔐 Encrypted files created on client side")

# --------------------------------------------------
# REAL TRANSMISSION (FILE COPY)
# --------------------------------------------------
st.subheader("📡 Secure Transmission: Client → Cloud")

progress = st.progress(0)
for i in range(100):
    time.sleep(0.01)
    progress.progress(i + 1)

shutil.copy(
    "client_side/encrypted_logs.bin",
    "cloud_side/received_encrypted_logs.bin"
)

st.success("☁️ Encrypted logs transmitted to cloud_side/")

# --------------------------------------------------
# CLOUD SIDE: DECRYPTION
# --------------------------------------------------
st.subheader("☁️ Cloud Side: Decryption")

encrypted_cloud = open("cloud_side/received_encrypted_logs.bin", "rb").read()
key = open("client_side/secret.key", "rb").read()
iv = open("client_side/iv.bin", "rb").read()

cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
decryptor = cipher.decryptor()
decrypted = decryptor.update(encrypted_cloud) + decryptor.finalize()

open("cloud_side/decrypted_logs.npz", "wb").write(decrypted)

st.success("🔓 Logs decrypted on cloud side")

# --------------------------------------------------
# IDS INFERENCE (CLOUD ONLY)
# --------------------------------------------------
st.subheader("🧠 Cloud Side: IDS Deep Learning Analysis")

data = np.load("cloud_side/decrypted_logs.npz")
X_cloud = data["X"]
ports_cloud = data["original_ports"]

probs = MODEL.predict(X_cloud, verbose=0).ravel()
preds = (probs >= THRESHOLD).astype(int)

st.success("✅ Intrusion detection completed")

# --------------------------------------------------
# ALERT GENERATION (REAL CSV)
# --------------------------------------------------
alerts = []

for i, p in enumerate(probs):
    if p >= THRESHOLD:
        alerts.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Packet ID": i,
            "Destination Port": int(ports_cloud[i]),
            "Confidence": round(float(p), 4),
            "Severity": "HIGH" if p >= 0.95 else "MEDIUM"
        })

alerts_df = pd.DataFrame(alerts)
alerts_df.to_csv("cloud_side/alert_log.csv", index=False)

st.success("🚨 Alerts generated (cloud_side/alert_log.csv)")

# --------------------------------------------------
# ALERT DASHBOARD
# --------------------------------------------------
st.subheader("🚨 Alert Summary")

c1, c2, c3 = st.columns(3)
c1.metric("Total Attacks", len(alerts_df))
c2.metric("Unique Ports", alerts_df["Destination Port"].nunique())
c3.metric("Max Confidence", alerts_df["Confidence"].max())

with st.expander("📜 View Alerts (First 50)"):
    st.dataframe(alerts_df.head(50), use_container_width=True)

st.download_button(
    "⬇️ Download Full Alert Log",
    alerts_df.to_csv(index=False),
    "alert_log.csv",
    "text/csv"
)

# --------------------------------------------------
# METRICS & CONFUSION MATRIX
# --------------------------------------------------
st.subheader("📊 Evaluation Metrics")

tn, fp, fn, tp = confusion_matrix(y, preds).ravel()

metrics = {
    "Accuracy": accuracy_score(y, preds),
    "Precision": precision_score(y, preds),
    "Recall": recall_score(y, preds),
    "F1 Score": f1_score(y, preds),
    "False Positives": int(fp),
    "False Negatives": int(fn)
}

st.json(metrics)

st.subheader("📉 Confusion Matrix")

cm_df = pd.DataFrame(
    [[tn, fp], [fn, tp]],
    columns=["Predicted Normal", "Predicted Attack"],
    index=["Actual Normal", "Actual Attack"]
)

st.dataframe(cm_df, use_container_width=True)
