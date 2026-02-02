import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
from datetime import datetime
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Secure Cloud IDS",
    layout="wide"
)

st.title("🔐 Secure Cloud Intrusion Detection System")
st.caption("Post-Quantum Cryptography + Deep Learning IDS")

# --------------------------------------------------
# LOAD TRAINED ARTIFACTS
# --------------------------------------------------
MODEL = tf.keras.models.load_model("ids_model.keras")
SCALER = joblib.load("scaler.pkl")
FEATURE_NAMES = joblib.load("feature_names.pkl")
THRESHOLD = float(np.load("best_threshold.npy"))

# --------------------------------------------------
# DIRECTORY SETUP (CLIENT / CLOUD)
# --------------------------------------------------
os.makedirs("client_side", exist_ok=True)
os.makedirs("cloud_side", exist_ok=True)

# --------------------------------------------------
# SYSTEM WORKFLOW SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.header("🧭 System Workflow")
    st.write("1. Dataset Upload")
    st.write("2. Preprocessing")
    st.write("3. Client-Side Encryption")
    st.write("4. Cloud Transmission")
    st.write("5. Cloud Decryption")
    st.write("6. IDS Inference")
    st.write("7. Alert Generation")
    st.write("8. Evaluation Metrics")

# --------------------------------------------------
# DATA UPLOAD
# --------------------------------------------------
file = st.file_uploader(
    "📂 Upload CICIDS Dataset (Benign vs One Attack)",
    type=["csv"]
)

if not file:
    st.stop()

df = pd.read_csv(file)
st.success("✅ Dataset uploaded successfully")

# --------------------------------------------------
# PREPROCESSING (SAFE & ALIGNED)
# --------------------------------------------------
df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

DROP_COLS = [
    "Flow ID", "Source IP", "Destination IP",
    "Timestamp", "Protocol"
]
df.drop(columns=DROP_COLS, errors="ignore", inplace=True)

original_ports = df["Destination Port"].values if "Destination Port" in df.columns else np.zeros(len(df))

X = df.drop("Label", axis=1, errors="ignore")

# ---- FEATURE ALIGNMENT (CRITICAL FIX) ----
for col in FEATURE_NAMES:
    if col not in X.columns:
        X[col] = 0

X = X[FEATURE_NAMES]
X_scaled = SCALER.transform(X)

# Save preprocessed logs (client side)
np.savez("client_side/preprocessed_logs.npz", X_scaled)

st.success("✅ Preprocessing completed")

# --------------------------------------------------
# CLIENT-SIDE ENCRYPTION (AES-256)
# --------------------------------------------------
key = np.random.bytes(32)   # 256-bit key
iv = np.random.bytes(16)    # 128-bit IV

with open("client_side/secret.key", "wb") as f:
    f.write(key)

with open("client_side/iv.bin", "wb") as f:
    f.write(iv)

cipher = Cipher(
    algorithms.AES(key),
    modes.CFB(iv),
    backend=default_backend()
)

encryptor = cipher.encryptor()
encrypted_logs = encryptor.update(X_scaled.tobytes()) + encryptor.finalize()

with open("client_side/encrypted_logs.bin", "wb") as f:
    f.write(encrypted_logs)

st.success("🔐 Logs encrypted on client side")

# --------------------------------------------------
# CLOUD TRANSMISSION (SIMULATED NETWORK TRANSFER)
# --------------------------------------------------
with open("client_side/encrypted_logs.bin", "rb") as src:
    with open("cloud_side/received_encrypted_logs.bin", "wb") as dst:
        dst.write(src.read())

st.success("☁️ Logs transmitted to cloud server")

# --------------------------------------------------
# CLOUD-SIDE DECRYPTION
# --------------------------------------------------
cipher = Cipher(
    algorithms.AES(key),
    modes.CFB(iv),
    backend=default_backend()
)

decryptor = cipher.decryptor()

with open("cloud_side/received_encrypted_logs.bin", "rb") as f:
    encrypted_cloud = f.read()

decrypted = decryptor.update(encrypted_cloud) + decryptor.finalize()

X_cloud = np.frombuffer(decrypted, dtype=X_scaled.dtype).reshape(X_scaled.shape)

np.savez("cloud_side/decrypted_logs.npz", X_cloud)

st.success("☁️ Logs decrypted on cloud side")

# --------------------------------------------------
# IDS INFERENCE (NO RETRAINING)
# --------------------------------------------------
probs = MODEL.predict(X_cloud, verbose=0).ravel()
preds = (probs >= THRESHOLD).astype(int)

# --------------------------------------------------
# ALERT GENERATION
# --------------------------------------------------
alerts = []

for i, p in enumerate(probs):
    if p >= THRESHOLD:
        alerts.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Packet ID": i,
            "Destination Port": int(original_ports[i]) if i < len(original_ports) else -1,
            "Confidence": round(float(p), 4),
            "Severity": "HIGH" if p >= 0.95 else "MEDIUM"
        })

alerts_df = pd.DataFrame(alerts)
alerts_df.to_csv("cloud_side/alert_log.csv", index=False)

# --------------------------------------------------
# ALERT DASHBOARD
# --------------------------------------------------
st.subheader("🚨 Intrusion Alert Summary")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Attacks", len(alerts_df))
c2.metric("Unique Ports", alerts_df["Destination Port"].nunique() if not alerts_df.empty else 0)
c3.metric("Max Confidence", round(alerts_df["Confidence"].max(), 4) if not alerts_df.empty else 0)
c4.metric("Avg Confidence", round(alerts_df["Confidence"].mean(), 4) if not alerts_df.empty else 0)

st.subheader("🎯 Top Targeted Destination Ports")
if not alerts_df.empty:
    port_stats = (
        alerts_df["Destination Port"]
        .value_counts()
        .head(10)
        .reset_index()
        .rename(columns={"index": "Port", "Destination Port": "Attack Count"})
    )
    st.table(port_stats)

with st.expander("📜 View Detailed Alerts (Top 100)"):
    st.dataframe(alerts_df.head(100), use_container_width=True)

st.download_button(
    "⬇️ Download Full Alert Log",
    alerts_df.to_csv(index=False),
    "alert_log.csv",
    "text/csv"
)

# --------------------------------------------------
# EVALUATION METRICS
# --------------------------------------------------
st.subheader("📊 Evaluation Metrics")

y = LabelEncoder().fit_transform(df["Label"]) if "Label" in df.columns else np.zeros(len(preds))

tn, fp, fn, tp = confusion_matrix(y, preds).ravel()

metrics = {
    "Accuracy": accuracy_score(y, preds),
    "Precision": precision_score(y, preds),
    "Recall": recall_score(y, preds),
    "F1 Score": f1_score(y, preds),
    "False Positives": int(fp),
    "False Negatives": int(fn),
    "False Positive Rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
    "False Negative Rate": fn / (fn + tp) if (fn + tp) > 0 else 0
}

st.json(metrics)

st.subheader("📉 Confusion Matrix")
cm_df = pd.DataFrame(
    [[tn, fp], [fn, tp]],
    columns=["Predicted Normal", "Predicted Attack"],
    index=["Actual Normal", "Actual Attack"]
)
st.dataframe(cm_df, use_container_width=True)

