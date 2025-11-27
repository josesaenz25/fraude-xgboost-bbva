# Generating Streamlit app for fraud detection with XGBoost
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
import streamlit as st

# Load data
df = pd.read_csv("transactions_full.csv")

# Load model
model = joblib.load("model.joblib")

# Title and description
st.set_page_config(page_title="Fraude Bancario", layout="centered")
st.title("🛡️ Detección de Fraude Bancario con XGBoost")
st.write("Esta aplicación permite simular transacciones y predecir la probabilidad de fraude en tiempo real.")

# Show data
st.subheader("📊 Primeros registros del dataset")
st.dataframe(df.head())

# Input fields
st.subheader("✍️ Simula una transacción")
amount = st.number_input("Monto de la transacción", min_value=0.0, value=1000.0)
hour = st.slider("Hora del día", min_value=0, max_value=23, value=12)
channel = st.selectbox("Canal", options=["APP", "WEB", "CAJERO", "CALL_CENTER"])
device = st.selectbox("Dispositivo", options=["Android", "iOS", "Desktop", "ATM"])
risk_cluster = st.selectbox("Cluster de riesgo", options=["Low", "Medium", "High"])

# Encode inputs
input_dict = {
    "amount": amount,
    "hour": hour,
    "is_night": int(hour >= 0 and hour <= 4),
    "channel_APP": int(channel == "APP"),
    "channel_WEB": int(channel == "WEB"),
    "channel_CAJERO": int(channel == "CAJERO"),
    "channel_CALL_CENTER": int(channel == "CALL_CENTER"),
    "device_Android": int(device == "Android"),
    "device_iOS": int(device == "iOS"),
    "device_Desktop": int(device == "Desktop"),
    "device_ATM": int(device == "ATM"),
    "risk_cluster_Low": int(risk_cluster == "Low"),
    "risk_cluster_Medium": int(risk_cluster == "Medium"),
    "risk_cluster_High": int(risk_cluster == "High")
}
X_input = pd.DataFrame([input_dict])

# Predict
proba = model.predict_proba(X_input)[0][1]
st.subheader("🔍 Resultado de la predicción")
st.metric(label="Probabilidad de fraude", value=f"{proba:.2%}")

# Optional: show confusion matrix and PR curve
if st.checkbox("Mostrar curva Precision-Recall y matriz de confusión"):
    y_test = pd.read_csv("/mnt/data/y_test.csv").values.ravel()
    X_test = pd.read_csv("/mnt/data/X_test.csv")
    y_proba = np.load("/mnt/data/test_proba.npy")

    # Confusion matrix
    y_pred = (y_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    st.write("📌 Matriz de confusión")
    st.write(pd.DataFrame(cm, index=["No Fraude", "Fraude"], columns=["Pred No Fraude", "Pred Fraude"]))

    # PR Curve
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    fig, ax = plt.subplots()
    ax.plot(rec, prec, label=f"AP={ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curva Precision-Recall")
    ax.legend()
    st.pyplot(fig)

print("Streamlit app generada como app interactiva para detección de fraude con XGBoost")
