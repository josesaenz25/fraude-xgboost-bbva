import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc
from mlxtend.frequent_patterns import apriori, association_rules
import shap
import plotly.express as px

st.title("🚨 Detección de Fraude - Demo Interactiva")

# -----------------------------
# Celda 1: Generación de datos
# -----------------------------
np.random.seed(42)
n = 200
df = pd.DataFrame({
    "amount": np.random.randint(10, 1000, n),
    "channel": np.random.choice(["WEB","ATM","POS"], n),
    "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
    "is_fraud": np.random.choice([0,1], n, p=[0.9,0.1])
})

st.subheader("📊 Datos originales")
st.dataframe(df.head())

# -----------------------------
# Celda 2: Agregados
# -----------------------------
df["hour_bucket"] = df["timestamp"].dt.floor("h")
agg = (
    df.groupby(["hour_bucket","channel"])
      .agg(tx_count=("is_fraud","count"),
           fraud_count=("is_fraud","sum"))
      .reset_index()
)
agg["fraud_rate"] = agg["fraud_count"]/agg["tx_count"]

st.subheader("📊 Agregados por hora y canal")
st.dataframe(agg.head())

# -----------------------------
# Celda 3: Gráfica fraude
# -----------------------------
fig, ax = plt.subplots(figsize=(12,6))
sns.lineplot(data=agg, x="hour_bucket", y="fraud_rate", hue="channel", marker="o", ax=ax)
plt.xticks(rotation=45)
ax.set_ylabel("Fraud Rate")
st.pyplot(fig)

# -----------------------------
# Celda 5: Colas M/M/1 y M/M/c
# -----------------------------
def mm1_metrics(lmbda, mu):
    rho = lmbda / mu
    if rho >= 1:
        return {"Modelo": "M/M/1", "Estable": False, "Rho": float(rho)}
    Lq = (rho**2) / (1 - rho)
    Wq = Lq / lmbda
    W = Wq + (1 / mu)
    L = lmbda * W
    return {"Modelo": "M/M/1", "Estable": True, "Rho": float(rho),
            "Lq": Lq, "Wq_min": Wq, "W_min": W, "L": L}

def mmc_metrics(lmbda, mu, servers):
    a = lmbda / mu
    rho = a / servers
    if rho >= 1:
        return {"Modelo": "M/M/c", "Estable": False, "Rho": float(rho)}
    sum_terms = sum((a**n) / math.factorial(n) for n in range(servers))
    P0 = 1.0 / (sum_terms + (a**servers) / (math.factorial(servers) * (1 - rho)))
    Pc = ((a**servers) / math.factorial(servers)) * (P0 / (1 - rho))
    Wq = Pc * (1 / mu) * (1 / (servers - a))
    W = Wq + (1 / mu)
    Lq = lmbda * Wq
    L = lmbda * W
    return {"Modelo": "M/M/c", "Estable": True, "Rho": float(rho),
            "P_espera": Pc, "Lq": Lq, "Wq_min": Wq, "W_min": W, "L": L}

lambda_h = df["is_fraud"].sum()
mu_h = 12
c = 5

mm1 = mm1_metrics(lambda_h/60, mu_h/60)
mmc = mmc_metrics(lambda_h/60, mu_h/60, c)

df_queues = pd.DataFrame([mm1, mmc])
st.subheader("📊 Resultados de modelos de colas")
st.dataframe(df_queues)

# -----------------------------
# Celda 6: Modelos ML
# -----------------------------
df["hour"] = df["timestamp"].dt.hour
X = df[["amount","hour"]]
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train,y_train)
rf_pred = rf.predict(X_test)

xgb_model = xgb.XGBClassifier(eval_metric="logloss", random_state=42)
xgb_model.fit(X_train,y_train)
xgb_pred = xgb_model.predict(X_test)

rf_report = pd.DataFrame(classification_report(y_test, rf_pred, output_dict=True)).T
xgb_report = pd.DataFrame(classification_report(y_test, xgb_pred, output_dict=True)).T

st.subheader("📊 Reporte RandomForest")
st.dataframe(rf_report)

st.subheader("📊 Reporte XGBoost")
st.dataframe(xgb_report)

# -----------------------------
# Celda 9: ROC
# -----------------------------
rf_prob = rf.predict_proba(X_test)[:,1]
xgb_prob = xgb_model.predict_proba(X_test)[:,1]

rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_prob)

rf_auc = auc(rf_fpr, rf_tpr)
xgb_auc = auc(xgb_fpr, xgb_tpr)

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(rf_fpr, rf_tpr, label=f"RandomForest (AUC={rf_auc:.2f})")
ax.plot(xgb_fpr, xgb_tpr, label=f"XGBoost (AUC={xgb_auc:.2f})")
ax.plot([0,1],[0,1],"k--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
st.pyplot(fig)

# -----------------------------
# Celda 11: Plotly interactivo
# -----------------------------
fig = px.line(agg, x="hour_bucket", y="fraud_rate", color="channel",
              title="Tasa de fraude por hora y canal (interactivo)")
st.plotly_chart(fig)

# -----------------------------
# Celda 13: Narrativa ejecutiva
# -----------------------------
summary_text = f"""
### 📑 Resumen Ejecutivo
- Se analizaron {len(df)} transacciones ficticias.
- La tasa de fraude promedio fue {df['is_fraud'].mean():.2%}.
- RandomForest F1={rf_report.loc['weighted avg','f1-score']:.2f}.
- XGBoost F1={xgb_report.loc['weighted avg','f1-score']:.2f}.
"""
st.markdown(summary_text)
