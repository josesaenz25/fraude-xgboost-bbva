import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

st.title("🚨 Detección de Fraude - Demo Interactiva")

# -----------------------------
# Celda 0: Verificación del entorno
# -----------------------------
import sklearn
st.write("pandas:", pd.__version__)
st.write("numpy:", np.__version__)
st.write("scikit-learn:", sklearn.__version__)
st.write("xgboost:", xgb.__version__)

# -----------------------------
# Celda 1: Generación de datos ficticios
# -----------------------------
np.random.seed(42)
n = 100  # número de registros

df = pd.DataFrame({
    "amount": np.random.randint(10, 1000, n),
    "channel": np.random.choice(["WEB","ATM","POS"], n),
    "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
    "fraud": np.random.choice([0,1], n, p=[0.9,0.1])
})

df.rename(columns={"fraud": "is_fraud"}, inplace=True)
df["is_fraud"] = df["is_fraud"].astype(int)

st.subheader("📊 Base ficticia generada")
st.dataframe(df.head())

# -----------------------------
# Celda 2: Agregados por hora y canal
# -----------------------------
df["hour_bucket"] = df["timestamp"].dt.floor("h")
df["fraud_flag_int"] = df["is_fraud"].astype(int)

agg = (
    df.groupby(["hour_bucket", "channel"])
      .agg(tx_count=("fraud_flag_int", "count"),
           fraud_count=("fraud_flag_int", "sum"))
      .reset_index()
)
agg["fraud_rate"] = agg["fraud_count"] / agg["tx_count"]

st.subheader("📊 Agregados por hora y canal")
st.dataframe(agg.head())

# -----------------------------
# Celda 3: Gráfica de tasa de fraude
# -----------------------------
fig, ax = plt.subplots(figsize=(12,6))
sns.lineplot(data=agg, x="hour_bucket", y="fraud_rate", hue="channel", marker="o", ax=ax)
plt.xticks(rotation=45)
ax.set_ylabel("Fraud Rate")
st.pyplot(fig)

# -----------------------------
# Celda 4: Colas M/M/1 y M/M/c
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
# Celda 5: Reglas de asociación
# -----------------------------
df["is_high_amount"] = df["amount"] > 1000
df["is_night"] = df["timestamp"].dt.hour.isin([0,1,2,3,4,23])
channels = pd.get_dummies(df["channel"], prefix="channel")

basket = pd.concat([df[["is_fraud","is_high_amount","is_night"]], channels], axis=1).astype(bool)

freq = apriori(basket, min_support=0.05, use_colnames=True)
rules = association_rules(freq, metric="lift", min_threshold=1.0)
top_rules = rules.sort_values("lift", ascending=False).head(10)

st.subheader("📊 Top 10 reglas de asociación")
st.dataframe(top_rules[["antecedents","consequents","support","confidence","lift"]])

# -----------------------------
# Celda 6: Entrenamiento de modelos
# -----------------------------
df["hour"] = df["timestamp"].dt.hour
X = df[["amount", "hour"]]
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

xgb_model = xgb.XGBClassifier(eval_metric="logloss", random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

st.subheader("✅ Modelos entrenados y predicciones generadas")
st.write("📊 Primeras predicciones RandomForest:", rf_pred[:10])
st.write("📊 Primeras predicciones XGBoost:", xgb_pred[:10])

st.write("🔎 Importancia de variables RandomForest:")
for feature, importance in zip(X.columns, rf.feature_importances_):
    st.write(f" - {feature}: {importance:.4f}")

st.write("🔎 Importancia de variables XGBoost:")
for feature, importance in zip(X.columns, xgb_model.feature_importances_):
    st.write(f" - {feature}: {importance:.4f}")
