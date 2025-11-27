#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Celda 0: Verificación del entorno

import pandas as pd, numpy as np, sklearn, xgboost
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("scikit-learn:", sklearn.__version__)
print("xgboost:", xgboost.__version__)


# In[2]:


# Celda 1: Generación de datos ficticios

import numpy as np
import pandas as pd

np.random.seed(42)
n = 100  # número de registros

df = pd.DataFrame({
    "amount": np.random.randint(10, 1000, n),
    "channel": np.random.choice(["WEB","ATM","POS"], n),
    "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
    "fraud": np.random.choice([0,1], n, p=[0.9,0.1])
})

# Renombrar columna para consistencia
df.rename(columns={"fraud": "is_fraud"}, inplace=True)
df["is_fraud"] = df["is_fraud"].astype(int)

print("📊 Base ficticia generada:")
display(df.head())


# In[3]:


# Celda 2: Agregados por hora y canal

df["hour_bucket"] = df["timestamp"].dt.floor("h")
df["fraud_flag_int"] = df["is_fraud"].astype(int)

agg = (
    df.groupby(["hour_bucket", "channel"])
      .agg(tx_count=("fraud_flag_int", "count"),
           fraud_count=("fraud_flag_int", "sum"))
      .reset_index()
)

# Calcular tasa de fraude
agg["fraud_rate"] = agg["fraud_count"] / agg["tx_count"]

print("📊 Agregados por hora y canal:")
display(agg.head())


# In[4]:


# Celda 3: Gráfica de tasa de fraude

import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")
plt.figure(figsize=(12,6))

sns.lineplot(data=agg, x="hour_bucket", y="fraud_rate", hue="channel", marker="o")
plt.title("Tasa de fraude por hora y canal")
plt.xticks(rotation=45)
plt.ylabel("Fraud Rate")
plt.show()


# In[5]:


# Celda 4: Colas M/M/1 y M/M/c
import math

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
print("📊 Resultados de modelos de colas:")
display(df_queues)


# In[6]:


# Celda 5: Reglas de asociación corregida
from mlxtend.frequent_patterns import apriori, association_rules

# Indicadores binarios
df["is_high_amount"] = df["amount"] > 1000
df["is_night"] = df["timestamp"].dt.hour.isin([0,1,2,3,4,23])

# One-hot de canales
channels = pd.get_dummies(df["channel"], prefix="channel")

# Matriz booleana
basket = pd.concat([df[["is_fraud","is_high_amount","is_night"]], channels], axis=1).astype(bool)

# Itemsets frecuentes con soporte más bajo (ej. 0.05 = 5%)
freq = apriori(basket, min_support=0.05, use_colnames=True)

# Reglas de asociación
rules = association_rules(freq, metric="lift", min_threshold=1.0)

# Top reglas
top_rules = rules.sort_values("lift", ascending=False).head(10)

print("📊 Top 10 reglas de asociación (con soporte mínimo 5%):")
display(top_rules[["antecedents","consequents","support","confidence","lift"]])


# In[7]:


# Celda 6: Entrenamiento de modelos corregida y extendida

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Features y target
df["hour"] = df["timestamp"].dt.hour  # crear columna 'hour' a partir de timestamp
X = df[["amount", "hour"]]
y = df["is_fraud"]

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Entrenar RandomForest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Entrenar XGBoost
xgb_model = xgb.XGBClassifier(
    eval_metric="logloss",
    random_state=42
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

# Mostrar resultados
print("✅ Modelos entrenados y predicciones generadas")
print("\n📊 Primeras predicciones RandomForest:", rf_pred[:10])
print("📊 Primeras predicciones XGBoost:", xgb_pred[:10])

print("\n🔎 Importancia de variables RandomForest:")
for feature, importance in zip(X.columns, rf.feature_importances_):
    print(f" - {feature}: {importance:.4f}")

print("\n🔎 Importancia de variables XGBoost:")
for feature, importance in zip(X.columns, xgb_model.feature_importances_):
    print(f" - {feature}: {importance:.4f}")



# In[8]:


# Celda 7: Evaluación y visualizaciones corregida

from sklearn.metrics import classification_report
import pandas as pd

# Guardar reportes como DataFrame para poder usarlos después
rf_report = pd.DataFrame(
    classification_report(y_test, rf_pred, zero_division=0, output_dict=True)
).T

xgb_report = pd.DataFrame(
    classification_report(y_test, xgb_pred, zero_division=0, output_dict=True)
).T

print("📊 Reporte RandomForest:")
display(rf_report)

print("📊 Reporte XGBoost:")
display(xgb_report)



# In[9]:


# Celda 8: comparación de métricas y curva ROC

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 1) Resumen de métricas (usa rf_report y xgb_report creados en Celda 7)
rf_sum = rf_report.loc["weighted avg", ["precision", "recall", "f1-score"]]
xgb_sum = xgb_report.loc["weighted avg", ["precision", "recall", "f1-score"]]

metrics_df = pd.DataFrame({
    "RandomForest": rf_sum,
    "XGBoost": xgb_sum
}).T

print("📊 Comparación de métricas (Weighted Avg):")
display(metrics_df)

# 2) Curva ROC comparativa (asegúrate de tener rf y xgb_model de Celda 6)
rf_prob = rf.predict_proba(X_test)[:, 1]
xgb_prob = xgb_model.predict_proba(X_test)[:, 1]  # ✅ usar el modelo, no el módulo

rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_prob)

rf_auc = auc(rf_fpr, rf_tpr)
xgb_auc = auc(xgb_fpr, xgb_tpr)

plt.figure(figsize=(8, 6))
plt.plot(rf_fpr, rf_tpr, label=f"RandomForest (AUC = {rf_auc:.2f})")
plt.plot(xgb_fpr, xgb_tpr, label=f"XGBoost (AUC = {xgb_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Comparación ROC - RandomForest vs XGBoost")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


# In[10]:


# Celda 9: Importancia de variables

import seaborn as sns
import matplotlib.pyplot as plt

# RandomForest feature importance
rf_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=rf_importances.values, y=rf_importances.index)
plt.title("Importancia de variables - RandomForest")
plt.show()

# XGBoost feature importance
xgb_importances = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=xgb_importances.values, y=xgb_importances.index)
plt.title("Importancia de variables - XGBoost")
plt.show()



# In[11]:


# Celda 10: SHAP values (barras + dispersión + interactivo)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import shap

# Crear el explainer con el modelo entrenado
explainer = shap.TreeExplainer(xgb_model)   # ✅ usar el modelo, no el módulo
shap_values = explainer(X_test)

print("📊 Importancia promedio de variables (SHAP - barras):")
shap.summary_plot(shap_values, X_test, plot_type="bar")

print("📊 Importancia detallada de variables (SHAP - dispersión):")
shap.summary_plot(shap_values, X_test)


# In[12]:


# Celda 11: Dashboard interactivo con Plotly
import plotly.express as px

fig = px.line(agg, x="hour_bucket", y="fraud_rate", color="channel",
              title="Tasa de fraude por hora y canal (interactivo)")
fig.show()


# In[13]:


# Celda 12: Mostrar resultados en el notebook en lugar de exportar a Excel

print("📊 Datos originales (primeras filas):")
display(df.head())

print("📊 Agregados de fraude por canal y hora:")
display(agg.head(10))

print("📊 Resultados de modelos de colas:")
display(df_queues)

print("📊 Top reglas de asociación:")
display(top_rules[["antecedents","consequents","support","confidence","lift"]])

print("📊 Reporte RandomForest (Weighted Avg y Macro Avg):")
display(rf_report.loc[["weighted avg","macro avg"]][["precision","recall","f1-score"]])

print("📊 Reporte XGBoost (Weighted Avg y Macro Avg):")
display(xgb_report.loc[["weighted avg","macro avg"]][["precision","recall","f1-score"]])



# In[14]:


# Celda 13: Narrativa ejecutiva automática
summary_text = f"""
Resumen Ejecutivo:
- Se analizaron {len(df)} transacciones ficticias.
- La tasa de fraude promedio fue {df['is_fraud'].mean():.2%}.
- El modelo RandomForest obtuvo F1={rf_report.loc['weighted avg','f1-score']:.2f}.
- El modelo XGBoost obtuvo F1={xgb_report.loc['weighted avg','f1-score']:.2f}.
- El sistema de colas M/M/1 resultó {'estable' if mm1['Estable'] else 'inestable'} con rho={mm1['Rho']:.2f}.
- El sistema M/M/c resultó {'estable' if mmc['Estable'] else 'inestable'} con rho={mmc['Rho']:.2f}.
"""

print(summary_text)


# In[15]:


# Celda 14: Simulación de reducción de alertas irrelevantes
lambda_h_reduced = df["is_fraud"].sum() * 0.7  # reducción del 30% en alertas
mmc_reduced = mmc_metrics(lambda_h_reduced/60, mu_h/60, c)

print("📊 Colas con reducción de alertas irrelevantes:")
display(pd.DataFrame([mmc, mmc_reduced], index=["Original","Reducido"]))


# In[16]:


# Celda 15: Conclusiones finales
print("✅ Conclusiones:")
print("- Los modelos entrenados permiten detectar fraude con métricas aceptables en un dataset ficticio.")
print("- La interpretabilidad (feature importance, SHAP) muestra qué variables son más relevantes.")
print("- El análisis de colas evidencia la necesidad de ajustar agentes o automatización.")
print("- Las reglas de asociación aportan insights adicionales sobre patrones de fraude.")
print("- El reporte exportado resume todo en tablas y métricas para stakeholders.")


# In[17]:


# Proceso de Poisson

import numpy as np
import matplotlib.pyplot as plt

# Supongamos que detectamos 10 fraudes en 1 hora
lambda_rate = 10  # fraudes por hora

# Tiempo esperado hasta el próximo fraude
expected_time = 1 / lambda_rate
print(f"Tiempo esperado hasta el próximo fraude: {expected_time*60:.2f} minutos")

# Simulación de tiempos de detección (distribución exponencial)
n_samples = 1000
times = np.random.exponential(scale=1/lambda_rate, size=n_samples)

# Graficar histograma
plt.hist(times*60, bins=30, density=True, alpha=0.6, color='blue')
plt.title("Distribución del tiempo hasta detección de fraude (minutos)")
plt.xlabel("Tiempo (minutos)")
plt.ylabel("Densidad")
plt.show()


# In[ ]:




