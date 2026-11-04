"""
App Integrada: Hub Completo de Machine Learning (3 en 1)
Materia: Aprendizaje Automático - TUCD (UNdeC)
Profesor: Ing. Matias Agustín Pérez

Objetivo didáctico:
Integrar en una sola aplicación las 3 técnicas vistas a lo largo del cuatrimestre:
1. Clasificación (Árbol de Decisión).
2. Regresión (Regresión Lineal).
3. Clustering (K-Means).
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

# Configuración básica
st.set_page_config(page_title="Hub Completo ML | TUCD UNdeC", page_icon="🤖")

# Rutas de archivos
DIR_ACTUAL = Path(__file__).parent
RUTA_DATOS = DIR_ACTUAL / "datos_clientes.csv"
RUTA_CLF = DIR_ACTUAL / "modelo_clasificacion_arbol.joblib"
RUTA_REG = DIR_ACTUAL / "modelo_regresion_lineal.joblib"
RUTA_CLU = DIR_ACTUAL / "modelo_clustering_kmeans.joblib"

# Carga de recursos
@st.cache_resource
def cargar_modelos():
    return (
        joblib.load(RUTA_CLF),
        joblib.load(RUTA_REG),
        joblib.load(RUTA_CLU)
    )

@st.cache_data
def cargar_datos():
    return pd.read_csv(RUTA_DATOS)

modelo_clf, modelo_reg, modelo_clu = cargar_modelos()
df = cargar_datos()

# Encabezado principal
st.title("🤖 Hub Integrado de Machine Learning")
st.write("Tres modelos de Machine Learning en producción sobre el mismo caso de estudio.")

st.markdown("---")

# 1. Variables de entrada compartidas
st.subheader("1. Ingrese los datos del cliente")
col1, col2 = st.columns(2)

with col1:
    edad = st.slider("Edad (años)", min_value=18, max_value=70, value=35)
    ingresos_k = st.slider("Ingresos mensuales (miles $)", min_value=150.0, max_value=1500.0, value=650.0, step=10.0)

with col2:
    score_crediticio = st.slider("Score crediticio (300 a 850)", min_value=300, max_value=850, value=680, step=5)

datos_entrada = pd.DataFrame([{
    'edad': edad,
    'ingresos_k': ingresos_k,
    'score_crediticio': score_crediticio
}])

st.markdown("---")
st.subheader("2. Seleccione el Tipo de Modelo")

# Pestañas para cada técnica
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Clasificación (Árbol)",
    "📈 Regresión (Lineal)",
    "🧩 Clustering (K-Means)",
    "📊 Ver Dataset"
])

# ----------------- CLASIFICACIÓN -----------------
with tab1:
    st.write("### Árbol de Decisión (`DecisionTreeClassifier`)")
    if st.button("Evaluar Aprobación", type="primary", key="btn_clf_int"):
        pred_clf = modelo_clf.predict(datos_entrada)[0]
        prob_clf = modelo_clf.predict_proba(datos_entrada)[0][1]
        
        if pred_clf == 1:
            st.success("✅ **CRÉDITO APROBADO**")
        else:
            st.error("❌ **CRÉDITO RECHAZADO**")
            
        st.metric("Probabilidad estimada", f"{prob_clf * 100:.1f} %")

# ----------------- REGRESIÓN -----------------
with tab2:
    st.write("### Regresión Lineal Múltiple (`LinearRegression`)")
    if st.button("Calcular Monto Sugerido", type="primary", key="btn_reg_int"):
        monto = max(0.0, modelo_reg.predict(datos_entrada)[0])
        st.metric("Monto Máximo de Préstamo", f"$ {monto:,.1f} mil")

# ----------------- CLUSTERING -----------------
with tab3:
    st.write("### K-Means Clustering (`KMeans k=3`)")
    if st.button("Segmentar Cliente", type="primary", key="btn_clu_int"):
        cluster = modelo_clu.predict(datos_entrada)[0]
        perfiles = {
            0: "Perfil A: Clientes de Ingresos Altos",
            1: "Perfil B: Clientes de Ingresos Iniciales / Jóvenes",
            2: "Perfil C: Clientes de Ingresos Medios"
        }
        st.metric("Cluster Asignado", f"Cluster {cluster}")
        st.info(f"📌 **{perfiles.get(cluster, '')}**")

# ----------------- DATASET -----------------
with tab4:
    st.write("### Datos Históricos de Entrenamiento")
    st.dataframe(df.head(15))

