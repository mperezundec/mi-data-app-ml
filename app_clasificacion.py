"""
App 1: Clasificación con Árbol de Decisión
Materia: Aprendizaje Automático - TUCD (UNdeC)
Profesor: Ing. Matias Agustín Pérez

Objetivo didáctico:
Una aplicación súper simple y concisa enfocada exclusivamente
en el modelo de Árbol de Decisión (Aprobación de Créditos).
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

# Configuración básica
st.set_page_config(page_title="Clasificación - Créditos", page_icon="💳")

# Rutas de archivos
DIR_ACTUAL = Path(__file__).parent
RUTA_MODELO = DIR_ACTUAL / "modelo_clasificacion_arbol.joblib"
RUTA_DATOS = DIR_ACTUAL / "datos_clientes.csv"

# Carga de recursos
@st.cache_resource
def cargar_modelo():
    return joblib.load(RUTA_MODELO)

@st.cache_data
def cargar_datos():
    return pd.read_csv(RUTA_DATOS)

modelo = cargar_modelo()
df = cargar_datos()

# Encabezado
st.title("💳 Simulador de Aprobación de Créditos")
st.write("Modelo de **Clasificación**: Árbol de Decisión (`DecisionTreeClassifier`).")

st.subheader("1. Ingrese los datos del solicitante")

# Controles nativos de entrada
col1, col2 = st.columns(2)

with col1:
    edad = st.slider("Edad (años)", min_value=18, max_value=70, value=35)
    ingresos_k = st.slider("Ingresos mensuales (miles $)", min_value=150.0, max_value=1500.0, value=650.0, step=10.0)

with col2:
    score_crediticio = st.slider("Score crediticio (300 a 850)", min_value=300, max_value=850, value=680, step=5)

st.subheader("2. Predicción del Modelo")

# Botón de predicción
if st.button("Evaluar Aprobación", type="primary"):
    datos_nuevos = pd.DataFrame([{
        'edad': edad,
        'ingresos_k': ingresos_k,
        'score_crediticio': score_crediticio
    }])
    
    prediccion = modelo.predict(datos_nuevos)[0]
    probabilidad = modelo.predict_proba(datos_nuevos)[0][1]
    
    if prediccion == 1:
        st.success("✅ **Resultado: ¡CRÉDITO APROBADO!**")
    else:
        st.error("❌ **Resultado: CRÉDITO RECHAZADO**")
        
    st.metric(
        label="Probabilidad estimada de aprobación",
        value=f"{probabilidad * 100:.1f} %"
    )

# Muestra de datos histórica
with st.expander("📊 Ver muestra de datos de entrenamiento"):
    st.dataframe(df[['edad', 'ingresos_k', 'score_crediticio', 'aprobado']].head(10))

