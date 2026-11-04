"""
App 2: Regresión Lineal Múltiple
Materia: Aprendizaje Automático - TUCD (UNdeC)
Profesor: Ing. Matias Agustín Pérez

Objetivo didáctico:
Una aplicación enfocada exclusivamente en el modelo de
Regresión Lineal para estimar un valor numérico continuo (Monto de Préstamo).
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

# Configuración básica
st.set_page_config(page_title="Regresión - Monto de Préstamo", page_icon="📈")

# Rutas de archivos
DIR_ACTUAL = Path(__file__).parent
RUTA_MODELO = DIR_ACTUAL / "modelo_regresion_lineal.joblib"
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
st.title("📈 Estimación del Monto de Préstamo")
st.write("Modelo de **Regresión**: Regresión Lineal Múltiple (`LinearRegression`).")

st.subheader("1. Ingrese los datos del solicitante")

# Controles de entrada
col1, col2 = st.columns(2)

with col1:
    edad = st.slider("Edad (años)", min_value=18, max_value=70, value=35)
    ingresos_k = st.slider("Ingresos mensuales (miles $)", min_value=150.0, max_value=1500.0, value=650.0, step=10.0)

with col2:
    score_crediticio = st.slider("Score crediticio (300 a 850)", min_value=300, max_value=850, value=680, step=5)

st.subheader("2. Predicción del Modelo")

# Botón de cálculo
if st.button("Calcular Monto de Préstamo", type="primary"):
    datos_nuevos = pd.DataFrame([{
        'edad': edad,
        'ingresos_k': ingresos_k,
        'score_crediticio': score_crediticio
    }])
    
    monto_estimado = modelo.predict(datos_nuevos)[0]
    monto_estimado = max(0.0, monto_estimado)
    
    st.metric(
        label="Monto Máximo de Préstamo Estimado",
        value=f"$ {monto_estimado:,.1f} mil"
    )
    st.info(
        f"El modelo calcula el monto combinando: "
        f"Ingresos (${ingresos_k}k) × 2.2 + Score ({score_crediticio}) × 0.8 + Edad ({edad}) × 5.3"
    )

# Muestra de datos histórica
with st.expander("📊 Ver muestra de datos de entrenamiento"):
    st.dataframe(df[['edad', 'ingresos_k', 'score_crediticio', 'monto_prestamo_k']].head(10))

