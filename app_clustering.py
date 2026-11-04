"""
App 3: Clustering No Supervisado con K-Means
Materia: Aprendizaje Automático - TUCD (UNdeC)
Profesor: Ing. Matias Agustín Pérez

Objetivo didáctico:
Una aplicación enfocada en Aprendizaje No Supervisado con K-Means (k=3)
para segmentar automáticamente perfiles de clientes sin etiquetas previas.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

# Configuración básica
st.set_page_config(page_title="Clustering - Segmentación", page_icon="🧩")

# Rutas de archivos
DIR_ACTUAL = Path(__file__).parent
RUTA_MODELO = DIR_ACTUAL / "modelo_clustering_kmeans.joblib"
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
st.title("🧩 Segmentación de Perfiles de Clientes")
st.write("Modelo de **Clustering** (No Supervisado): K-Means (`KMeans k=3`).")

st.subheader("1. Ingrese los datos del cliente")

# Controles de entrada
col1, col2 = st.columns(2)

with col1:
    edad = st.slider("Edad (años)", min_value=18, max_value=70, value=35)
    ingresos_k = st.slider("Ingresos mensuales (miles $)", min_value=150.0, max_value=1500.0, value=650.0, step=10.0)

with col2:
    score_crediticio = st.slider("Score crediticio (300 a 850)", min_value=300, max_value=850, value=680, step=5)

st.subheader("2. Asignación de Cluster")

# Botón de inferencia
if st.button("Segmentar Cliente", type="primary"):
    datos_nuevos = pd.DataFrame([{
        'edad': edad,
        'ingresos_k': ingresos_k,
        'score_crediticio': score_crediticio
    }])
    
    cluster_asignado = modelo.predict(datos_nuevos)[0]
    
    nombres_perfiles = {
        0: "Perfil A: Clientes de Ingresos Altos y Gran Capacidad",
        1: "Perfil B: Clientes de Ingresos Iniciales / Jóvenes",
        2: "Perfil C: Clientes de Ingresos Medios Consolidados"
    }
    
    descripcion = nombres_perfiles.get(cluster_asignado, f"Cluster {cluster_asignado}")
    
    st.metric(
        label="Cluster Asignado por K-Means",
        value=f"Cluster {cluster_asignado}"
    )
    st.success(f"📌 **{descripcion}**")

# Muestra de datos histórica
with st.expander("📊 Ver muestra de datos con clusters asignados"):
    st.dataframe(df[['edad', 'ingresos_k', 'score_crediticio', 'cluster']].head(10))

