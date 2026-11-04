"""
Entrenamiento y Serialización de Modelos - Clase 12: Deploy
Materia: Aprendizaje Automático - TUCD (UNdeC)
Profesor: Ing. Matias Agustín Pérez

Recorrido Integral de Machine Learning:
1. Clasificación: Árbol de Decisión (Aprobación de Crédito).
2. Regresión: Regresión Lineal (Estimación de Monto de Préstamo).
3. Clustering: K-Means (Segmentación de Perfiles de Clientes).

Variables de Entrada (Features):
- edad: edad del cliente (18 a 65 años).
- ingresos_k: ingreso mensual en miles de pesos (200k a 1500k).
- score_crediticio: puntaje crediticio tradicional (300 a 850).
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import joblib

DIRECTORIO_ACTUAL = Path(__file__).parent

def generar_datos(n_muestras: int = 300, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    
    # 1. Tres variables predictoras (Features)
    edad = np.random.randint(18, 66, size=n_muestras)
    ingresos_k = np.round(np.random.uniform(200, 1500, size=n_muestras), 1)
    score_crediticio = np.random.randint(300, 851, size=n_muestras)
    
    # 2. Target de Clasificación: aprobado (1 = Aprobado, 0 = Rechazado)
    puntaje = (
        (score_crediticio >= 620).astype(int) * 0.5 +
        (ingresos_k >= 500).astype(int) * 0.3 +
        (edad >= 25).astype(int) * 0.2 +
        np.random.normal(0, 0.08, size=n_muestras)
    )
    aprobado = (puntaje >= 0.55).astype(int)
    
    # 3. Target de Regresión: monto_prestamo_k (miles de pesos sugeridos)
    # Lógica económica: se presta más a mayores ingresos y mejor score
    monto_base = (ingresos_k * 2.2) + (score_crediticio * 0.8) + (edad * 5)
    ruido_monto = np.random.normal(0, 50, size=n_muestras)
    monto_prestamo_k = np.round(np.clip(monto_base + ruido_monto, 300, 4500), 1)
    
    df = pd.DataFrame({
        'edad': edad,
        'ingresos_k': ingresos_k,
        'score_crediticio': score_crediticio,
        'aprobado': aprobado,
        'monto_prestamo_k': monto_prestamo_k
    })
    
    return df

def main():
    print("=========================================================")
    print("  CLASE 12: ENTRENAMIENTO DE LOS 3 MODELOS PARA DEPLOY   ")
    print("=========================================================")
    
    # Paso 1: Generar datos
    df = generar_datos(n_muestras=300)
    features = ['edad', 'ingresos_k', 'score_crediticio']
    X = df[features]
    
    print(f"\n1. Conjunto de datos generado ({len(df)} filas, {len(df.columns)} columnas):")
    print(df.head())
    
    # -------------------------------------------------------------
    # MODELO 1: CLASIFICACIÓN (Árbol de Decisión)
    # -------------------------------------------------------------
    print("\n---------------------------------------------------------")
    print("2. Entrenando MODELO DE CLASIFICACIÓN (Árbol de Decisión)...")
    y_clf = df['aprobado']
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    
    modelo_arbol = DecisionTreeClassifier(max_depth=3, random_state=42)
    modelo_arbol.fit(X_train_c, y_train_c)
    acc = accuracy_score(y_test_c, modelo_arbol.predict(X_test_c))
    print(f"   Exactitud (Accuracy): {acc * 100:.1f}%")
    
    archivo_clf = DIRECTORIO_ACTUAL / "modelo_clasificacion_arbol.joblib"
    joblib.dump(modelo_arbol, archivo_clf)
    print(f"   -> Guardado: {archivo_clf.name}")
    
    # -------------------------------------------------------------
    # MODELO 2: REGRESIÓN (Regresión Lineal)
    # -------------------------------------------------------------
    print("\n---------------------------------------------------------")
    print("3. Entrenando MODELO DE REGRESIÓN (Regresión Lineal)...")
    y_reg = df['monto_prestamo_k']
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    
    modelo_regresion = LinearRegression()
    modelo_regresion.fit(X_train_r, y_train_r)
    r2 = r2_score(y_test_r, modelo_regresion.predict(X_test_r))
    print(f"   Coeficiente R2: {r2:.3f}")
    print(f"   Intercepto: {modelo_regresion.intercept_:.2f}")
    for col, coef in zip(features, modelo_regresion.coef_):
        print(f"   Coeficiente ({col}): {coef:.2f}")
        
    archivo_reg = DIRECTORIO_ACTUAL / "modelo_regresion_lineal.joblib"
    joblib.dump(modelo_regresion, archivo_reg)
    print(f"   -> Guardado: {archivo_reg.name}")
    
    # -------------------------------------------------------------
    # MODELO 3: CLUSTERING (K-Means, Aprendizaje No Supervisado)
    # -------------------------------------------------------------
    print("\n---------------------------------------------------------")
    print("4. Entrenando MODELO DE CLUSTERING (K-Means k=3)...")
    modelo_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    modelo_kmeans.fit(X)
    
    # Asignamos el cluster a cada registro para que se pueda ver en el dataset
    df['cluster'] = modelo_kmeans.labels_
    
    archivo_clu = DIRECTORIO_ACTUAL / "modelo_clustering_kmeans.joblib"
    joblib.dump(modelo_kmeans, archivo_clu)
    print(f"   -> Guardado: {archivo_clu.name}")
    print("   Centros de cada Cluster:")
    centros = pd.DataFrame(modelo_kmeans.cluster_centers_, columns=features)
    centros.index = [f"Cluster {i}" for i in range(3)]
    print(centros.round(1))
    
    # Guardamos el dataset final
    archivo_csv = DIRECTORIO_ACTUAL / "datos_clientes.csv"
    df.to_csv(archivo_csv, index=False)
    print(f"\n5. Dataset guardado en: {archivo_csv.name}")
    print("=========================================================")
    print("¡Listo! Los 3 modelos y el dataset están serializados.")

if __name__ == "__main__":
    main()
