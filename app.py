# Librerías
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Precios de Viviendas",
    layout="centered"
)

# Título principal
st.title("📊 Simulador de Precios de Viviendas")

st.write(
    "Introduzca las características de la propiedad "
    "para estimar su precio de venta en el mercado."
)

# ==========================================
# BARRA LATERAL
# ==========================================

st.sidebar.header("📌 Información del Estudiante")

st.sidebar.write("**Nombre:** Hector Salvatierra Valle")
st.sidebar.write("**Código ISIL:** 46887492")

st.sidebar.markdown(
    "[🔗 Enlace a mi Cuaderno de Google Colab]"
    "(https://colab.research.google.com/drive/1ZZEZdQGhq_9rmMphs4nqFd_GoaP7TZWL?usp=sharing)"
)

st.sidebar.divider()

# ==========================================
# SELECCIÓN DEL MODELO
# ==========================================

st.sidebar.header("🤖 Configuración del Modelo")

tipo_modelo = st.sidebar.radio(
    "Seleccione el modelo de Machine Learning:",
    (
        "Regresión Lineal",
        "Random Forest"
    )
)

# ==========================================
# CARGA DEL MODELO
# ==========================================

try:

    if tipo_modelo == "Regresión Lineal":
        modelo = joblib.load(
            "modelos/modelo_regresion_logistica.pkl"
        )

    else:
        modelo = joblib.load(
            "modelos/modelo_bosque.pkl"
        )

except FileNotFoundError:

    st.error(
        "⚠️ No se encontraron los archivos .pkl "
        "en la carpeta 'modelos/'."
    )

    st.stop()

# ==========================================
# FORMULARIO
# ==========================================

st.subheader("🏠 Características de la Propiedad")

col1, col2 = st.columns(2)

with col1:

    m2 = st.number_input(
        "Tamaño de la vivienda en m²:",
        min_value=10.0,
        max_value=500.0,
        value=120.0,
        step=1.0
    )

    habitaciones = st.number_input(
        "Número de Habitaciones:",
        min_value=1,
        max_value=10,
        value=3,
        step=1
    )

with col2:

    banos = st.number_input(
        "Número de Baños:",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )

    zona = st.selectbox(
        "Zona de la Propiedad:",
        [
            "Cono Norte",
            "Cono Sur",
            "Cono Este",
            "Cono Oeste"
        ]
    )

# Variable adicional usada en el entrenamiento
ofertas = st.slider(
    "Número de Ofertas recibidas previamente:",
    min_value=1,
    max_value=4,
    value=2,
    step=1
)

# ==========================================
# DATAFRAME DE ENTRADA
# ==========================================

input_df = pd.DataFrame([{

    'm2': m2,
    'Habitaciones': habitaciones,
    'Banos': banos,
    'Ofertas': ofertas,

    # Variables dummy
    'Zona_Cono Norte': 1 if zona == "Cono Norte" else 0,
    'Zona_Cono Oeste': 1 if zona == "Cono Oeste" else 0,
    'Zona_Cono Sur': 1 if zona == "Cono Sur" else 0

}])

# ==========================================
# ALINEAR COLUMNAS CON EL MODELO
# ==========================================

if hasattr(modelo, 'feature_names_in_'):

    columnas_entrenamiento = modelo.feature_names_in_

    # Agregar columnas faltantes
    for col in columnas_entrenamiento:

        if col not in input_df.columns:
            input_df[col] = 0

    # Reordenar columnas
    input_df = input_df[columnas_entrenamiento]

# ==========================================
# BOTÓN DE PREDICCIÓN
# ==========================================

st.write("---")

if st.button("🚀 Calcular Precio Estimado"):

    try:

        # Predicción
        prediccion = modelo.predict(input_df)

        # Convertir a número
        prediccion = float(prediccion[0])

        # Mostrar resultado
        st.success(
            f"### 💵 El precio de venta estimado es: "
            f"${prediccion:,.2f}"
        )

        st.balloons()

    except Exception as e:

        st.error(
            f"❌ Ocurrió un error al realizar la predicción:\n\n{e}"
        )

        st.info(
            "Revisa que las variables numéricas y "
            "dummificadas coincidan con el entrenamiento."
        )
