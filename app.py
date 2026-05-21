# Librerías
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Predicción de Precios de Viviendas", layout="centered")

# Título principal de la aplicación
st.title("📊 Simulador de Precios de Viviendas")
st.write("Introduzca las características de la propiedad para estimar su precio de venta en el mercado.")

# --- BARRA LATERAL: INFORMACIÓN DEL ESTUDIANTE (Requisito de la Evaluación) ---
st.sidebar.header("📌 Información del Estudiante")
st.sidebar.write("**Nombre:** Hector Salvatierra Valle")
st.sidebar.write("**Código ISIL:** 46887492")
st.sidebar.markdown("[🔗 Enlace a mi Cuaderno de Google Colab](https://colab.research.google.com/drive/1ZZEZdQGhq_9rmMphs4nqFd_GoaP7TZWL?usp=sharing) (Modo Lector)")

st.sidebar.divider()

# Selección del modelo a usar
st.sidebar.header("🤖 Configuración del Modelo")
tipo_modelo = st.sidebar.radio(
    "Seleccione el modelo de Machine Learning:",
    ("Regresión Lineal (Guardado como Logística)", "Random Forest")
)

# Cargar los modelos correspondientes de la carpeta 'modelos'
try:
    if tipo_modelo == "Regresión Lineal (Guardado como Logística)":
        modelo = joblib.load("modelos/modelo_regresion_logistica.pkl")
    else:
        modelo = joblib.load("modelos/modelo_bosque.pkl")
except FileNotFoundError:
    st.error("⚠️ No se encontraron los archivos del modelo en la carpeta 'modelos/'. Asegúrate de haber subido correctamente los archivos .pkl a GitHub.")
    st.stop()

# --- CUERPO PRINCIPAL: FORMULARIO DE ENTRADA ---
st.subheader("🏠 Características de la Propiedad")

# Distribución en columnas para un diseño más limpio
col1, col2 = st.columns(2)

with col1:
    m2 = st.number_input("Tamaño de la vivienda en m²:", min_value=10.0, max_value=500.0, value=120.0, step=1.0)
    habitaciones = st.number_input("Número de Habitaciones:", min_value=1, max_value=10, value=3, step=1)

with col2:
    banos = st.number_input("Número de Baños:", min_value=1, max_value=10, value=2, step=1)
    zona = st.selectbox("Zona de la Propiedad:", ["Cono Norte", "Cono Sur", "Cono Este", "Cono Oeste"])

# Nota técnica: El modelo original fue entrenado con la columna 'Ofertas'. 
# Añadimos este input filtrado de 1 a 4 según el procesamiento previo para evitar errores de dimensiones.
ofertas = st.slider("Número de Ofertas recibidas previamente:", min_value=1, max_value=4, value=2, step=1)

# --- PROCESAMIENTO DE DATOS (One-Hot Encoding Manual) ---
# Creamos el DataFrame base con las entradas del usuario
input_df = pd.DataFrame([{
    'm2': m2,
    'Habitaciones': habitaciones,
    'Banos': banos,
    'Ofertas': ofertas,
    'Zona_Cono Norte': 1 if zona == "Cono Norte" else 0,
    'Zona_Cono Oeste': 1 if zona == "Cono Oeste" else 0,
    'Zona_Cono Sur': 1 if zona == "Cono Sur" else 0,
    'Propiedad': 1  # Valor por defecto en caso de que la columna 'Propiedad' (Home) no se haya eliminado
}])

# Robustez: Alinear las columnas dinámicamente con lo que espera el modelo entrenado
if hasattr(modelo, 'feature_names_in_'):
    columnas_entrenamiento = modelo.feature_names_in_
    # Si faltan columnas esperadas por el modelo, las agregamos con valor 0
    for col in columnas_entrenamiento:
        if col not in input_df.columns:
            input_df[col] = 0
    # Reordenamos las columnas exactamente igual a como se entrenó el modelo
    input_df = input_df[columnas_entrenamiento]

# --- BOTÓN DE PREDICCIÓN ---
st.write("---")
if st.button("🚀 Calcular Precio Estimado"):
    try:
        # Realizar la predicción
        prediccion = modelo.predict(input_df)
        
        # Mostrar el resultado estilizado
        st.success(f"### 💵 El precio de venta estimado es: ${prediccion:,.2f}")
        st.balloons()
    except Exception as e:
        st.error(f"❌ Ocurrió un error al realizar la predicción: {e}")
        st.info("Revisa que las variables numéricas y dummificadas coincidan con el entrenamiento de tu Colab.")
"""

# Define the content for requirements.txt
requirements_code = streamlit==1.32.0
pandas==2.1.4
numpy==1.26.2
scikit-learn==1.3.2
joblib==1.3.2
"""

# Define the content for anotaciones.txt
# PROMPT UTILIZADO PARA LA CREACIÓN DEL APLICATIVO:
"Necesito que generes una página web en streamlit desde github. Esta pagina contiene dos modelos entrenados para poder obtener el precio de venta de una propiedad. Los modelos fueron generados con joblib, uno se llama modelo_regresion_logistica.pkl y el otro modelo_bosque.pkl. ambos están almacenados en un github en una carpeta llamada modelos. Necesito que el usuario ingrese los siguientes valores: m2, Habitaciones, Banos, Zona. Finalmente agrega un botón para obtener la predicción. Incluye también los requisitos de la rúbrica de ISIL: nombre del estudiante, código ISIL y enlace al cuaderno de Colab en modo lector."


# Save files
with open("app.py", "w", encoding="utf-8") as f:
    f.write(app_code)

with open("requirements.txt", "w", encoding="utf-8") as f:
    f.write(requirements_code)

with open("anotaciones.txt", "w", encoding="utf-8") as f:
    f.write(anotaciones_code)

print("Files generated successfully.")
