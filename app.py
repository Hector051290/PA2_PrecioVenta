import streamlit as st
import numpy as np
import joblib
import requests
import os
from io import BytesIO

# ─── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictor de Precio de Propiedad",
    page_icon="🏠",
    layout="centered",
)

# ─── Estilos personalizados ────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.6rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        font-size: 1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .card {
        background: #f8f7f4;
        border-left: 5px solid #c8973a;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.2rem;
    }
    .result-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white;
        border-radius: 14px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .result-label {
        font-size: 0.9rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        opacity: 0.7;
        margin-bottom: 0.4rem;
    }
    .result-price {
        font-family: 'Playfair Display', serif;
        font-size: 2.8rem;
        font-weight: 700;
        color: #c8973a;
    }
    .isil-box {
        background: #f0ede8;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        font-size: 0.88rem;
        color: #444;
        margin-top: 2.5rem;
        border: 1px solid #ddd;
    }
    .isil-box a { color: #c8973a; text-decoration: none; }
    .isil-box a:hover { text-decoration: underline; }
    div[data-testid="stButton"] > button {
        background: #1a1a2e;
        color: #c8973a;
        border: 2px solid #c8973a;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 600;
        padding: 0.6rem 2rem;
        width: 100%;
        transition: all 0.2s;
    }
    div[data-testid="stButton"] > button:hover {
        background: #c8973a;
        color: #1a1a2e;
    }
</style>
""", unsafe_allow_html=True)

# ─── Constantes ────────────────────────────────────────────────────────────────
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/ywchiu/riii/refs/heads/master/data/house-prices.csv"

ZONA_MAP = {
    "Norte": 0,
    "Sur":   1,
    "Este":  2,
    "Oeste": 3,
}

# ─── Carga de modelos desde GitHub ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(filename: str):
    url = GITHUB_RAW_BASE + filename
    response = requests.get(url)
    if response.status_code != 200:
        st.error(f"❌ No se pudo descargar el modelo '{filename}'. Verifica la URL del repositorio.")
        st.stop()
    return joblib.load(BytesIO(response.content))


# ─── Encabezado ────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🏠 Predictor de Precio</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Ingresa las características de la propiedad y obtén una estimación de precio al instante.</div>', unsafe_allow_html=True)

# ─── Formulario de entrada ─────────────────────────────────────────────────────
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 📐 Datos de la Propiedad")

    col1, col2 = st.columns(2)
    with col1:
        m2          = st.number_input("Metros cuadrados (m²)", min_value=10, max_value=2000, value=80, step=5)
        habitaciones = st.number_input("Habitaciones", min_value=1, max_value=20, value=3, step=1)
    with col2:
        banos = st.number_input("Baños", min_value=1, max_value=10, value=2, step=1)
        zona  = st.selectbox("Zona", options=list(ZONA_MAP.keys()))

    modelo_seleccionado = st.radio(
        "Modelo de predicción",
        options=["Regresión Logística", "Bosque Aleatorio (Random Forest)"],
        horizontal=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ─── Predicción ────────────────────────────────────────────────────────────────
if st.button("🔍 Obtener Predicción"):
    with st.spinner("Cargando modelo y calculando..."):
        if modelo_seleccionado == "Regresión Logística":
            model = load_model("modelo_regresion_logistica.pkl")
        else:
            model = load_model("modelo_bosque.pkl")

        zona_encoded = ZONA_MAP[zona]
        features = np.array([[m2, habitaciones, banos, zona_encoded]])

        prediction = model.predict(features)[0]

    st.markdown(f"""
    <div class="result-box">
        <div class="result-label">Precio Estimado · {modelo_seleccionado}</div>
        <div class="result-price">S/ {prediction:,.2f}</div>
        <div style="opacity:0.6; font-size:0.85rem; margin-top:0.8rem;">
            {m2} m² &nbsp;·&nbsp; {habitaciones} hab. &nbsp;·&nbsp; {banos} baños &nbsp;·&nbsp; Zona {zona}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Rúbrica ISIL ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="isil-box">
    <strong>📋 Información del Estudiante — ISIL</strong><br><br>
    <b>Nombre:</b> NOMBRE DEL ESTUDIANTE<br>
    <b>Código ISIL:</b> CÓDIGO ISIL<br>
    <b>Cuaderno de Colab:</b>
    <a href="https://colab.research.google.com/drive/1ZZEZdQGhq_9rmMphs4nqFd_GoaP7TZWL?usp=sharing" target="_blank">
        Ver cuaderno en Google Colab 🔗
    </a>
</div>
""", unsafe_allow_html=True)
