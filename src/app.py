import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
from pathlib import Path

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Monitor Democrático LatAm", layout="wide")

# Rutas
BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_DATOS = BASE_DIR / "data" / "processed" / "datos_limpios.csv"
MODEL_PATH = BASE_DIR / "models" / "modelo_democracia.pkl"

# --- CARGA DE RECURSOS ---
@st.cache_data
def load_data():
    return pd.read_csv(RUTA_DATOS)

@st.cache_resource
def load_model():
    return pickle.load(open(MODEL_PATH, "rb"))

df = load_data()
model = load_model()

# --- INTERFAZ ---
st.title("🌍 Monitor Democrático de América Latina")
st.markdown("Análisis interactivo de la democracia liberal (2005-actualidad)")

# Sidebar - Filtros y Predicción
st.sidebar.header("Opciones")
paises = sorted(df['country_name'].unique())
pais_seleccionado = st.sidebar.multiselect("Selecciona países:", paises, default=["Chile"])

st.sidebar.subheader("Predicción de Democracia")
year_input = st.sidebar.number_input("Año (2026-2036):", 2026, 2036)
poly_input = st.sidebar.slider("Nivel de Poliarquía (0 a 1):", 0.0, 1.0, 0.4)

# --- LÓGICA DE VISUALIZACIÓN Y PREDICCIÓN ---
if pais_seleccionado:
    # Gráfico
    df_filtrado = df[df['country_name'].isin(pais_seleccionado)]
    fig = px.line(
        df_filtrado, x='year', y='v2x_libdem', color='country_name',
        color_discrete_sequence=['#FF00FF', '#00FF00', '#00FFFF', '#FFFF00', '#FF4500', '#7FFF00'],
        title="Tendencia de Democracia Liberal"
    )
    fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)', 'paper_bgcolor': 'rgba(0,0,0,0)', 'font_color': 'white'})
    st.plotly_chart(fig, use_container_width=True)

    # Predicción única
    if st.sidebar.button("Predecir índice"):
        # Asegúrate de que coincida con lo que entrenaste (ej: año y poliarquía)
        prediccion = model.predict([[year_input, poly_input]])
        st.sidebar.success(f"Índice proyectado: **{prediccion[0]:.2f}**")
else:
    st.info("Por favor, selecciona al menos un país en el menú lateral para ver el gráfico.")