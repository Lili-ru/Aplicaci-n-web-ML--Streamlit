import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
from pathlib import Path

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="DemocraScope LatAm", layout="wide")

# Rutas optimizadas con constantes
BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_DATOS = BASE_DIR / "data" / "processed" / "datos_limpios.csv"
MODEL_PATH = BASE_DIR / "models" / "modelo_democracia.pkl"

# --- CARGA EFICIENTE ---
@st.cache_data
def load_data():
    return pd.read_csv(RUTA_DATOS)

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

df = load_data()
model = load_model()

# --- INTERFAZ ---
st.title("⚖️ DemocraScope LatAm")
st.subheader("Análisis predictivo de la salud democrática en América Latina")
st.markdown("""
Esta herramienta te permite explorar la evolución histórica de la democracia liberal 
desde el año 2000 y proyectar escenarios futuros utilizando modelos de Machine Learning.

**¿Qué puedes hacer aquí?**
* **Monitorear:** Visualiza tendencias históricas de diversos países de la región.
* **Simular:** Evalúa cómo variables clave influyen en el futuro de nuestras instituciones.
""")

# --- SIDEBAR Y LÓGICA ---
st.sidebar.header("Opciones")
paises_disponibles = sorted(df['country_name'].unique())
pais_seleccionado = st.sidebar.multiselect("Selecciona países:", paises_disponibles, default=["Chile"])

if pais_seleccionado:
    # 1. Gráfico
    df_filtrado = df[df['country_name'].isin(pais_seleccionado)]
    fig = px.line(df_filtrado, x='year', y='v2x_libdem', color='country_name', 
                  title="Tendencia de Democracia Liberal")
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    # 2. Predicción (Lógica para 4 variables)
    st.sidebar.subheader("Simulación de Futuro")
    year_input = st.sidebar.number_input("Año proyectado:", 2026, 2036)
    poly_input = st.sidebar.slider("Nivel de Poliarquía (0 a 1):", 0.0, 1.0, 0.4)
    
    if st.sidebar.button("Predecir índice"):
        # Calculamos los inputs basados en el último dato real del primer país seleccionado
        ultimo_dato = df_filtrado.iloc[-1]
        media_movil = ultimo_dato.get('v2x_media_movil_2a', 0.5)
        cambio_reciente = ultimo_dato.get('v2x_cambio_reciente', 0.0)
        
        # Predicción con las 4 variables
        input_data = [[year_input, poly_input, media_movil, cambio_reciente]]
        prediccion = model.predict(input_data)
        
        st.sidebar.success(f"Índice proyectado para {year_input}: **{prediccion[0]:.2f}**")
else:
    st.info("Selecciona un país para comenzar.")