"""Interfaz de Ventana de Cumbre (Streamlit). Consume la API de app/api/main.py.

Correr:  streamlit run app/ui/streamlit_app.py   (con la API levantada)
"""
import os

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

API = os.getenv("API_URL", "http://localhost:8000")
COLORES = {"verde": "#2e9e5b", "amarillo": "#e0a800", "rojo": "#d64545"}

st.set_page_config(page_title="Ventana de Cumbre", page_icon="⛰️", layout="wide")
st.title("Ventana de Cumbre")
st.caption("¿A qué hora salir para estar arriba con viento y frío seguros? Pronóstico a 3 km con IA.")

try:
    rutas = requests.get(f"{API}/rutas", timeout=10).json()
except requests.RequestException:
    st.error(f"No puedo conectar con la API en {API}. ¿Está corriendo?")
    st.stop()

col1, col2 = st.columns([1, 2])
with col1:
    ruta = st.selectbox("Ruta", rutas, format_func=lambda r: r["nombre"])
    perfil = st.radio("Perfil del grupo", ["principiante", "intermedio", "experto"], index=1)
    horas = st.slider("Horas de pronóstico", 24, 168, 72, step=12)
    calcular = st.button("Calcular plan", type="primary")

with col2:
    m = folium.Map(location=[ruta["puntos"][0]["lat"], ruta["puntos"][0]["lon"]], zoom_start=12)
    folium.PolyLine([(p["lat"], p["lon"]) for p in ruta["puntos"]], weight=3).add_to(m)
    for p in ruta["puntos"]:
        folium.Marker((p["lat"], p["lon"]), tooltip=f'{p["nombre"]} · {p["altura_m"]} m').add_to(m)
    st_folium(m, height=380, use_container_width=True)

if calcular:
    with st.spinner("Bajando el GFS y corriendo el modelo..."):
        r = requests.get(f"{API}/plan", params={"ruta_id": ruta["id"], "perfil": perfil, "horas": horas}, timeout=600)
    if r.status_code != 200:
        st.error(r.json().get("detail", r.text))
        st.stop()
    plan = r.json()
    color = COLORES[plan["semaforo"]]
    st.markdown(f"### Mejor salida: **{plan['mejor_salida_lima']}** (hora de Lima)")
    st.markdown(f"<span style='color:{color};font-weight:700'>Condición: {plan['semaforo'].upper()}</span>",
                unsafe_allow_html=True)
    st.table(plan["detalle"])
