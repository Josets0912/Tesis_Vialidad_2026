import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

# Silenciamos advertencias matemáticas
warnings.filterwarnings("ignore")

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Gestión Vial - Tesis José Tapia",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CARGA DE DATOS ---
@st.cache_data
def cargar_datos():
    archivo = "DATA_MAESTRA_TESIS.xlsx"
    try:
        df = pd.read_excel(archivo)
        
        # LIMPIEZA DE DATOS
        cols_limpiar = ['ROL', 'ROL NUEVO', 'NOMBRE DEL CAMINO', 'TIPO DE CARPETA', 'CLASIFICACIÓN', 'ESTACIÓN', 'CALZADA']
        for col in cols_limpiar:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # CORRECCIÓN 115 CANALES
        errores_115 = ['115 Canales', '115 CANALES', '115-Canales', '115 CH', '115-CH']
        df['ROL'] = df['ROL'].replace(errores_115, 'Ruta 115 CH')
        df['ROL NUEVO'] = df['ROL NUEVO'].replace(errores_115, 'Ruta 115 CH')

        return df
    except FileNotFoundError:
        st.error(f"❌ Error Crítico: No encuentro el archivo '{archivo}'. Asegúrate de haber ejecutado el script de fusión.")
        st.stop()

df = cargar_datos()

# --- 3. MENÚ LATERAL ---
st.sidebar.header("🔍 Panel de Control")
roles = sorted(df['ROL'].unique())
rol_sel = st.sidebar.selectbox("Seleccione Rol:", roles)

df_rol = df[df['ROL'] == rol_sel]
df_rol['ETIQUETA'] = df_rol['NOMBRE DEL CAMINO'] + " (" + df_rol['ESTACIÓN'] + ")"
tramo_sel = st.sidebar.selectbox("Seleccione Sector:", df_rol['ETIQUETA'].tolist())

st.sidebar.markdown("---")
btn_calc = st.sidebar.button("Generar Informe Técnico 🚀")

# --- ESTILOS CSS (Tarjetas bonitas) ---
st.markdown("""
<style>
    .info-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        height: 100%;
        text-align: left;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .info-label {
        font-size: 12px;
        color: #6c757d;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 5px;
        letter-spacing: 0.5px;
    }
    .info-value {
        font-size: 15px;
        color: #212529;
        font-weight: 600;
        line-height: 1.4;
        word-wrap: break-word;
    }
</style>
""", unsafe_allow_html=True)

# --- 4. INTERFAZ Y CÁLCULOS ---

if not btn_calc:
    # --- PORTADA ---
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
        <h1 style='text-align: center; color: #0E1117; font-size: 55px;'>
            🚧 Sistema de Gestión de Pavimentos
        </h1>
        <h2 style='text-align: center; color: #666;'>
            y Proyección de Demanda Vial
        </h2>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
        <h3 style='text-align: center; color: #1f77b4;'>
            Desarrollado por José Tapia
        </h3>
        <p style='text-align: center; font-size: 18px;'>
            Memoria para optar al título de Ingeniero Civil
        </p>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 Seleccione un camino en el menú lateral para iniciar el análisis.")

else:
    # --- REPORTE TÉCNICO ---
    st.markdown("### 🚧 Sistema de Gestión de Pavimentos y Proyección de Demanda")
    st.markdown("---")

    # Datos
    fila = df_rol[df_rol['ETIQUETA'] == tramo_sel].iloc[0]
    nombre = fila['NOMBRE DEL CAMINO']
    rol_oficial = fila['ROL NUEVO']
    carpeta = fila['TIPO DE CARPETA']
    clasificacion = fila['CLASIFICACIÓN']
    calzada_info = fila['CALZADA'] if 'CALZADA' in fila else "No Inf"
    
    st.title(f"📍 {nombre}")
    
    # Tarjetas HTML
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='info-card'><div class='info-label'>Rol Oficial</div><div class='info-value'>{rol_oficial}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='info-card'><div class='info-label'>Tipo de Carpeta</div><div class='info-value'>{carpeta}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='info-card'><div class='info-label'>Clasificación</div><div class='info-value'>{clasificacion}</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='info-card'><div class='info-label'>Calzada</div><div class='info-value'>{calzada_info}</div></div>", unsafe_allow_html=True)
    
    # --- CÁLCULO HOLT ---
    try:
        anios = [2015, 2017, 2018, 2020, 2022, 2024]
        vals = fila[[f'TMDA {a}' for a in anios]].values.flatten().astype(float)
        
        serie = pd.Series(index=np.arange(2015, 2025), dtype=float)
        for a, v in zip(anios, vals):
            serie[a] = v
        serie = serie.interpolate(method='linear')
        
        modelo = ExponentialSmoothing(serie, trend='add', seasonal=None, damped_trend=True).fit()
        anios_fut = np.arange(2025, 2046)
        pred = modelo.forecast(len(anios_fut))
        pred.index = anios_fut
        
        tmda_24 = vals[-1]
        tmda_26 = pred.loc[2026]
        tmda_45 = pred.loc[2045]
        delta = ((tmda_26 - tmda_24)/tmda_24)*100

    except Exception as e:
        st.error(f"Error matemático: {e}")
        st.stop()

    # --- KPI CON UNIDADES (AQUÍ ESTÁ EL CAMBIO) ---
    st.markdown("<br>", unsafe_allow_html=True)
    colA, colB, colC = st.columns(3)
    
    # Agregamos "veh/día" para que quede claro técnicamente
    colA.metric("🚗 Censo 2024", f"{int(tmda_24)} veh/día")
    colB.metric("📈 Proyección 2026", f"{int(tmda_26)} veh/día", f"{delta:.1f}%")
    colC.metric("🔭 Proyección 2045", f"{int(tmda_45)} veh/día")

    # --- GRÁFICO ---
    st.subheader("Evolución de la Demanda y Umbrales")
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(serie.index, serie.values, 'o-', color='black', label='Histórico')
    ax.plot(pred.index, pred.values, '--', color='#2ca02c', linewidth=2, label='Proyección Holt')
    ax.axhline(5000, color='gray', linestyle=':', alpha=0.5, label='Umbral 5.000')
    
    # Punto Rojo
    anio_saturacion = None
    val_saturacion = None
    
    # 1. Historia
    for y in serie.index:
        if serie[y] >= 5000:
            anio_saturacion = y
            val_saturacion = serie[y]
            break 
    # 2. Futuro
    if anio_saturacion is None:
        for y in pred.index:
            if pred[y] >= 5000:
                anio_saturacion = y
                val_saturacion = pred[y]
                break
    
    if anio_saturacion is not None:
        ax.scatter([anio_saturacion], [val_saturacion], color='red', s=150, zorder=10, edgecolors='white')
        texto_sat = f"¡SATURACIÓN!\nAño {int(anio_saturacion)}"
        offset_y = 600 if val_saturacion < 10000 else -1500
        ax.annotate(texto_sat, xy=(anio_saturacion, val_saturacion), 
                    xytext=(anio_saturacion, val_saturacion + offset_y),
                    arrowprops=dict(facecolor='red', shrink=0.05),
                    color='red', fontweight='bold', ha='center')

    ax.set_ylabel("Flujo Vehicular (veh/día)") # También corregí el eje Y
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # --- DIAGNÓSTICO ---
    st.subheader("📋 Diagnóstico Técnico y Recomendaciones")
    
    carpeta_up = carpeta.upper()
    calzada_up = calzada_info.upper()
    
    es_no_pavimentado = any(x in carpeta_up for x in ["TIERRA", "RIPIO", "GRAVA", "SUELO"])
    es_pavimentado = not es_no_pavimentado
    es_doble_via = "DOBLE" in calzada_up or "DOBLE" in carpeta_up

    if es_no_pavimentado:
        if tmda_24 > 300:
            st.error(f"🔴 **PRIORIDAD ALTA:** Camino granular con {int(tmda_24)} veh/día. Supera norma. **Se recomienda Pavimentación.**")
        else:
            st.success(f"🟢 **CONSERVACIÓN:** Tránsito bajo ({int(tmda_24)} veh/día). Mantener perfilado.")
            
    elif es_pavimentado:
        if not es_doble_via:
            if tmda_24 > 5000:
                st.error(f"🔴 **SATURACIÓN:** Vía simple con {int(tmda_24)} veh/día. **Se sugiere Estudio de Segunda Calzada.**")
            elif tmda_26 > 5000:
                st.warning(f"🟡 **ALERTA:** Se proyecta saturación el año {anio_saturacion}. **Planificar ampliación.**")
            else:
                st.success("🟢 **OPERACIÓN NORMAL:** Capacidad suficiente.")
        else:
            st.success("🟢 **ESTÁNDAR ADECUADO:** Doble Calzada acorde al flujo.")

    # Footer
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; color: #888;'><small>Creado por José Tapia - Tesis Ingeniería Civil</small></div>", unsafe_allow_html=True)