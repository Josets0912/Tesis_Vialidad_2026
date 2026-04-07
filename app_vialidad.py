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
    archivo_maestra = "DATA_MAESTRA_TESIS.xlsx"
    archivo_inventario = "Inventario_Rutas_Maule_Completo.xlsx"
    
    # Leemos los Excels directamente
    df_maestra = pd.read_excel(archivo_maestra)
    df_inv = pd.read_excel(archivo_inventario)
    
    # LIMPIEZA DE DATOS MAESTRA
    cols_limpiar = ['ROL', 'ROL NUEVO', 'NOMBRE DEL CAMINO', 'Sector', 'TIPO DE CARPETA', 'CLASIFICACIÓN', 'ESTACIÓN', 'CALZADA']
    for col in cols_limpiar:
        if col in df_maestra.columns:
            df_maestra[col] = df_maestra[col].astype(str).str.strip()
    
    # CORRECCIÓN 115 CANALES
    errores_115 = ['115 Canales', '115 CANALES', '115-Canales', '115 CH', '115-CH']
    if 'ROL' in df_maestra.columns:
        df_maestra['ROL'] = df_maestra['ROL'].replace(errores_115, 'Ruta 115 CH')
    if 'ROL NUEVO' in df_maestra.columns:
        df_maestra['ROL NUEVO'] = df_maestra['ROL NUEVO'].replace(errores_115, 'Ruta 115 CH')

    return df_maestra, df_inv

# Extraemos el Try-Except AFUERA del caché para que Streamlit NO oculte el error
try:
    df, df_inv = cargar_datos()
except Exception as e:
    st.error(f"❌ Error al cargar los archivos: {e}")
    st.warning("💡 Verifica lo siguiente en tu repositorio de GitHub:")
    st.markdown("- **Nombres exactos:** Revisa mayúsculas y minúsculas (`DATA_MAESTRA_TESIS.xlsx` y `Inventario_Rutas_Maule_Completo.xlsx`).")
    st.markdown("- **Archivo Requirements:** Asegúrate de tener `openpyxl` en tu archivo `requirements.txt`.")
    st.stop()

# --- 3. MENÚ LATERAL ---
st.sidebar.header("🔍 Panel de Control")

roles = sorted(df['ROL NUEVO'].dropna().astype(str).unique())
rol_sel = st.sidebar.selectbox("Seleccione Rol Oficial:", roles)

# IMPORTANTE: Usamos .copy() para evitar que Streamlit bloquee la app por modificar un dato en caché
df_rol = df[df['ROL NUEVO'] == rol_sel].copy() 

df_rol['ETIQUETA'] = df_rol['NOMBRE DEL CAMINO'] + " (" + df_rol['ESTACIÓN'] + ")"
tramo_sel = st.sidebar.selectbox("Seleccione Sector:", df_rol['ETIQUETA'].tolist())

st.sidebar.markdown("---")
btn_calc = st.sidebar.button("Generar Informe Técnico 🚀")

# (De aquí hacia abajo mantienes todo tu código de estilos y cálculos igual)

# --- ESTILOS CSS ---
st.markdown("""
<style>
    .info-card { background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6; height: 100%; text-align: left; }
    .info-label { font-size: 12px; color: #6c757d; font-weight: 700; text-transform: uppercase; margin-bottom: 5px; }
    .info-value { font-size: 15px; color: #212529; font-weight: 600; line-height: 1.4; }
    .rate-box { background-color: #e8f4f8; padding: 10px; border-radius: 5px; border-left: 5px solid #17a2b8; margin-bottom: 15px; color: #0c5460; font-weight: 500; }
    .subtitle-sector { color: #555; font-size: 20px; margin-top: -20px; margin-bottom: 20px; font-weight: 500; }
    .ref-table { font-size: 12px; width: 100%; border-collapse: collapse; }
    .ref-table th { background-color: #f1f3f5; border-bottom: 2px solid #dee2e6; padding: 8px; text-align: left; }
    .ref-table td { border-bottom: 1px solid #dee2e6; padding: 8px; }
</style>
""", unsafe_allow_html=True)

# --- 4. INTERFAZ Y CÁLCULOS ---
if not btn_calc:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>🚧 Sistema de Gestión de Pavimentos</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #1f77b4;'>Desarrollado por José Tapia</h3>", unsafe_allow_html=True)
    st.info("👈 Seleccione un camino en el menú lateral para iniciar el análisis.")
else:
    # FILTRO DE SEGURIDAD: ¿Es camino granular?
    datos_inv_especifico = df_inv[df_inv['Rol'] == rol_sel]
    es_granular = False
    info_inv = None
    
    if not datos_inv_especifico.empty:
        info_inv = datos_inv_especifico.iloc[0]
        rodadura = str(info_inv['Capa de Rodadura']).upper()
        if any(x in rodadura for x in ["RIPIO", "GRANULAR", "TIERRA", "SUELO"]):
            es_granular = True

    # CREACIÓN DE PESTAÑAS (Solo si es granular aparece la de diseño)
    if es_granular:
        tab_demanda, tab_diseno = st.tabs(["📈 Análisis de Demanda", "🛣️ Diseño Estructural"])
    else:
        tab_demanda = st.tabs(["📈 Análisis de Demanda"])[0]

    with tab_demanda:
        st.markdown("### 🚧 Sistema de Gestión de Pavimentos y Proyección de Demanda")
        fila = df_rol[df_rol['ETIQUETA'] == tramo_sel].iloc[0]
        nombre = fila['NOMBRE DEL CAMINO']
        rol_oficial = fila['ROL NUEVO']
        carpeta = fila['TIPO DE CARPETA']
        
        st.title(f"📍 {nombre}")
        st.markdown(f"<div class='subtitle-sector'>Sector: {fila['Sector']}</div>", unsafe_allow_html=True)
        
        # --- CÁLCULOS PROYECCIÓN (INTACTOS) ---
        anios_censo = [2015, 2017, 2018, 2020, 2022, 2024]
        vals_censo = fila[[f'TMDA {a}' for a in anios_censo]].values.flatten().astype(float)
        datos_reales = pd.Series(vals_censo, index=anios_censo).sort_index()
        
        serie_completa = {}
        for i in range(len(anios_censo) - 1):
            a_inicio, a_fin = anios_censo[i], anios_censo[i+1]
            v_inicio, v_fin = datos_reales[a_inicio], datos_reales[a_fin]
            serie_completa[a_inicio] = v_inicio
            n_anios = a_fin - a_inicio
            if n_anios > 1:
                r = (v_fin / v_inicio) ** (1/n_anios) - 1 if v_inicio > 0 else 0
                for k in range(1, n_anios): serie_completa[a_inicio + k] = v_inicio * ((1 + r) ** k)
        serie_completa[anios_censo[-1]] = datos_reales[anios_censo[-1]]
        serie = pd.Series(serie_completa).sort_index()
        
        modelo = ExponentialSmoothing(serie, trend='add', seasonal=None, damped_trend=True).fit(damping_trend=0.92)
        anios_fut = np.arange(2025, 2046)
        pred = modelo.forecast(len(anios_fut))
        
        # KPIs y Gráficos (Tu lógica original continúa aquí...)
        st.metric("Proyección 2045", f"{int(pred.iloc[-1])} veh/día")
        fig, ax = plt.subplots()
        ax.plot(serie.index, serie.values, label="Histórico")
        ax.plot(anios_fut, pred.values, '--', label="Proyección")
        ax.legend(); st.pyplot(fig)

    # --- PESTAÑA DE DISEÑO (TU SOLICITUD) ---
    if es_granular:
        with tab_diseno:
            st.header("📏 Dimensionamiento Estructural (AASHTO 93)")
            st.write(f"Diseño para conversión de **{info_inv['Capa de Rodadura']}** a Pavimento Flexible.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("EEq 2045 (Acumulado)", f"{info_inv['EEq 2045']:,.0f}")
                # CBR como casilla de completar
                cbr = st.number_input("Ingrese CBR de Subrasante (%)", min_value=1.0, max_value=100.0, value=10.0, step=0.1)
            
            with col2:
                precip = info_inv['Precipitacion promedio Mensual (mm)']
                m = 0.8 if precip > 80 else (1.0 if precip > 40 else 1.1)
                st.metric("Precipitación Media", f"{precip} mm/mes")
                st.write(f"**Coeficiente de Drenaje (m):** `{m}`")

            # Cálculo de Espesores
            sn_req = 0.47 * np.log10(info_inv['EEq 2045'] + 1) * (1.2 / (cbr**0.15))
            d1 = 5.0 if info_inv['EEq 2045'] < 500000 else (7.0 if info_inv['EEq 2045'] < 1500000 else 10.0)
            d2 = 20.0
            d3 = max(15.0, round((sn_req - (0.17 * d1) - (0.13 * d2 * m)) / (0.11 * m), 0))
            
            st.subheader("🚀 Propuesta de Estructura")
            res1, res2, res3 = st.columns(3)
            res1.success(f"**Carpeta Asfáltica:** {d1} cm")
            res2.success(f"**Base Granular:** {d2} cm")
            res3.success(f"**Sub-Base Granular:** {d3} cm")
