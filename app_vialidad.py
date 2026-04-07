import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

warnings.filterwarnings("ignore")

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Gestión Vial - Tesis José Tapia", layout="wide", initial_sidebar_state="expanded")

# --- FUNCIONES CACHEADAS (Para evitar que la página se reinicie/congele) ---
@st.cache_data
def cargar_datos():
    df_maestra = pd.read_excel("DATA_MAESTRA_TESIS.xlsx")
    df_inv = pd.read_excel("Inventario_Rutas_Maule_Completo.xlsx")
    
    cols_limpiar = ['ROL', 'ROL NUEVO', 'NOMBRE DEL CAMINO', 'Sector', 'TIPO DE CARPETA', 'CLASIFICACIÓN', 'ESTACIÓN', 'CALZADA']
    for col in cols_limpiar:
        if col in df_maestra.columns:
            df_maestra[col] = df_maestra[col].astype(str).str.strip()
            
    if 'Rol' in df_inv.columns:
        df_inv['Rol'] = df_inv['Rol'].astype(str).str.strip()
    
    errores_115 = ['115 Canales', '115 CANALES', '115-Canales', '115 CH', '115-CH']
    if 'ROL' in df_maestra.columns: df_maestra['ROL'] = df_maestra['ROL'].replace(errores_115, 'Ruta 115 CH')
    if 'ROL NUEVO' in df_maestra.columns: df_maestra['ROL NUEVO'] = df_maestra['ROL NUEVO'].replace(errores_115, 'Ruta 115 CH')
    return df_maestra, df_inv

@st.cache_data
def calcular_proyeccion(serie_datos):
    """Esta función evita que el modelo Holt-Winters se recalcule al mover un slider"""
    try:
        try:
            modelo = ExponentialSmoothing(serie_datos, trend='mul', seasonal=None, damped_trend=True).fit(damping_trend=0.92)
        except:
            modelo = ExponentialSmoothing(serie_datos, trend='add', seasonal=None, damped_trend=True).fit(damping_trend=0.92)
            
        anios_fut = np.arange(2025, 2046)
        pred_raw = pd.Series(modelo.forecast(len(anios_fut)).values, index=anios_fut)
        
        tasa_crecimiento_inicial = (pred_raw.iloc[1] / pred_raw.iloc[0]) if pred_raw.iloc[0] > 0 and pred_raw.iloc[1] > 0 else 1.0
        base_teorica_modelo = pred_raw.iloc[0] / tasa_crecimiento_inicial
        ultimo_real = serie_datos.iloc[-1]
        factor_ajuste = ultimo_real / base_teorica_modelo if base_teorica_modelo > 0 else 1.0
        pred_escalada = pred_raw * factor_ajuste
        
        pred_ajustada = []
        piso = ultimo_real 
        for y in anios_fut:
            val = pred_escalada[y]
            if val < piso: val = piso
            else: piso = val
            pred_ajustada.append(val)
            
        return pd.Series(pred_ajustada, index=anios_fut)
    except Exception as e:
        return None

def resolver_sn_aashto(W18, ZR, So, dPSI, MR):
    """Algoritmo matemático para resolver la ecuación AASHTO 93 y obtener el SN Requerido"""
    if W18 <= 0 or MR <= 0: return 0.1
    sn_min, sn_max = 0.1, 20.0
    for _ in range(50): # Búsqueda Binaria
        sn_guess = (sn_min + sn_max) / 2.0
        term1 = 9.36 * math.log10(sn_guess + 1) - 0.20
        term2 = math.log10(dPSI / 2.7) / (0.40 + (1094 / ((sn_guess + 1) ** 5.19)))
        term3 = 2.32 * math.log10(MR) - 8.07
        log_W18_guess = (ZR * So) + term1 + term2 + term3
        
        if (10 ** log_W18_guess) > W18: sn_max = sn_guess
        else: sn_min = sn_guess
    return (sn_min + sn_max) / 2.0

# --- 2. CARGA DE ARCHIVOS ---
try:
    df, df_inv = cargar_datos()
except Exception as e:
    st.error(f"❌ Error al cargar los archivos: {e}")
    st.stop()

# --- 3. MENÚ LATERAL ---
st.sidebar.header("🔍 Panel de Control")
roles = sorted(df['ROL NUEVO'].dropna().astype(str).unique())
rol_sel = st.sidebar.selectbox("Seleccione Rol Oficial:", roles)

df_rol = df[df['ROL NUEVO'] == rol_sel].copy()
df_rol['ETIQUETA'] = df_rol['NOMBRE DEL CAMINO'] + " (" + df_rol['ESTACIÓN'] + ")"
tramo_sel = st.sidebar.selectbox("Seleccione Sector:", df_rol['ETIQUETA'].tolist())

st.sidebar.markdown("---")
btn_calc = st.sidebar.button("Generar Informe Técnico 🚀")

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
    .verdict-ok { background-color: #d4edda; color: #155724; padding: 15px; border-radius: 5px; font-weight: bold; border-left: 8px solid #28a745; font-size: 18px; text-align: center;}
    .verdict-bad { background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px; font-weight: bold; border-left: 8px solid #dc3545; font-size: 18px; text-align: center;}
    .sn-box { background-color: #e2e3e5; padding: 10px; border-radius: 5px; text-align: center; border: 1px solid #ced4da; }
</style>
""", unsafe_allow_html=True)

if not btn_calc:
    st.markdown("<br><br><h1 style='text-align: center;'>🚧 Sistema de Gestión de Pavimentos</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #1f77b4;'>Desarrollado por José Tapia</h3>", unsafe_allow_html=True)
    st.info("👈 Seleccione un camino en el menú lateral para iniciar el análisis.")
else:
    fila = df_rol[df_rol['ETIQUETA'] == tramo_sel].iloc[0]
    nombre = fila['NOMBRE DEL CAMINO']
    rol_oficial = fila['ROL NUEVO']
    carpeta = fila['TIPO DE CARPETA']
    clasificacion = fila['CLASIFICACIÓN']
    calzada_info = fila['CALZADA'] if 'CALZADA' in fila else "No Inf"
    sector_especifico = fila['Sector']
    
    rodadura_maestra = str(carpeta).upper()
    es_granular = any(x in rodadura_maestra for x in ["RIPIO", "GRANULAR", "TIERRA", "SUELO", "NATURAL"])
    
    datos_inv_especifico = df_inv[df_inv['Rol'] == rol_sel].copy()
    info_inv = datos_inv_especifico.iloc[0] if not datos_inv_especifico.empty else None

    # ESTABILIDAD DE PESTAÑAS: Siempre se crean ambas
    tab_demanda, tab_diseno = st.tabs(["📈 Análisis de Demanda", "🛣️ Diseño Estructural (AASHTO)"])

    # =========================================================================
    # PESTAÑA 1: DEMANDA
    # =========================================================================
    with tab_demanda:
        st.markdown("### 🚧 Sistema de Gestión de Pavimentos y Proyección de Demanda")
        st.title(f"📍 {nombre}")
        st.markdown(f"<div class='subtitle-sector'>Sector: {sector_especifico}</div>", unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(f"<div class='info-card'><div class='info-label'>Rol Oficial</div><div class='info-value'>{rol_oficial}</div></div>", unsafe_allow_html=True)
        with c2: st.markdown(f"<div class='info-card'><div class='info-label'>Tipo de Carpeta</div><div class='info-value'>{carpeta}</div></div>", unsafe_allow_html=True)
        with c3: st.markdown(f"<div class='info-card'><div class='info-label'>Clasificación</div><div class='info-value'>{clasificacion}</div></div>", unsafe_allow_html=True)
        with c4: st.markdown(f"<div class='info-card'><div class='info-label'>Calzada</div><div class='info-value'>{calzada_info}</div></div>", unsafe_allow_html=True)
        
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
        
        # LLAMADA A LA FUNCIÓN CACHEADA (Para máxima velocidad)
        pred = calcular_proyeccion(serie)
        
        if pred is not None:
            tmda_24 = serie[2024]
            tmda_26 = pred[2026]
            tmda_45 = pred[2045]
            
            tasa_24_26 = ((tmda_26 / tmda_24) ** (1/2) - 1) * 100 if tmda_24 > 0 and tmda_26 > 0 else 0
            tasa_26_45 = ((tmda_45 / tmda_26) ** (1/19) - 1) * 100 if tmda_26 > 0 and tmda_45 > 0 else 0

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<div class='rate-box'>📊 Tasa Promedio Anual (2024-2026): <b>{tasa_24_26:.2f}%</b> &nbsp;|&nbsp; Tasa Promedio Anual (2026-2045): <b>{tasa_26_45:.2f}%</b></div>", unsafe_allow_html=True)
            
            colA, colB, colC = st.columns(3)
            colA.metric("🚗 Censo 2024", f"{int(tmda_24)} veh/día")
            colB.metric("📈 Proyección 2026", f"{int(tmda_26)} veh/día")
            colC.metric("🔭 Proyección 2045", f"{int(tmda_45)} veh/día")

            st.subheader("Evolución de la Demanda y Umbrales")
            fig, ax = plt.subplots(figsize=(10, 5))
            
            x_interp = [a for a in serie.index if a not in anios_censo]
            y_interp = [serie[a] for a in x_interp]
            x_real = anios_censo
            y_real = [serie[a] for a in x_real if a in serie.index]
            
            ax.plot(serie.index, serie.values, '-', color='gray', alpha=0.4, linewidth=1)
            ax.scatter(x_interp, y_interp, color='#fd7e14', s=40, label='Interpolado (Geométrico)', zorder=5)
            ax.scatter(x_real, y_real, color='black', s=60, label='Censo Oficial', zorder=10)
            
            x_proyeccion = [2024] + list(pred.index)
            y_proyeccion = [serie[2024]] + list(pred.values)
            ax.plot(x_proyeccion, y_proyeccion, '--.', color='#2ca02c', linewidth=1, markersize=4, label='Proyección (Holt)')
            
            ax.axhline(5000, color='gray', linestyle=':', alpha=0.5, label='Umbral 5.000')
            ax.set_ylabel("Flujo Vehicular (veh/día)")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.error("Error al calcular proyecciones.")

    # =========================================================================
    # PESTAÑA 2: DISEÑO ESTRUCTURAL (NUEVO DASHBOARD TIPO AASHTO)
    # =========================================================================
    with tab_diseno:
        if not es_granular:
            st.warning(f"⚠️ Este camino ya cuenta con una superficie de **{carpeta}**. El diseño estructural inicial está deshabilitado.")
        elif info_inv is None:
            st.warning("⚠️ No existen datos de Ejes Equivalentes (EEq) para este Rol en el inventario.")
        else:
            st.header("📏 MÉTODO AASHTO 93 - DISEÑO DE PAVIMENTO FLEXIBLE")
            st.markdown("---")

            col_izq, col_der = st.columns([1, 1])

            # --- COLUMNA IZQUIERDA: DATOS DE ENTRADA ---
            with col_izq:
                st.subheader("📝 DATOS DE ENTRADA")
                
                eeq_val = info_inv['EEq 2045']
                st.text_input("Tráfico en Ejes Equivalentes (EES)", value=f"{eeq_val:,.0f}", disabled=True)
                
                confiabilidad_pct = st.number_input("Nivel de Confiabilidad (R) %", min_value=50.0, max_value=99.9, value=95.0, step=1.0)
                
                # Conversión de Confiabilidad a Desviación Normal (Zr)
                dict_zr = {50: 0.0, 75: -0.674, 80: -0.841, 85: -1.036, 90: -1.282, 95: -1.645, 99: -2.327}
                # Buscar el más cercano si no es exacto
                zr = min(dict_zr.keys(), key=lambda k: abs(k - confiabilidad_pct))
                zr_val = dict_zr[zr]

                so_val = st.number_input("Desviación Estándar (So)", min_value=0.1, max_value=0.9, value=0.45, step=0.01)
                dpsi_val = st.number_input("Pérdida de Serviciabilidad (ΔPSI)", min_value=0.5, max_value=4.0, value=2.2, step=0.1)
                
                cbr_subrasante = st.number_input("C.B.R. de la Subrasante (%)", min_value=1.0, max_value=100.0, value=4.0, step=0.1)
                
                # Cálculo de Módulo Resiliente (Ecuación típica chilena/AASHTO)
                mr_calc = 2555 * (cbr_subrasante ** 0.64)
                mr_val = st.number_input("Módulo Resiliente (MR) psi", value=float(round(mr_calc, 2)))

                # Cálculo de NE Requerido Interno
                ne_req = resolver_sn_aashto(eeq_val, zr_val, so_val, dpsi_val, mr_val)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"<div class='sn-box'><h4>NE Requerido: <b>{ne_req:.2f}</b></h4></div>", unsafe_allow_html=True)

            # --- COLUMNA DERECHA: PROPIEDADES DE MATERIALES ---
            with col_der:
                st.subheader("🧱 PROPIEDADES DE LOS MATERIALES")
                
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    a1 = st.number_input("Coef. Estructural a1", value=0.170, step=0.005, format="%.3f")
                    a2 = st.number_input("Coef. Estructural a2", value=0.130, step=0.005, format="%.3f")
                    a3 = st.number_input("Coef. Estructural a3", value=0.110, step=0.005, format="%.3f")
                with col_m2:
                    st.markdown("<br><br><br>", unsafe_allow_html=True) # Espaciador para alinear
                    m2 = st.number_input("Coef. Drenaje m2", value=1.00, step=0.05, format="%.2f")
                    m3 = st.number_input("Coef. Drenaje m3", value=1.00, step=0.05, format="%.2f")

            st.markdown("---")

            # --- SECCIÓN INFERIOR: ESPESORES Y RESULTADO ---
            st.subheader("🏗️ ESPESORES PROPUESTOS")
            st.caption("Ajuste los espesores (D1, D2, D3) para cumplir con el NE Requerido. (Nota: Asumiendo fórmula SN = a * D * m)")

            col_e1, col_e2, col_e3, col_res = st.columns([1, 1, 1, 1.5])

            with col_e1:
                d1 = st.number_input("D1 (Carpeta)", value=4.0, step=0.5)
                sn1 = a1 * d1
                st.caption(f"SN1 Aportado: {sn1:.2f}")

            with col_e2:
                d2 = st.number_input("D2 (Base)", value=10.0, step=0.5)
                sn2 = a2 * d2 * m2
                st.caption(f"SN2 Aportado: {sn2:.2f}")

            with col_e3:
                d3 = st.number_input("D3 (Subbase)", value=23.0, step=0.5)
                sn3 = a3 * d3 * m3
                st.caption(f"SN3 Aportado: {sn3:.2f}")

            sn_total = sn1 + sn2 + sn3

            with col_res:
                st.markdown("<div style='text-align: center; color: #555; margin-bottom: 5px;'>SN Total Calculado</div>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='text-align: center; margin-top: 0;'>{sn_total:.2f}</h2>", unsafe_allow_html=True)
                
                if sn_total >= ne_req:
                    st.markdown(f"<div class='verdict-ok'>✅ OK</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='verdict-bad'>⚠️ INSUFICIENTE</div>", unsafe_allow_html=True)
