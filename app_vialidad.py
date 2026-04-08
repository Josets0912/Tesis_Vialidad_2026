import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

warnings.filterwarnings("ignore")

# --- 1. CONFIGURACIÓN Y MEMORIA ---
st.set_page_config(page_title="Gestión Vial - Tesis", layout="wide", initial_sidebar_state="expanded")

if "informe_generado" not in st.session_state:
    st.session_state.informe_generado = False
if "d1_val" not in st.session_state: st.session_state.d1_val = 5.0
if "d2_val" not in st.session_state: st.session_state.d2_val = 15.0
if "d3_val" not in st.session_state: st.session_state.d3_val = 15.0

# --- FUNCIONES MATEMÁTICAS ---
@st.cache_data
def cargar_datos():
    df_maestra = pd.read_excel("DATA_MAESTRA_TESIS.xlsx")
    df_inv = pd.read_excel("Inventario_Rutas_Maule_Completo.xlsx")
    cols_limpiar = ['ROL', 'ROL NUEVO', 'NOMBRE DEL CAMINO', 'Sector', 'TIPO DE CARPETA', 'CLASIFICACIÓN', 'ESTACIÓN', 'CALZADA']
    for col in cols_limpiar:
        if col in df_maestra.columns: df_maestra[col] = df_maestra[col].astype(str).str.strip()
    if 'Rol' in df_inv.columns: df_inv['Rol'] = df_inv['Rol'].astype(str).str.strip()
    return df_maestra, df_inv

@st.cache_data
def calcular_proyeccion(serie_datos):
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
    except:
        return None

def calcular_ee_soportado(d1, d2, d3, a1, a2, a3, m2, m3, zr_val, so_val, pi_val, pf_val, mr_val_mpa):
    """Cálculo exacto de tu macro. NE solo depende de espesores y coeficientes."""
    ne_mm = (d1 * 10 * 1 * a1) + (d2 * 10 * m2 * a2) + (d3 * 10 * m3 * a3)
    try:
        f6_beta = 0.4 + (97.81 / (ne_mm + 25.4)) ** 5.19
        
        # === AQUÍ ESTÁ TU FÓRMULA EXACTA DEL EXCEL ===
        # POTENCIA(Ne + 25.4, 9.36) * POTENCIA(10, -16.4+Zr*So) * Potencia(Mr, 2.32) * Potencia(((Pi - Pf)/(Pi - 1.5)), 1/beta)
        ee = ((ne_mm + 25.4) ** 9.36) * (10 ** (-16.4 + (zr_val * so_val))) * (mr_val_mpa ** 2.32) * (((pi_val - pf_val) / (pi_val - 1.5)) ** (1 / f6_beta))
        return ee
    except: 
        return 0

def optimizar_espesores_vba(ee_req, a1, a2, a3, m2, m3, zr_val, so_val, pi_val, pf_val, mr_val_mpa):
    for sumaTotal in range(35, 161):
        for hAsf in range(5, 7):
            for hBase in range(15, 51):
                hSub = sumaTotal - hAsf - hBase
                if 15 <= hSub <= 80:
                    ee_dis = calcular_ee_soportado(hAsf, hBase, hSub, a1, a2, a3, m2, m3, zr_val, so_val, pi_val, pf_val, mr_val_mpa)
                    if ee_dis >= ee_req: 
                        return float(hAsf), float(hBase), float(hSub), ee_dis
    return None, None, None, None

# --- CARGA Y SIDEBAR ---
try:
    df, df_inv = cargar_datos()
except Exception as e:
    st.error(f"❌ Error al cargar los archivos: {e}")
    st.stop()

st.sidebar.header("🔍 Panel de Control")
roles = sorted(df['ROL NUEVO'].dropna().astype(str).unique())
rol_sel = st.sidebar.selectbox("Seleccione Rol Oficial:", roles)
df_rol = df[df['ROL NUEVO'] == rol_sel].copy()
df_rol['ETIQUETA'] = df_rol['NOMBRE DEL CAMINO'] + " (" + df_rol['ESTACIÓN'] + ")"
tramo_sel = st.sidebar.selectbox("Seleccione Sector:", df_rol['ETIQUETA'].tolist())

st.sidebar.markdown("---")
if st.sidebar.button("Generar Informe Técnico 🚀"): 
    st.session_state.informe_generado = True

st.markdown("<style>.info-card{background-color:#f8f9fa; padding:15px; border-radius:8px; border:1px solid #dee2e6; text-align:left;} .verdict-ok{background-color:#d4edda; color:#155724; padding:15px; border-radius:5px; font-weight:bold; border-left:8px solid #28a745; font-size:16px; text-align:center;} .verdict-bad{background-color:#f8d7da; color:#721c24; padding:15px; border-radius:5px; font-weight:bold; border-left:8px solid #dc3545; font-size:16px; text-align:center;} .sn-box{background-color:#e2e3e5; padding:10px; border-radius:5px; text-align:center; border:1px solid #ced4da; margin-top:10px;}</style>", unsafe_allow_html=True)

if not st.session_state.informe_generado:
    st.markdown("<br><br><h1 style='text-align:center;'>🚧 Gestión de Pavimentos</h1><h3 style='text-align:center; color:#1f77b4;'>José Tapia</h3>", unsafe_allow_html=True)
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
    info_inv = df_inv[df_inv['Rol'] == rol_sel].iloc[0] if not df_inv[df_inv['Rol'] == rol_sel].empty else None

    tab_demanda, tab_diseno = st.tabs(["📈 Análisis de Demanda", "🛣️ Diseño Estructural (AASHTO 93)"])

    # --- PESTAÑA DEMANDA (SIN CAMBIOS) ---
    with tab_demanda:
        st.markdown("### 🚧 Sistema de Gestión de Pavimentos y Proyección de Demanda")
        st.title(f"📍 {nombre}")
        
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
        
        pred = calcular_proyeccion(serie)
        if pred is not None:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(serie.index, serie.values, '-', color='gray', alpha=0.4)
            ax.scatter(anios_censo, [serie[a] for a in anios_censo], color='black', s=60)
            ax.plot([2024] + list(pred.index), [serie[2024]] + list(pred.values), '--.', color='#2ca02c')
            ax.axhline(5000, color='gray', linestyle=':', alpha=0.5)
            st.pyplot(fig)

    # --- PESTAÑA DISEÑO ESTRUCTURAL (CON FÓRMULA CORREGIDA) ---
    with tab_diseno:
        if not es_granular:
            st.warning(f"⚠️ Este camino ya cuenta con una superficie de **{carpeta}**. El diseño estructural inicial está deshabilitado.")
        elif info_inv is None:
            st.warning("⚠️ No existen datos de Ejes Equivalentes (EEq) para este Rol en el inventario.")
        else:
            st.header("📏 MÉTODO AASHTO 93 - DISEÑO ESTRUCTURAL")
            st.markdown("---")

            col_izq, col_der = st.columns([1, 1])

            # DATOS DE ENTRADA GENERALES
            with col_izq:
                eeq_val = info_inv['EEq 2045']
                st.metric("Tráfico Proyectado (EES)", f"{eeq_val:,.0f} EEq")
                
                cbr_subrasante = st.number_input("C.B.R. de la Subrasante (%)", min_value=1.0, max_value=100.0, value=25.0, step=0.1)
                if cbr_subrasante < 12: mr_calc_mpa = 17.6 * (cbr_subrasante ** 0.64)
                else: mr_calc_mpa = 22.1 * (cbr_subrasante ** 0.55)
                mr_val_mpa = st.number_input("Módulo Resiliente (MR) en MPa", value=float(round(mr_calc_mpa, 2)))

            with col_der:
                st.markdown("**Parámetros de Diseño (Factores AASHTO)**")
                confiabilidad_pct = st.number_input("Confiabilidad (R) %", min_value=50.0, value=95.0, step=1.0)
                dict_zr = {50: 0.0, 75: -0.674, 80: -0.841, 85: -1.036, 90: -1.282, 95: -1.645, 99: -2.327}
                zr_val = dict_zr[min(dict_zr.keys(), key=lambda k: abs(k - confiabilidad_pct))]
                so_val = st.number_input("Desviación Estándar (So)", value=0.50, step=0.01)
                
                # SEPARACIÓN DE SERVICIABILIDAD INICIAL Y FINAL (Para respetar tu fórmula)
                col_pi, col_pf = st.columns(2)
                with col_pi:
                    pi_val = st.number_input("Serv. Inicial (pi)", value=4.2, step=0.1)
                with col_pf:
                    pf_val = st.number_input("Serv. Final (pf)", value=2.0, step=0.1)

            st.markdown("---")

            # MATERIALES Y MACRO
            st.subheader("⚙️ Materiales y Cálculo Automático")
            col_mat1, col_mat2, col_opt = st.columns([1, 1, 1.2])
            
            with col_mat1:
                a1 = st.number_input("Coef. Asfalto a1 (1/mm)", value=0.197, format="%.3f")
                a2 = st.number_input("Coef. Base a2 (1/mm)", value=0.090, format="%.3f")
                a3 = st.number_input("Coef. Subbase a3 (1/mm)", value=0.090, format="%.3f")
            
            with col_mat2:
                precip = info_inv['Precipitacion promedio Mensual (mm)']
                m_sugerido = 0.8 if precip > 80 else (1.0 if precip > 40 else 1.1)
                m2 = st.number_input(f"Coef. Drenaje Base m2", value=m_sugerido, format="%.2f")
                m3 = st.number_input(f"Coef. Drenaje Subbase m3", value=m_sugerido, format="%.2f")

            with col_opt:
                st.markdown("<br>", unsafe_allow_html=True)
                st.info("💡 Ejecuta la macro para buscar la combinación óptima de espesores.")
                if st.button("🔄 Ejecutar Optimización (Macro)"):
                    opt_d1, opt_d2, opt_d3, opt_ee = optimizar_espesores_vba(
                        eeq_val, a1, a2, a3, m2, m3, zr_val, so_val, pi_val, pf_val, mr_val_mpa
                    )
                    if opt_d1 is not None:
                        st.session_state.d1_val, st.session_state.d2_val, st.session_state.d3_val = opt_d1, opt_d2, opt_d3
                        st.success("✅ ¡Diseño óptimo encontrado!")
                        st.rerun()
                    else:
                        st.error("❌ No se encontró solución factible en los rangos.")

            st.markdown("---")

            # ESPESORES Y VISUALIZACIÓN
            st.subheader("🏗️ Propuesta Estructural Interactiva")
            col_esp, col_graf = st.columns([1, 1.5])

            with col_esp:
                d1 = st.number_input("D1 (Carpeta) cm", value=st.session_state.d1_val, step=0.5, key="inp_d1")
                d2 = st.number_input("D2 (Base) cm", value=st.session_state.d2_val, step=0.5, key="inp_d2")
                d3 = st.number_input("D3 (Subbase) cm", value=st.session_state.d3_val, step=0.5, key="inp_d3")
                
                st.session_state.d1_val, st.session_state.d2_val, st.session_state.d3_val = d1, d2, d3

                # EL CÁLCULO DIRECTO DEL NE (Celda F5)
                ne_aportado = (d1 * 10 * 1 * a1) + (d2 * 10 * m2 * a2) + (d3 * 10 * m3 * a3)
                
                st.markdown(f"<div class='sn-box'><h4>NE Aportado (Celda F5): <b>{ne_aportado:.2f} mm</b></h4></div>", unsafe_allow_html=True)

                ee_soportado = calcular_ee_soportado(d1, d2, d3, a1, a2, a3, m2, m3, zr_val, so_val, pi_val, pf_val, mr_val_mpa)
                holgura = ee_soportado - eeq_val if ee_soportado > eeq_val else 0

                st.markdown("<br>", unsafe_allow_html=True)
                if ee_soportado >= eeq_val:
                    st.markdown(f"<div class='verdict-ok'>✅ APROBADO<br><span style='font-size:14px; font-weight:normal;'>Soporta: {ee_soportado:,.0f} EEq<br>Holgura: +{holgura:,.0f} EEq</span></div>", unsafe_allow_html=True)
                else:
                    deficit = eeq_val - ee_soportado
                    st.markdown(f"<div class='verdict-bad'>⚠️ INSUFICIENTE<br><span style='font-size:14px; font-weight:normal;'>Soporta solo: {ee_soportado:,.0f} EEq<br>Faltan: {deficit:,.0f} EEq</span></div>", unsafe_allow_html=True)

            with col_graf:
                h_d1 = max(40, d1 * 4.5) 
                h_d2 = max(50, d2 * 3.5)
                h_d3 = max(50, d3 * 3.0)
                
                try: beta_calculado = 0.4 + (97.81 / (ne_aportado + 25.4)) ** 5.19
                except: beta_calculado = 1.0
                
                html_capas = f"""
                <div style="width: 100%; max-width: 400px; margin: auto; border: 3px solid #2c3e50; border-radius: 6px; overflow: hidden; text-align: center; font-family: 'Segoe UI', sans-serif; box-shadow: 0 6px 12px rgba(0,0,0,0.15);">
                    <div style="background: linear-gradient(180deg, #595959 0%, #3b3b3b 100%); color: white; height: {h_d1}px; display: flex; align-items: center; justify-content: center; font-weight: bold;">
                        Carpeta Asfáltica ({d1} cm)
                    </div>
                    <div style="background: linear-gradient(180deg, #e3c988 0%, #d4b872 100%); color: #333; height: {h_d2}px; display: flex; align-items: center; justify-content: center; font-weight: bold; border-top: 2px solid #2c3e50;">
                        Base Granular ({d2} cm)
                    </div>
                    <div style="background: linear-gradient(180deg, #b8865b 0%, #a67c52 100%); color: white; height: {h_d3}px; display: flex; align-items: center; justify-content: center; font-weight: bold; border-top: 2px solid #2c3e50;">
                        Subbase Granular ({d3} cm)
                    </div>
                    <div style="background: linear-gradient(180deg, #6e4e37 0%, #4a3322 100%); color: #e0e0e0; height: 90px; display: flex; flex-direction: column; align-items: center; justify-content: center; font-weight: bold; border-top: 4px dashed #1a110b;">
                        <div>Suelo Subrasante (CBR {cbr_subrasante}%)</div>
                        <div style="font-size: 11px; font-weight: normal; margin-top: 3px; opacity: 0.8;">Beta (β): {beta_calculado:.3f}</div>
                    </div>
                </div>
                """
                st.markdown(html_capas, unsafe_allow_html=True)
                st.caption("Gráfico Estratigráfico (Se ajusta en tiempo real)")
