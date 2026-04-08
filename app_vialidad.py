import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

warnings.filterwarnings("ignore")

# --- 1. CONFIGURACIÓN Y MEMORIA (Evita que la página se reinicie) ---
st.set_page_config(page_title="Gestión Vial - Tesis", layout="wide", initial_sidebar_state="expanded")

if "informe_generado" not in st.session_state:
    st.session_state.informe_generado = False
if "d1_val" not in st.session_state:
    st.session_state.d1_val = 5.0
if "d2_val" not in st.session_state:
    st.session_state.d2_val = 15.0
if "d3_val" not in st.session_state:
    st.session_state.d3_val = 15.0

# --- FUNCIONES MATEMÁTICAS (TRADUCCIÓN EXACTA DE TU EXCEL) ---
@st.cache_data
def cargar_datos():
    df_maestra = pd.read_excel("DATA_MAESTRA_TESIS.xlsx")
    df_inv = pd.read_excel("Inventario_Rutas_Maule_Completo.xlsx")
    
    cols_limpiar = ['ROL', 'ROL NUEVO', 'NOMBRE DEL CAMINO', 'Sector', 'TIPO DE CARPETA', 'CLASIFICACIÓN', 'ESTACIÓN', 'CALZADA']
    for col in cols_limpiar:
        if col in df_maestra.columns:
            df_maestra[col] = df_maestra[col].astype(str).str.strip()
            
    if 'Rol' in df_inv.columns: df_inv['Rol'] = df_inv['Rol'].astype(str).str.strip()
    
    errores_115 = ['115 Canales', '115 CANALES', '115-Canales', '115 CH', '115-CH']
    if 'ROL' in df_maestra.columns: df_maestra['ROL'] = df_maestra['ROL'].replace(errores_115, 'Ruta 115 CH')
    if 'ROL NUEVO' in df_maestra.columns: df_maestra['ROL NUEVO'] = df_maestra['ROL NUEVO'].replace(errores_115, 'Ruta 115 CH')
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

def calcular_ee_soportado(d1, d2, d3, a1, a2, a3, m2, m3, zr_val, so_val, dpsi_val, mr_val_mpa):
    """Traducción exacta de celda B11 de tu Excel. Requiere MR en MPa."""
    sn_mm = (a1 * d1 * 10) + (a2 * d2 * 10 * m2) + (a3 * d3 * 10 * m3)
    try:
        # AQUÍ ESTÁ TU FÓRMULA EXACTA DEL EXCEL (Celda D6)
        f6_beta = 0.4 + (97.81 / (sn_mm + 25.4)) ** 5.19
        ee = ((sn_mm + 25.4) ** 9.36) * (10 ** (-16.4 + (zr_val * so_val))) * (mr_val_mpa ** 2.32) * ((dpsi_val / 2.7) ** (1 / f6_beta))
        return ee
    except:
        return 0

def resolver_sn_aashto_metric(W18, ZR, So, dPSI, MR_MPa):
    """Calcula el NE Requerido exacto invirtiendo tu fórmula B11"""
    if W18 <= 0 or MR_MPa <= 0: return 0.1
    sn_min, sn_max = 0.1, 500.0 # Búsqueda en mm
    for _ in range(60):
        sn_guess_mm = (sn_min + sn_max) / 2.0
        ee_guess = calcular_ee_soportado(sn_guess_mm/10, 0, 0, 1, 0, 0, 1, 1, ZR, So, dPSI, MR_MPa)
        if ee_guess > W18: sn_max = sn_guess_mm
        else: sn_min = sn_guess_mm
    return (sn_min + sn_max) / 2.0

def optimizar_espesores_vba(ee_req, a1, a2, a3, m2, m3, zr_val, so_val, dpsi_val, mr_val_mpa):
    """Tu Macro de VBA traducida a Python (Itera espesores)"""
    for sumaTotal in range(35, 161):
        for hAsf in range(5, 7): # De 5 a 6 cm
            for hBase in range(15, 51): # De 15 a 50 cm
                hSub = sumaTotal - hAsf - hBase
                if 15 <= hSub <= 80:
                    ee_dis = calcular_ee_soportado(hAsf, hBase, hSub, a1, a2, a3, m2, m3, zr_val, so_val, dpsi_val, mr_val_mpa)
                    if ee_dis >= ee_req:
                        return float(hAsf), float(hBase), float(hSub), ee_dis
    return None, None, None, None

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
# GRABAMOS EN MEMORIA SI SE PRESIONA EL BOTÓN
if st.sidebar.button("Generar Informe Técnico 🚀"):
    st.session_state.informe_generado = True

# --- ESTILOS CSS ---
st.markdown("""
<style>
    .info-card { background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6; height: 100%; text-align: left; }
    .info-label { font-size: 12px; color: #6c757d; font-weight: 700; text-transform: uppercase; margin-bottom: 5px; }
    .info-value { font-size: 15px; color: #212529; font-weight: 600; line-height: 1.4; }
    .rate-box { background-color: #e8f4f8; padding: 10px; border-radius: 5px; border-left: 5px solid #17a2b8; margin-bottom: 15px; color: #0c5460; font-weight: 500; }
    .subtitle-sector { color: #555; font-size: 20px; margin-top: -20px; margin-bottom: 20px; font-weight: 500; }
    .verdict-ok { background-color: #d4edda; color: #155724; padding: 15px; border-radius: 5px; font-weight: bold; border-left: 8px solid #28a745; font-size: 16px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    .verdict-bad { background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px; font-weight: bold; border-left: 8px solid #dc3545; font-size: 16px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    .sn-box { background-color: #e2e3e5; padding: 10px; border-radius: 5px; text-align: center; border: 1px solid #ced4da; }
</style>
""", unsafe_allow_html=True)

if not st.session_state.informe_generado:
    st.markdown("<br><br><h1 style='text-align: center;'>🚧 Sistema de Gestión de Pavimentos</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #1f77b4;'>Desarrollado por José Tapia</h3>", unsafe_allow_html=True)
    st.info("👈 Seleccione un camino en el menú lateral y haga clic en 'Generar Informe Técnico'.")
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

    # PESTAÑAS
    tab_demanda, tab_diseno = st.tabs(["📈 Análisis de Demanda", "🛣️ Diseño Estructural (AASHTO 93)"])

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

    # =========================================================================
    # PESTAÑA 2: DISEÑO ESTRUCTURAL INTERACTIVO Y GRÁFICO
    # =========================================================================
    with tab_diseno:
        if not es_granular:
            st.warning(f"⚠️ Este camino ya cuenta con una superficie de **{carpeta}**. El diseño estructural inicial está deshabilitado.")
        elif info_inv is None:
            st.warning("⚠️ No existen datos de Ejes Equivalentes (EEq) para este Rol en el inventario.")
        else:
            st.header("📏 MÉTODO AASHTO 93 - DISEÑO ESTRUCTURAL")
            st.markdown("---")

            col_izq, col_der = st.columns([1, 1])

            # --- PARÁMETROS BASE ---
            with col_izq:
                eeq_val = info_inv['EEq 2045']
                st.metric("Tráfico Proyectado (EES)", f"{eeq_val:,.0f} EEq")
                
                cbr_subrasante = st.number_input("C.B.R. de la Subrasante (%)", min_value=1.0, max_value=100.0, value=4.0, step=0.1)
                
                # CÁLCULO DE MR CORREGIDO A MEGA-PASCALES (MPa) PARA LA FÓRMULA -16.4
                mr_calc_mpa = 17.61 * (cbr_subrasante ** 0.64)
                mr_val_mpa = st.number_input("Módulo Resiliente (MR) en MPa", value=float(round(mr_calc_mpa, 2)))

            with col_der:
                st.markdown("**Parámetros de Diseño**")
                confiabilidad_pct = st.number_input("Confiabilidad (R) %", min_value=50.0, value=95.0, step=1.0)
                dict_zr = {50: 0.0, 75: -0.674, 80: -0.841, 85: -1.036, 90: -1.282, 95: -1.645, 99: -2.327}
                zr_val = dict_zr[min(dict_zr.keys(), key=lambda k: abs(k - confiabilidad_pct))]
                so_val = st.number_input("Desviación Estándar (So)", value=0.45, step=0.01)
                dpsi_val = st.number_input("Pérdida de Serviciabilidad (ΔPSI)", value=2.2, step=0.1)
                
                # Mostrar el NE Requerido exacto calculado a la inversa
                ne_req_mm = resolver_sn_aashto_metric(eeq_val, zr_val, so_val, dpsi_val, mr_val_mpa)
                st.markdown(f"<div class='sn-box'><h4>NE Requerido (mm): <b>{ne_req_mm:.2f}</b></h4></div>", unsafe_allow_html=True)

            st.markdown("---")

            # --- MATERIALES Y BOTÓN MACRO ---
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
                st.info("💡 Ejecuta la iteración para buscar la combinación óptima de espesores.")
                # BOTÓN QUE ACTIVA LA MACRO TRADUCIDA
                if st.button("🔄 Ejecutar Optimización (Macro)"):
                    opt_d1, opt_d2, opt_d3, opt_ee = optimizar_espesores_vba(
                        eeq_val, a1, a2, a3, m2, m3, zr_val, so_val, dpsi_val, mr_val_mpa
                    )
                    if opt_d1 is not None:
                        st.session_state.d1_val = opt_d1
                        st.session_state.d2_val = opt_d2
                        st.session_state.d3_val = opt_d3
                        st.success("✅ ¡Diseño óptimo encontrado!")
                        st.rerun() # Fuerza a la página a redibujarse con los nuevos datos
                    else:
                        st.error("❌ No se encontró solución factible en los rangos (D1: 5-6, D2: 15-50).")

            st.markdown("---")

            # --- ESPESORES Y VISUALIZACIÓN GRÁFICA (TU IMAGEN DE TIERRA DIVIDIDA) ---
            st.subheader("🏗️ Propuesta Estructural Interactiva")
            col_esp, col_graf = st.columns([1, 1.5])

            with col_esp:
                # Extraemos de la sesión para que la Macro los pueda actualizar
                d1 = st.number_input("D1 (Carpeta) cm", value=st.session_state.d1_val, step=0.5, key="inp_d1")
                d2 = st.number_input("D2 (Base) cm", value=st.session_state.d2_val, step=0.5, key="inp_d2")
                d3 = st.number_input("D3 (Subbase) cm", value=st.session_state.d3_val, step=0.5, key="inp_d3")
                
                # Actualizamos la memoria
                st.session_state.d1_val = d1
                st.session_state.d2_val = d2
                st.session_state.d3_val = d3

                ee_soportado = calcular_ee_soportado(d1, d2, d3, a1, a2, a3, m2, m3, zr_val, so_val, dpsi_val, mr_val_mpa)
                holgura = ee_soportado - eeq_val if ee_soportado > eeq_val else 0

                st.markdown("<br>", unsafe_allow_html=True)
                if ee_soportado >= eeq_val:
                    st.markdown(f"<div class='verdict-ok'>✅ APROBADO<br><span style='font-size:14px; font-weight:normal;'>Soporta: {ee_soportado:,.0f} EEq<br>Holgura: +{holgura:,.0f} EEq</span></div>", unsafe_allow_html=True)
                else:
                    deficit = eeq_val - ee_soportado
                    st.markdown(f"<div class='verdict-bad'>⚠️ INSUFICIENTE<br><span style='font-size:14px; font-weight:normal;'>Soporta solo: {ee_soportado:,.0f} EEq<br>Faltan: {deficit:,.0f} EEq</span></div>", unsafe_allow_html=True)

            with col_graf:
                # ESTA ES LA VISUALIZACIÓN DINÁMICA DE LA ESTRATIGRAFÍA
                # El "height" se escala según el valor ingresado para que sea visual.
                h_d1 = max(40, d1 * 4.5) 
                h_d2 = max(50, d2 * 3.5)
                h_d3 = max(50, d3 * 3.0)
                
                # TU FÓRMULA EXACTA DEL EXCEL AÑADIDA AL CÁLCULO
                sn_total_mm = (a1 * d1 * 10) + (a2 * d2 * 10 * m2) + (a3 * d3 * 10 * m3)
                try:
                    beta_calculado = 0.4 + (97.81 / (sn_total_mm + 25.4)) ** 5.19
                except:
                    beta_calculado = 1.0
                
                html_capas = f"""
                <div style="width: 100%; max-width: 400px; margin: auto; border: 3px solid #2c3e50; border-radius: 6px; overflow: hidden; text-align: center; font-family: 'Segoe UI', sans-serif; box-shadow: 0 6px 12px rgba(0,0,0,0.15);">
                    <div style="background: linear-gradient(180deg, #595959 0%, #3b3b3b 100%); color: white; height: {h_d1}px; display: flex; align-items: center; justify-content: center; font-weight: bold; letter-spacing: 0.5px;">
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
                        <div style="font-size: 11px; font-weight: normal; margin-top: 3px; opacity: 0.8;">Beta (β): {beta_calculado:.3f} | SN: {sn_total_mm:.1f} mm</div>
                    </div>
                </div>
                """
                st.markdown(html_capas, unsafe_allow_html=True)
                st.caption("Gráfico Estratigráfico (Se ajusta en tiempo real según espesores)")
