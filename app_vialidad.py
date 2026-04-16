import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

# Silenciamos advertencias matemáticas
warnings.filterwarnings("ignore")

# --- 1. CONFIGURACIÓN Y MEMORIA ---
st.set_page_config(
    page_title="Gestión Vial - Tesis José Tapia",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "informe_generado" not in st.session_state:
    st.session_state.informe_generado = False
if "inp_d1" not in st.session_state: st.session_state.inp_d1 = 5.0
if "inp_d2" not in st.session_state: st.session_state.inp_d2 = 10.0
if "inp_d3" not in st.session_state: st.session_state.inp_d3 = 15.0

# --- 2. FUNCIONES Y DATOS ---
@st.cache_data
def cargar_datos():
    archivo_maestro = "DATA_MAESTRA_TESIS.xlsx"
    archivo_inv = "Inventario_Rutas_Maule_Completo.xlsx"
    try:
        df = pd.read_excel(archivo_maestro)
        df_inv = pd.read_excel(archivo_inv)
        
        # LIMPIEZA DE DATOS
        cols_limpiar = ['ROL', 'ROL NUEVO', 'NOMBRE DEL CAMINO', 'Sector', 'TIPO DE CARPETA', 'CLASIFICACIÓN', 'ESTACIÓN', 'CALZADA']
        for col in cols_limpiar:
            if col in df.columns: df[col] = df[col].astype(str).str.strip()
        if 'Rol' in df_inv.columns: df_inv['Rol'] = df_inv['Rol'].astype(str).str.strip()
        
        # CORRECCIÓN 115 CANALES
        errores_115 = ['115 Canales', '115 CANALES', '115-Canales', '115 CH', '115-CH']
        if 'ROL' in df.columns: df['ROL'] = df['ROL'].replace(errores_115, 'Ruta 115 CH')
        if 'ROL NUEVO' in df.columns: df['ROL NUEVO'] = df['ROL NUEVO'].replace(errores_115, 'Ruta 115 CH')

        return df, df_inv
    except FileNotFoundError as e:
        st.error(f"❌ Error Crítico: No encuentro un archivo. Verifica que {e.filename} esté en la carpeta.")
        st.stop()

# --- MOTOR DE ANÁLISIS REGIONAL (CUADRO DE MANDO) ---
@st.cache_data
def analizar_red_vial_completa(df_m, df_i):
    resultados = []
    anios_censo = [2015, 2017, 2018, 2020, 2022, 2024]
    roles_unicos = df_m['ROL NUEVO'].dropna().unique()
    
    for rol in roles_unicos:
        estaciones_rol = df_m[df_m['ROL NUEVO'] == rol]
        if estaciones_rol.empty: continue
        
        anio_critico = 9999
        tipo_inv = ""
        
        info_inv = df_i[df_i['Rol'] == rol]
        provincia = info_inv.iloc[0]['Provincia'] if (not info_inv.empty and 'Provincia' in info_inv.columns) else "Sin Info"
        
        carpeta_actual = str(estaciones_rol.iloc[0]['TIPO DE CARPETA']).upper()
        es_granular = any(x in carpeta_actual for x in ["RIPIO", "GRANULAR", "TIERRA", "SUELO", "NATURAL"])
        umbral = 300 if es_granular else 5000
        
        for _, fila in estaciones_rol.iterrows():
            try:
                vals = fila[[f'TMDA {a}' for a in anios_censo]].values.flatten().astype(float)
                serie = pd.Series(vals, index=anios_censo).sort_index()
                
                try: modelo = ExponentialSmoothing(serie, trend='mul', damped_trend=True).fit(damping_trend=0.92)
                except: modelo = ExponentialSmoothing(serie, trend='add', damped_trend=True).fit(damping_trend=0.92)
                    
                anios_fut = np.arange(2025, 2046)
                pred_raw = pd.Series(modelo.forecast(len(anios_fut)).values, index=anios_fut)
                
                tasa_crecimiento_inicial = pred_raw.iloc[1] / pred_raw.iloc[0] if pred_raw.iloc[0] > 0 and pred_raw.iloc[1] > 0 else 1.0
                base_teorica_modelo = pred_raw.iloc[0] / tasa_crecimiento_inicial
                ultimo_real = serie.iloc[-1]
                factor_ajuste = ultimo_real / base_teorica_modelo if base_teorica_modelo > 0 else 1.0
                pred_escalada = pred_raw * factor_ajuste
                
                piso = ultimo_real 
                for y in anios_fut:
                    val = pred_escalada[y]
                    if val < piso: val = piso
                    else: piso = val
                    
                    if val >= umbral:
                        if y < anio_critico:
                            anio_critico = y
                            tipo_inv = "Pavimentación" if es_granular else "Segunda Calzada"
                        break
            except: continue
                
        if anio_critico <= 2045:
            resultados.append({"Rol": rol, "Anio": int(anio_critico), "Tipo": tipo_inv, "Provincia": str(provincia).upper()})
            
    return pd.DataFrame(resultados)
# -----------------------------------------------------------

# Funciones Matemáticas de Diseño Estructural
def calcular_so_polinomico(eeq, cv_cbr):
    A5, A6, A7 = 500000, 1500000, 5000000
    cv = float(cv_cbr)
    if eeq < A5: so_calc = -0.0000007*(cv**3) + 0.00007*(cv**2) - 0.0004*cv + 0.4452
    elif eeq < A6: so_calc = -0.0000007*(cv**3) + 0.00007*(cv**2) - 0.0004*cv + 0.4352
    elif eeq < A7: so_calc = -0.000002*(cv**3) + 0.0002*(cv**2) - 0.0051*cv + 0.4553
    else: so_calc = -0.000002*(cv**3) + 0.0002*(cv**2) - 0.0051*cv + 0.4353
    return min(0.5, so_calc)

def calcular_ee_soportado(d1, d2, d3, a1, a2, a3, m2, m3, zr_val, so_val, pi_val, pf_val, mr_val_mpa):
    ne_mm = (d1 * 10 * 1 * a1) + (d2 * 10 * m2 * a2) + (d3 * 10 * m3 * a3)
    try:
        f6_beta = 0.4 + (97.81 / (ne_mm + 25.4)) ** 5.19
        ee = ((ne_mm + 25.4) ** 9.36) * (10 ** (-16.4 + (zr_val * so_val))) * (mr_val_mpa ** 2.32) * (((pi_val - pf_val) / (pi_val - 1.5)) ** (1 / f6_beta))
        return ee
    except: return 0

def optimizar_espesores_vba(ee_req, a1, a2, a3, m2, m3, zr_val, so_val, pi_val, pf_val, mr_val_mpa, is_cape_seal):
    hAsf = 0.0 if is_cape_seal else 5.0
    for sumaGranular in range(25, 131):
        for hBase in range(10, 51):
            hSub = sumaGranular - hBase
            if 15 <= hSub <= 80:
                ee_dis = calcular_ee_soportado(hAsf, hBase, hSub, a1, a2, a3, m2, m3, zr_val, so_val, pi_val, pf_val, mr_val_mpa)
                if ee_dis >= ee_req: return float(hAsf), float(hBase), float(hSub), ee_dis
    return None, None, None, None

df, df_inv = cargar_datos()

# --- 3. ESTILOS CSS ---
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
    .rate-box {
        background-color: #e8f4f8;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #17a2b8;
        margin-bottom: 15px;
        color: #0c5460;
        font-weight: 500;
    }
    .subtitle-sector {
        color: #555;
        font-size: 20px;
        margin-top: -20px;
        margin-bottom: 20px;
        font-weight: 500;
    }
    .ref-table {
        font-size: 12px;
        width: 100%;
        border-collapse: collapse;
    }
    .ref-table th {
        background-color: #f1f3f5;
        border-bottom: 2px solid #dee2e6;
        padding: 8px;
        text-align: left;
        color: #495057;
    }
    .ref-table td {
        border-bottom: 1px solid #dee2e6;
        padding: 8px;
        color: #212529;
    }
    .verdict-ok {background-color:#d4edda; color:#155724; padding:15px; border-radius:5px; font-weight:bold; border-left:8px solid #28a745; font-size:16px; text-align:center;}
    .verdict-bad {background-color:#f8d7da; color:#721c24; padding:15px; border-radius:5px; font-weight:bold; border-left:8px solid #dc3545; font-size:16px; text-align:center;}
    .sn-box {background-color:#e2e3e5; padding:10px; border-radius:5px; text-align:center; border:1px solid #ced4da; margin-top:10px;}
    .money-box {background-color:#fff3cd; color:#856404; padding:15px; border-radius:5px; font-weight:bold; border-left:8px solid #ffc107; font-size:18px; text-align:center; margin-top:10px;}
</style>
""", unsafe_allow_html=True)

# --- 4. MENÚ LATERAL (MODIFICADO CON RESET AUTOMÁTICO) ---
st.sidebar.header("🔍 Panel de Control")

# Esta función apaga el informe cuando cambias de camino
def reset_vista():
    st.session_state.informe_generado = False

roles = sorted(df['ROL NUEVO'].dropna().astype(str).unique())
# Le decimos a las casillas que ejecuten la función reset_vista al cambiar
rol_sel = st.sidebar.selectbox("Seleccione Rol Oficial:", roles, on_change=reset_vista)
df_rol = df[df['ROL NUEVO'] == rol_sel].copy()
df_rol['ETIQUETA'] = df_rol['NOMBRE DEL CAMINO'] + " (" + df_rol['ESTACIÓN'] + ")"
tramo_sel = st.sidebar.selectbox("Seleccione Sector:", df_rol['ETIQUETA'].tolist(), on_change=reset_vista)

st.sidebar.markdown("---")
btn_calc = st.sidebar.button("Generar Informe Técnico 🚀")
if btn_calc:
    st.session_state.informe_generado = True
if st.sidebar.button("🏠 Volver a Visión Regional"):
    st.session_state.informe_generado = False

# --- 5. INTERFAZ PRINCIPAL ---
if not st.session_state.informe_generado:
    # --- PORTADA Y DASHBOARD REGIONAL ---
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
    
    with st.spinner("Analizando la red vial completa de la región..."):
        df_alertas = analizar_red_vial_completa(df, df_inv)

    st.markdown("<h3 style='text-align: center;'>Planificación de Inversiones Prioritarias (2025-2045)</h3>", unsafe_allow_html=True)
    
    tabs_reg = st.tabs(["🌎 Región Global", "📍 Provincia de Talca", "📍 Provincia de Curicó", "📍 Provincia de Linares", "📍 Provincia de Cauquenes"])

    def plot_dashboard(data, titulo):
        if data.empty:
            st.info(f"No hay proyectos críticos detectados para {titulo} en el periodo de diseño.")
            return
            
        pivote = data.groupby(['Anio', 'Tipo']).size().unstack(fill_value=0)
        anios_full = np.arange(2025, 2046)
        pivote = pivote.reindex(anios_full, fill_value=0)
        
        fig, ax = plt.subplots(figsize=(10, 4.5))
        pivote.plot(kind='bar', stacked=False, ax=ax, color=['#2ca02c', '#ff7f0e'], width=0.8)
        
        ax.set_title(f"Inversiones Requeridas por Saturación: {titulo}")
        ax.set_ylabel("Cantidad de Proyectos")
        ax.set_xlabel("Año Crítico")
        
        ax.set_xticks(range(len(anios_full)))
        ax.set_xticklabels(anios_full, rotation=45, ha='right')
        
        max_proyectos = pivote.values.max() if not pivote.empty else 0
        ax.set_ylim(0, max_proyectos + 5)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper left')
        
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
        
        with st.expander(f"📄 Ver detalle de Rutas y Años Críticos en {titulo}"):
            st.dataframe(data.sort_values('Anio').reset_index(drop=True))

    with tabs_reg[0]: plot_dashboard(df_alertas, "Región del Maule")
    with tabs_reg[1]: plot_dashboard(df_alertas[df_alertas['Provincia'].str.contains('TALCA', case=False, na=False)], "Provincia de Talca")
    with tabs_reg[2]: plot_dashboard(df_alertas[df_alertas['Provincia'].str.contains('CURICO|CURICÓ', case=False, na=False)], "Provincia de Curicó")
    with tabs_reg[3]: plot_dashboard(df_alertas[df_alertas['Provincia'].str.contains('LINARES', case=False, na=False)], "Provincia de Linares")
    with tabs_reg[4]: plot_dashboard(df_alertas[df_alertas['Provincia'].str.contains('CAUQUENES', case=False, na=False)], "Provincia de Cauquenes")

    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown("""
        <h3 style='text-align: center; color: #1f77b4;'>
            Desarrollado por José Tapia
        </h3>
        <p style='text-align: center; font-size: 18px;'>
            Memoria para optar al título de Ingeniero Civil
        </p>
    """, unsafe_allow_html=True)
    st.info("👈 Seleccione un camino en el menú lateral para iniciar el análisis estructural detallado.")

else:
    # EXTRACCIÓN DE DATOS DEL TRAMO
    fila = df_rol[df_rol['ETIQUETA'] == tramo_sel].iloc[0]
    nombre = fila['NOMBRE DEL CAMINO']
    sector_especifico = fila['Sector'] if 'Sector' in fila else "Sector No Especificado"
    rol_oficial = fila['ROL NUEVO']
    carpeta = fila['TIPO DE CARPETA']
    clasificacion = fila['CLASIFICACIÓN']
    calzada_info = fila['CALZADA'] if 'CALZADA' in fila else "No Inf"
    
    rodadura_maestra = str(carpeta).upper()
    es_granular = any(x in rodadura_maestra for x in ["RIPIO", "GRANULAR", "TIERRA", "SUELO", "NATURAL"])
    es_pavimentado = not es_granular
    info_inv = df_inv[df_inv['Rol'] == rol_sel].iloc[0] if not df_inv[df_inv['Rol'] == rol_sel].empty else None

    # EXTRAER KM INICIAL Y FINAL
    km_ini, km_fin = "No Inf", "No Inf"
    val_km_ini, val_km_fin = 0.0, 0.0
    if info_inv is not None:
        for col in info_inv.index:
            col_up = str(col).upper()
            if col_up in ['KM INICIAL', 'KM_INI', 'KILOMETRO INICIAL', 'KILÓMETRO INICIAL', 'KM INICIO']:
                km_ini = info_inv[col]
                try: val_km_ini = float(info_inv[col])
                except: pass
            if col_up in ['KM FINAL', 'KM_FIN', 'KILOMETRO FINAL', 'KILÓMETRO FINAL', 'KM FIN']:
                km_fin = info_inv[col]
                try: val_km_fin = float(info_inv[col])
                except: pass
                
    largo_km = abs(val_km_fin - val_km_ini) if km_ini != "No Inf" and km_fin != "No Inf" else 1.0

    # TÍTULO PRINCIPAL
    st.markdown("### 🚧 Sistema de Gestión de Pavimentos y Proyección de Demanda")
    st.title(f"📍 {nombre}")
    st.markdown(f"<div class='subtitle-sector'>Sector: {sector_especifico}</div>", unsafe_allow_html=True)
    
    # TARJETAS DE INFORMACIÓN GENERAL
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: st.markdown(f"<div class='info-card'><div class='info-label'>Rol Oficial</div><div class='info-value'>{rol_oficial}</div></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='info-card'><div class='info-label'>Tipo de Carpeta</div><div class='info-value'>{carpeta}</div></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='info-card'><div class='info-label'>Clasificación</div><div class='info-value'>{clasificacion}</div></div>", unsafe_allow_html=True)
    with c4: st.markdown(f"<div class='info-card'><div class='info-label'>Calzada</div><div class='info-value'>{calzada_info}</div></div>", unsafe_allow_html=True)
    with c5: st.markdown(f"<div class='info-card'><div class='info-label'>Km Inicial</div><div class='info-value'>{km_ini}</div></div>", unsafe_allow_html=True)
    with c6: st.markdown(f"<div class='info-card'><div class='info-label'>Km Final</div><div class='info-value'>{km_fin}</div></div>", unsafe_allow_html=True)
    st.markdown("---")

    # CREACIÓN DE PESTAÑAS
    tab_demanda, tab_diseno, tab_presupuesto = st.tabs(["📈 Análisis de Demanda y Proyección", "🛣️ Diseño Estructural (AASHTO 93)", "💰 Presupuesto Obra Gruesa"])

    # ==========================================
    # PESTAÑA 1: ANÁLISIS DE DEMANDA
    # ==========================================
    with tab_demanda:
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
        
        try:
            try: modelo = ExponentialSmoothing(serie, trend='mul', seasonal=None, damped_trend=True).fit(damping_trend=0.92)
            except: modelo = ExponentialSmoothing(serie, trend='add', seasonal=None, damped_trend=True).fit(damping_trend=0.92)
                
            anios_fut = np.arange(2025, 2046)
            pred_raw = pd.Series(modelo.forecast(len(anios_fut)).values, index=anios_fut)
            
            tasa_crecimiento_inicial = pred_raw.iloc[1] / pred_raw.iloc[0] if pred_raw.iloc[0] > 0 and pred_raw.iloc[1] > 0 else 1.0
            base_teorica_modelo = pred_raw.iloc[0] / tasa_crecimiento_inicial
            ultimo_real = serie.iloc[-1]
            factor_ajuste = ultimo_real / base_teorica_modelo if base_teorica_modelo > 0 else 1.0
            pred_escalada = pred_raw * factor_ajuste
            
            pred_ajustada = []
            piso = ultimo_real 
            for y in anios_fut:
                val = pred_escalada[y]
                if val < piso: val = piso
                else: piso = val
                pred_ajustada.append(val)
            pred = pd.Series(pred_ajustada, index=anios_fut)
        except Exception as e:
            st.error(f"Error en el cálculo: {e}")
            st.stop()
        
        tmda_24, tmda_26, tmda_45 = serie[2024], pred[2026], pred[2045]
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
        
        anio_saturacion, val_saturacion = None, None
        full_vals = pd.concat([serie, pred])
        solo_futuro = full_vals[full_vals.index >= 2024]
        for y in solo_futuro.index:
            if solo_futuro[y] >= 5000:
                anio_saturacion, val_saturacion = y, solo_futuro[y]
                break
        
        if anio_saturacion is not None:
            ax.scatter([anio_saturacion], [val_saturacion], color='red', s=150, zorder=15, edgecolors='white')
            texto_sat = f"¡SATURADO HOY!\n(Año 2024)" if anio_saturacion == 2024 else f"¡SATURACIÓN!\nAño {int(anio_saturacion)}"
            offset_y = 600 if val_saturacion < 10000 else -1500
            ax.annotate(texto_sat, xy=(anio_saturacion, val_saturacion), xytext=(anio_saturacion, val_saturacion + offset_y),
                        arrowprops=dict(facecolor='red', shrink=0.05), color='red', fontweight='bold', ha='center')

        ax.set_ylabel("Flujo Vehicular (veh/día)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        with st.expander("📅 Ver Histórico de Tránsito y Tasas Reales (2015-2024)", expanded=False):
            df_hist = pd.DataFrame({'Año': anios_censo, 'TMDA Real': vals_censo.astype(int)})
            crecimiento = [0.0]
            for i in range(1, len(df_hist)):
                v_actual, v_ant = df_hist.iloc[i]['TMDA Real'], df_hist.iloc[i-1]['TMDA Real']
                n_anios = df_hist.iloc[i]['Año'] - df_hist.iloc[i-1]['Año']
                tasa = ((v_actual / v_ant) ** (1/n_anios) - 1) * 100 if v_ant > 0 and n_anios > 0 else 0.0
                crecimiento.append(tasa)
            df_hist['Crecimiento Anual (%)'] = [f"{x:.2f}%" if i > 0 else "-" for i, x in enumerate(crecimiento)]
            st.table(df_hist.set_index('Año'))

        with st.expander("📄 Ver Tabla de Proyección Futura (2025-2045)", expanded=False):
            df_tabla = pd.DataFrame({'TMDA Proyectado': pred.values.astype(int)}, index=pred.index)
            serie_completa_calc = pd.concat([pd.Series([tmda_24], index=[2024]), pred])
            df_tabla['Crecimiento Anual (%)'] = (serie_completa_calc.pct_change() * 100).loc[2025:].apply(lambda x: f"{x:.2f}%")
            st.table(df_tabla)

        st.subheader("📋 Diagnóstico Técnico y Criterios de Diseño")
        col_diag, col_crit = st.columns([1.3, 1])

        with col_diag:
            st.markdown("#### 📢 Estado del Proyecto")
            carpeta_up, calzada_up = carpeta.upper(), calzada_info.upper()
            es_doble_via = "DOBLE" in calzada_up or "DOBLE" in carpeta_up

            if es_granular:
                if tmda_24 > 300: st.error(f"🔴 **PRIORIDAD ALTA:** Camino granular con {int(tmda_24)} veh/día. Supera norma (300). **Se recomienda Pavimentación.**")
                else: st.success(f"🟢 **CONSERVACIÓN:** Tránsito bajo ({int(tmda_24)} veh/día). Mantener perfilado.")
            elif es_pavimentado:
                if not es_doble_via:
                    if tmda_24 > 5000: st.error(f"🔴 **SATURACIÓN VIGENTE (2024):** Vía simple con {int(tmda_24)} veh/día. Supera capacidad. **Se sugiere Estudio de Segunda Calzada.**")
                    elif anio_saturacion and anio_saturacion > 2024: st.warning(f"🟡 **ALERTA FUTURA:** Se proyecta saturación para el año {anio_saturacion}. **Planificar ampliación.**")
                    else: st.success("🟢 **OPERACIÓN NORMAL:** Capacidad suficiente durante todo el periodo de proyección.")
                else: st.success("🟢 **ESTÁNDAR ADECUADO:** Doble Calzada acorde al flujo.")

        with col_crit:
            st.markdown("#### 📏 Referencia Manual de Carreteras (Vol. 3)")
            st.markdown("""
            <table class="ref-table">
                <thead><tr><th>TMDA (veh/día)</th><th>Categoría</th><th>Intervención Sugerida</th></tr></thead>
                <tbody>
                    <tr><td><b>&lt; 300</b></td><td>Tránsito Bajo</td><td>Mantener Carpeta Granular</td></tr>
                    <tr><td><b>300 – 5.000</b></td><td>Tránsito Medio</td><td>Pavimentación (Sello/Asfalto)</td></tr>
                    <tr><td><b>&gt; 5.000</b></td><td>Saturación</td><td>Estudio de Segunda Calzada</td></tr>
                </tbody>
            </table>
            """, unsafe_allow_html=True)


    # ==========================================
    # PESTAÑA 2: DISEÑO ESTRUCTURAL
    # ==========================================
    with tab_diseno:
        if info_inv is None:
            st.warning("⚠️ No existen datos de Ejes Equivalentes (EEq) para este Rol en el inventario.")
        else:
            if es_pavimentado:
                st.info("ℹ️ Este camino ya cuenta con pavimento. El cálculo a continuación corresponde al estudio de **Segunda Calzada**.")

            st.header("📏 MÉTODO AASHTO 93 - DISEÑO ESTRUCTURAL")
            st.markdown("---")
            
            col_in1, col_in2 = st.columns(2)
            with col_in1:
                eeq_base = info_inv['EEq 2045']
                if es_pavimentado:
                    st.markdown("**⚙️ Factores de Distribución (Doble Calzada)**")
                    st.caption("Nota: Se asume que el EEq del inventario ya posee factor direccional (0.5).")
                    fp = st.slider("Factor de Pista (fp)", min_value=0.50, max_value=1.00, value=0.80, step=0.05)
                    eeq_val = eeq_base * fp
                    st.metric("Tráfico de Diseño (EES)", f"{eeq_val:,.0f} EEq", f"Base: {eeq_base:,.0f} x {fp:.2f}")
                else:
                    eeq_val = eeq_base
                    st.metric("Tráfico Proyectado (EES)", f"{eeq_val:,.0f} EEq")
                    
            with col_in2:
                cbr_subrasante = st.number_input("C.B.R. de la Subrasante (%)", min_value=1.0, max_value=100.0, value=25.0, step=0.1)
                mr_calc_mpa = 17.6 * (cbr_subrasante ** 0.64) if cbr_subrasante < 12 else 22.1 * (cbr_subrasante ** 0.55)
                st.info(f"Módulo Resiliente (MR) auto-calculado: **{mr_calc_mpa:.2f} MPa**")
                mr_val_mpa = float(round(mr_calc_mpa, 2))

            st.markdown("---")

            st.subheader("📊 Parámetros de Diseño (Factores AASHTO)")
            c_so1, c_so2, c_so3 = st.columns([1, 4, 1])
            with c_so2:
                st.markdown("<div style='text-align:center; color:#555; font-size:14px; margin-bottom:5px;'><b>Tabla de Referencia:</b> Valores de Confiabilidad y Desviación Normal (Zr)</div>", unsafe_allow_html=True)
                try: st.image("So.jpg", use_container_width=True)
                except: st.caption("*(Imagen So.jpg no encontrada)*")

            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1: zr_val = st.number_input("Confiabilidad (Zr)", value=-0.253, step=0.010, format="%.3f")
            with col_p2:
                cv_cbr = st.number_input("Coef. Variación CBR (CV %)", value=50.0, step=5.0)
                so_val = calcular_so_polinomico(eeq_val, cv_cbr)
                st.info(f"Desviación ($S_o$): **{so_val:.4f}**")
            with col_p3:
                col_pi, col_pf = st.columns(2)
                with col_pi: pi_val = st.number_input("Serv. Inicial (pi)", value=4.2, step=0.1)
                with col_pf: pf_val = st.number_input("Serv. Final (pf)", value=2.0, step=0.1)

            st.markdown("---")

            st.subheader("⚙️ Materiales y Cálculo Automático")
            c_dr1, c_dr2, c_dr3 = st.columns([1, 4, 1])
            with c_dr2:
                st.markdown("<div style='text-align:center; color:#555; font-size:14px; margin-bottom:5px;'><b>Tabla de Referencia:</b> Coeficientes de Drenaje (m)</div>", unsafe_allow_html=True)
                try: st.image("drenaje.jpg", use_container_width=True)
                except: st.caption("*(Imagen drenaje.jpg no encontrada)*")

            col_mat1, col_mat2, col_opt = st.columns([1, 1, 1.2])
            with col_mat1:
                is_cape_seal = st.checkbox("🛣️ Camino Básico (Cape Seal / TSD)", value=not es_pavimentado)
                val_a1 = 0.000 if is_cape_seal else 0.197
                a1 = st.number_input("Coef. Asfalto (1/mm)", value=val_a1, disabled=True, format="%.3f")
                a2 = st.number_input("Coef. Base (1/mm)", value=0.090, format="%.3f")
                a3 = st.number_input("Coef. Subbase (1/mm)", value=0.090, format="%.3f")
            
            with col_mat2:
                precip = info_inv['Precipitacion promedio Mensual (mm)']
                m_sugerido = 0.8 if precip > 80 else (1.0 if precip > 40 else 1.1)
                m2 = st.number_input(f"Coef. Drenaje Base", value=m_sugerido, format="%.2f")
                m3 = st.number_input(f"Coef. Drenaje Subbase", value=m_sugerido, format="%.2f")

            with col_opt:
                st.markdown("<br>", unsafe_allow_html=True)
                st.info("💡 Ejecuta la macro para buscar la combinación óptima.")
                if st.button("🔄 Ejecutar Optimización (Macro)"):
                    opt_d1, opt_d2, opt_d3, opt_ee = optimizar_espesores_vba(eeq_val, a1, a2, a3, m2, m3, zr_val, so_val, pi_val, pf_val, mr_val_mpa, is_cape_seal)
                    if opt_d1 is not None:
                        st.session_state.inp_d1, st.session_state.inp_d2, st.session_state.inp_d3 = float(opt_d1), float(opt_d2), float(opt_d3)
                        st.success("✅ ¡Diseño óptimo encontrado!")
                        st.rerun()
                    else: st.error("❌ No se encontró solución factible en los rangos.")

            st.markdown("---")

            st.subheader("🏗️ Propuesta Estructural Interactiva")
            col_esp, col_graf = st.columns([1, 1.5])
            with col_esp:
                if is_cape_seal: st.session_state.inp_d1 = 0.0
                elif st.session_state.inp_d1 < 5.0: st.session_state.inp_d1 = 5.0

                min_d1 = 0.0 if is_cape_seal else 5.0
                d1 = st.number_input("Carpeta Asfáltica (cm)", min_value=min_d1, step=0.5, key="inp_d1", disabled=is_cape_seal)
                d2 = st.number_input("Base Granular (cm)", min_value=10.0, step=0.5, key="inp_d2")
                d3 = st.number_input("Subbase Granular (cm)", min_value=15.0, step=0.5, key="inp_d3")

                ne_aportado = (d1 * 10 * 1 * a1) + (d2 * 10 * m2 * a2) + (d3 * 10 * m3 * a3)
                st.markdown(f"<div class='sn-box'><h4>NE Aportado: <b>{ne_aportado:.2f} mm</b></h4></div>", unsafe_allow_html=True)
                ee_soportado = calcular_ee_soportado(d1, d2, d3, a1, a2, a3, m2, m3, zr_val, so_val, pi_val, pf_val, mr_val_mpa)
                holgura = ee_soportado - eeq_val if ee_soportado > eeq_val else 0

                st.markdown("<br>", unsafe_allow_html=True)
                if ee_soportado >= eeq_val:
                    st.markdown(f"<div class='verdict-ok'>✅ APROBADO<br><span style='font-size:14px; font-weight:normal;'>Soporta: {ee_soportado:,.0f} EEq<br>Holgura: +{holgura:,.0f} EEq</span></div>", unsafe_allow_html=True)
                else:
                    deficit = eeq_val - ee_soportado
                    st.markdown(f"<div class='verdict-bad'>⚠️ INSUFICIENTE<br><span style='font-size:14px; font-weight:normal;'>Soporta solo: {ee_soportado:,.0f} EEq<br>Faltan: {deficit:,.0f} EEq</span></div>", unsafe_allow_html=True)

            with col_graf:
                html_asfalto = """<div style="background: linear-gradient(180deg, #333 0%, #111 100%); color: #fff; height: 30px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 13px; border-bottom: 2px solid #000;">Cape Seal / TSD (Protección)</div>""" if is_cape_seal else f"""<div style="background: linear-gradient(180deg, #595959 0%, #3b3b3b 100%); color: white; height: {max(40, d1 * 4.5)}px; display: flex; align-items: center; justify-content: center; font-weight: bold;">Carpeta Asfáltica ({d1} cm)</div>"""
                try: beta_calculado = 0.4 + (97.81 / (ne_aportado + 25.4)) ** 5.19
                except: beta_calculado = 1.0
                
                html_capas = f"""
                <div style="width: 100%; max-width: 400px; margin: auto; border: 3px solid #2c3e50; border-radius: 6px; overflow: hidden; text-align: center; font-family: 'Segoe UI', sans-serif; box-shadow: 0 6px 12px rgba(0,0,0,0.15);">
                    {html_asfalto}
                    <div style="background: linear-gradient(180deg, #e3c988 0%, #d4b872 100%); color: #333; height: {max(50, d2 * 3.5)}px; display: flex; align-items: center; justify-content: center; font-weight: bold; border-top: 2px solid #2c3e50;">Base Granular ({d2} cm)</div>
                    <div style="background: linear-gradient(180deg, #b8865b 0%, #a67c52 100%); color: white; height: {max(50, d3 * 3.0)}px; display: flex; align-items: center; justify-content: center; font-weight: bold; border-top: 2px solid #2c3e50;">Subbase Granular ({d3} cm)</div>
                    <div style="background: linear-gradient(180deg, #6e4e37 0%, #4a3322 100%); color: #e0e0e0; height: 90px; display: flex; flex-direction: column; align-items: center; justify-content: center; font-weight: bold; border-top: 4px dashed #1a110b;">
                        <div>Suelo Subrasante (CBR {cbr_subrasante}%)</div>
                        <div style="font-size: 11px; font-weight: normal; margin-top: 3px; opacity: 0.8;">Beta (β): {beta_calculado:.3f}</div>
                    </div>
                </div>
                """
                st.markdown(html_capas, unsafe_allow_html=True)
                st.caption("Gráfico Estratigráfico (Se ajusta en tiempo real)")


    # ==========================================
    # PESTAÑA 3: PRESUPUESTO OBRA GRUESA (MODIFICADO)
    # ==========================================
    with tab_presupuesto:
        if info_inv is None:
            st.warning("⚠️ No existen datos del inventario para calcular el presupuesto de este Rol.")
        else:
            st.header("💰 Cubicaciones y Presupuesto Referencial")
            st.markdown("---")

            st.subheader("📏 Geometría del Proyecto a Intervenir")
            col_g1, col_g2, col_g3 = st.columns(3)
            
            with col_g1:
                max_largo = float(largo_km) if largo_km > 0 else 1.0
                largo_intervenir = st.number_input("Longitud a intervenir (km)", min_value=0.1, value=max_largo, step=0.1)
                largo_m = largo_intervenir * 1000
                st.caption(f"Largo total según inventario: {max_largo:.2f} km")
                
            with col_g2:
                ancho_calzada = st.number_input("Ancho de Calzada / Cape Seal (m)", min_value=3.0, max_value=12.0, value=6.0, step=0.1)
            with col_g3:
                ancho_imprimacion = ancho_calzada + 0.40
                st.metric("Ancho Imprimación (+40 cm)", f"{ancho_imprimacion:.2f} m")

            st.markdown("---")
            st.subheader("🧱 Cubicaciones Volumétricas (Según Diseño Estructural)")
            
            # Extraer espesores desde memoria (pasando a metros)
            h_base_m = st.session_state.inp_d2 / 100.0
            h_sub_m = st.session_state.inp_d3 / 100.0

            # Geometría Asfalto/Imprimación
            sup_asf = ancho_calzada * largo_m
            area_imprimacion = ancho_imprimacion * largo_m

            # Geometría Base Granular
            ancho_base_inf = ancho_imprimacion + (h_base_m * 3.0) 
            ancho_base_medio = ancho_imprimacion + (h_base_m * 1.5)
            vol_base = round(ancho_base_medio * h_base_m * largo_m, 1)

            # Geometría Subbase Granular
            ancho_subbase_sup = ancho_base_inf
            ancho_subbase_inf = ancho_subbase_sup + (h_sub_m * 3.0)
            area_subbase = ancho_subbase_sup * largo_m 
            
            # Lógica Asfalto vs Capseal
            if is_cape_seal:
                nombre_asfalto = "Cape Seal / TSD"
                uni_asf = "m²"
                cant_asf = sup_asf
                pu_asf = 7800
                delta_asf = "Espesor: N/A"
            else:
                nombre_asfalto = "Concreto Asfáltico (MAC)"
                uni_asf = "m³"
                cant_asf = round(sup_asf * (st.session_state.inp_d1 / 100.0), 1)
                pu_asf = 250000
                delta_asf = f"Espesor: {st.session_state.inp_d1} cm"

            c_cub1, c_cub2, c_cub3, c_cub4 = st.columns(4)
            c_cub1.metric(f"Cantidad {nombre_asfalto}", f"{cant_asf:,.1f} {uni_asf}", delta_asf, delta_color="off")
            c_cub2.metric("Área Imprimación", f"{area_imprimacion:,.1f} m²")
            c_cub3.metric("Volumen Base", f"{vol_base:,.1f} m³", f"Espesor: {st.session_state.inp_d2} cm", delta_color="off")
            c_cub4.metric("Área Subbase", f"{area_subbase:,.1f} m²", f"Espesor: {st.session_state.inp_d3} cm", delta_color="off")

            st.markdown("---")
            st.subheader("💵 Valorización")
            
            pu_imprimacion = 1500
            pu_base = 40000
            pu_subbase = 1300

            tot_asfalto = cant_asf * pu_asf
            tot_imprimacion = area_imprimacion * pu_imprimacion
            tot_base = vol_base * pu_base
            tot_subbase = area_subbase * pu_subbase
            tot_proyecto = tot_asfalto + tot_imprimacion + tot_base + tot_subbase

            col_p1, col_p2 = st.columns([2, 1])
            with col_p1:
                df_presupuesto = pd.DataFrame({
                    "Ítem": [nombre_asfalto, "Imprimación Asfáltica", "Base Granular (CBR 100%)", "Subbase Granular"],
                    "Unidad": [uni_asf, "m²", "m³", "m²"],
                    "Cantidad": [cant_asf, area_imprimacion, vol_base, area_subbase],
                    "P.U. ($)": [pu_asf, pu_imprimacion, pu_base, pu_subbase],
                    "Total ($)": [tot_asfalto, tot_imprimacion, tot_base, tot_subbase]
                })
                
                df_visual = df_presupuesto.copy()
                df_visual["Cantidad"] = df_visual["Cantidad"].apply(lambda x: f"{x:,.1f}".replace(',', '.'))
                df_visual["P.U. ($)"] = df_visual["P.U. ($)"].apply(lambda x: f"${x:,.0f}".replace(',', '.'))
                df_visual["Total ($)"] = df_visual["Total ($)"].apply(lambda x: f"${x:,.0f}".replace(',', '.'))
                
                st.table(df_visual.set_index("Ítem"))
            
            with col_p2:
                st.markdown(f"<div class='money-box'>Costo Total Directo:<br><span style='font-size:30px;'>${tot_proyecto:,.0f}</span></div>".replace(',', '.'), unsafe_allow_html=True)
                st.caption("*Costo referencial de obra gruesa en pesos chilenos. No incluye IVA, utilidades, ni obras de arte adicionales.*")

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("📐 Perfil Transversal Esquemático")
            
            fig_t, ax_t = plt.subplots(figsize=(10, 3.5))
            
            x_cs = [-ancho_calzada/2, ancho_calzada/2, ancho_calzada/2, -ancho_calzada/2]
            y_cs = [0, 0, -0.02, -0.02]
            
            x_imp = [-ancho_imprimacion/2, ancho_imprimacion/2, ancho_imprimacion/2, -ancho_imprimacion/2]
            y_imp = [-0.02, -0.02, -0.03, -0.03]
            
            x_base = [-ancho_imprimacion/2, ancho_imprimacion/2, ancho_base_inf/2, -ancho_base_inf/2]
            y_base = [-0.03, -0.03, -h_base_m - 0.03, -h_base_m - 0.03]
            
            x_sub = [-ancho_base_inf/2, ancho_base_inf/2, ancho_subbase_inf/2, -ancho_subbase_inf/2]
            y_sub = [-h_base_m - 0.03, -h_base_m - 0.03, -h_base_m - h_sub_m - 0.03, -h_base_m - h_sub_m - 0.03]
            
            # EL ORDEN INVERTIDO PARA QUE LA LEYENDA SALGA DE ARRIBA HACIA ABAJO
            if is_cape_seal:
                ax_t.fill(x_cs, y_cs, color='#222222', label='Cape Seal')
            else:
                y_asf = [0, 0, -(st.session_state.inp_d1/100.0), -(st.session_state.inp_d1/100.0)]
                ax_t.fill(x_cs, y_asf, color='#444444', label=f'Concreto Asfáltico ({st.session_state.inp_d1} cm)')

            ax_t.fill(x_imp, y_imp, color='#888888') # Sin etiqueta en la leyenda
            ax_t.fill(x_base, y_base, color='#d4b872', label=f'Base Granular ({st.session_state.inp_d2} cm)')
            ax_t.fill(x_sub, y_sub, color='#a67c52', label=f'Subbase ({st.session_state.inp_d3} cm)')
            
            ax_t.text(0, 0.05, f"Ancho Calzada: {ancho_calzada} m", ha='center', fontsize=10, fontweight='bold')
            ax_t.text(0, -h_base_m/2, f"Base", ha='center', fontsize=10)
            ax_t.text(0, -h_base_m - h_sub_m/2, f"Subbase", ha='center', fontsize=10, color='white')
            
            ax_t.set_xlim(-ancho_subbase_inf/2 - 1, ancho_subbase_inf/2 + 1)
            ax_t.set_ylim(-h_base_m - h_sub_m - 0.1, 0.2)
            ax_t.axis('off')
            ax_t.legend(loc='lower left')
            
            st.pyplot(fig_t)

    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; color: #888;'><small>Creado por José Tapia - Tesis Ingeniería Civil</small></div>", unsafe_allow_html=True)
