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
    
    df_maestra = pd.read_excel(archivo_maestra)
    df_inv = pd.read_excel(archivo_inventario)
    
    # LIMPIEZA DE DATOS MAESTRA
    cols_limpiar = ['ROL', 'ROL NUEVO', 'NOMBRE DEL CAMINO', 'Sector', 'TIPO DE CARPETA', 'CLASIFICACIÓN', 'ESTACIÓN', 'CALZADA']
    for col in cols_limpiar:
        if col in df_maestra.columns:
            df_maestra[col] = df_maestra[col].astype(str).str.strip()
            
    # LIMPIEZA DE DATOS INVENTARIO (Evita errores por espacios ocultos)
    if 'Rol' in df_inv.columns:
        df_inv['Rol'] = df_inv['Rol'].astype(str).str.strip()
    
    # CORRECCIÓN 115 CANALES
    errores_115 = ['115 Canales', '115 CANALES', '115-Canales', '115 CH', '115-CH']
    if 'ROL' in df_maestra.columns:
        df_maestra['ROL'] = df_maestra['ROL'].replace(errores_115, 'Ruta 115 CH')
    if 'ROL NUEVO' in df_maestra.columns:
        df_maestra['ROL NUEVO'] = df_maestra['ROL NUEVO'].replace(errores_115, 'Ruta 115 CH')

    return df_maestra, df_inv

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
    .verdict-ok { background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px; font-weight: bold; border-left: 5px solid #28a745; }
    .verdict-bad { background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; font-weight: bold; border-left: 5px solid #dc3545; }
</style>
""", unsafe_allow_html=True)

# --- 4. INTERFAZ Y CÁLCULOS ---
if not btn_calc:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>🚧 Sistema de Gestión de Pavimentos</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #1f77b4;'>Desarrollado por José Tapia</h3>", unsafe_allow_html=True)
    st.info("👈 Seleccione un camino en el menú lateral para iniciar el análisis.")
else:
    # 1. AISLAR LA FILA SELECCIONADA EN LA DATA MAESTRA
    fila = df_rol[df_rol['ETIQUETA'] == tramo_sel].iloc[0]
    nombre = fila['NOMBRE DEL CAMINO']
    rol_oficial = fila['ROL NUEVO']
    carpeta = fila['TIPO DE CARPETA']
    clasificacion = fila['CLASIFICACIÓN']
    calzada_info = fila['CALZADA'] if 'CALZADA' in fila else "No Inf"
    sector_especifico = fila['Sector']
    
    # 2. EVALUAR SI LA CARPETA ES GRANULAR DESDE LA DATA MAESTRA
    rodadura_maestra = str(carpeta).upper()
    es_granular = any(x in rodadura_maestra for x in ["RIPIO", "GRANULAR", "TIERRA", "SUELO", "NATURAL"])
    
    # 3. OBTENER EEq Y PRECIPITACIÓN DESDE EL INVENTARIO
    datos_inv_especifico = df_inv[df_inv['Rol'] == rol_sel].copy()
    info_inv = None
    if not datos_inv_especifico.empty:
        info_inv = datos_inv_especifico.iloc[0]

    # 4. CREACIÓN CONDICIONAL DE PESTAÑAS
    mostrar_diseno = es_granular and (info_inv is not None)
    
    if mostrar_diseno:
        tab_demanda, tab_diseno = st.tabs(["📈 Análisis de Demanda", "🛣️ Diseño Estructural"])
    else:
        tab_demanda = st.tabs(["📈 Análisis de Demanda"])[0]

    # =========================================================================
    # PESTAÑA 1: ANÁLISIS DE DEMANDA (INTACTA)
    # =========================================================================
    with tab_demanda:
        st.markdown("### 🚧 Sistema de Gestión de Pavimentos y Proyección de Demanda")
        
        if not es_granular:
            st.info(f"ℹ️ La pestaña de Diseño Estructural está oculta porque este sector figura con carpeta de tipo: **{carpeta}** en el censo oficial.")
        elif info_inv is None:
            st.warning(f"⚠️ Este camino es de **{carpeta}**, pero no se encuentra en el archivo de Inventario. Faltan los Ejes Equivalentes (EEq) para realizar el diseño estructural.")
                
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
        
        try:
            try:
                modelo = ExponentialSmoothing(serie, trend='mul', seasonal=None, damped_trend=True).fit(damping_trend=0.92)
            except:
                modelo = ExponentialSmoothing(serie, trend='add', seasonal=None, damped_trend=True).fit(damping_trend=0.92)
                
            anios_fut = np.arange(2025, 2046)
            pred_raw = modelo.forecast(len(anios_fut))
            pred_raw = pd.Series(pred_raw.values, index=anios_fut)
            
            if pred_raw.iloc[0] > 0 and pred_raw.iloc[1] > 0:
                tasa_crecimiento_inicial = pred_raw.iloc[1] / pred_raw.iloc[0]
            else:
                tasa_crecimiento_inicial = 1.0
            
            base_teorica_modelo = pred_raw.iloc[0] / tasa_crecimiento_inicial
            ultimo_real = serie.iloc[-1]
            factor_ajuste = ultimo_real / base_teorica_modelo if base_teorica_modelo > 0 else 1.0
            pred_escalada = pred_raw * factor_ajuste
            
            pred_ajustada = []
            piso = ultimo_real 
            for y in anios_fut:
                val = pred_escalada[y]
                if val < piso:
                    val = piso
                else:
                    piso = val
                pred_ajustada.append(val)
            pred = pd.Series(pred_ajustada, index=anios_fut)

        except Exception as e:
            st.error(f"Error en el cálculo: {e}")
            st.stop()
            
        tmda_24 = serie[2024]
        tmda_26 = pred[2026]
        tmda_45 = pred[2045]
        
        tasa_24_26 = ((tmda_26 / tmda_24) ** (1/2) - 1) * 100 if tmda_24 > 0 and tmda_26 > 0 else 0
        tasa_26_45 = ((tmda_45 / tmda_26) ** (1/19) - 1) * 100 if tmda_26 > 0 and tmda_45 > 0 else 0

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
            <div class='rate-box'>
                📊 Tasa Promedio Anual (2024-2026): <b>{tasa_24_26:.2f}%</b> &nbsp;|&nbsp; 
                Tasa Promedio Anual (2026-2045): <b>{tasa_26_45:.2f}%</b>
            </div>
        """, unsafe_allow_html=True)
        
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
        ax.plot(x_proyeccion, y_proyeccion, '--.', color='#2ca02c', linewidth=1, markersize=4, label='Proyección (Holt Multiplicativo)')
        
        ax.axhline(5000, color='gray', linestyle=':', alpha=0.5, label='Umbral 5.000')
        
        anio_saturacion = None
        val_saturacion = None
        full_vals = pd.concat([serie, pred])
        solo_futuro = full_vals[full_vals.index >= 2024]
        
        for y in solo_futuro.index:
            if solo_futuro[y] >= 5000:
                anio_saturacion = y
                val_saturacion = solo_futuro[y]
                break
        
        if anio_saturacion is not None:
            ax.scatter([anio_saturacion], [val_saturacion], color='red', s=150, zorder=15, edgecolors='white')
            if anio_saturacion == 2024:
                 texto_sat = f"¡SATURADO HOY!\n(Año 2024)"
            else:
                 texto_sat = f"¡SATURACIÓN!\nAño {int(anio_saturacion)}"
            offset_y = 600 if val_saturacion < 10000 else -1500
            ax.annotate(texto_sat, xy=(anio_saturacion, val_saturacion), 
                        xytext=(anio_saturacion, val_saturacion + offset_y),
                        arrowprops=dict(facecolor='red', shrink=0.05),
                        color='red', fontweight='bold', ha='center')

        ax.set_ylabel("Flujo Vehicular (veh/día)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        with st.expander("📅 Ver Histórico de Tránsito y Tasas Reales (2015-2024)", expanded=False):
            datos_hist = {'Año': anios_censo, 'TMDA Real': vals_censo.astype(int)}
            df_hist = pd.DataFrame(datos_hist)
            crecimiento = [0.0]
            for i in range(1, len(df_hist)):
                v_actual, v_ant = df_hist.iloc[i]['TMDA Real'], df_hist.iloc[i-1]['TMDA Real']
                n_anios = df_hist.iloc[i]['Año'] - df_hist.iloc[i-1]['Año']
                tasa = ((v_actual / v_ant) ** (1/n_anios) - 1) * 100 if v_ant > 0 and n_anios > 0 else 0.0
                crecimiento.append(tasa)
            df_hist['Crecimiento Anual (%)'] = crecimiento
            df_hist['Crecimiento Anual (%)'] = df_hist['Crecimiento Anual (%)'].apply(lambda x: f"{x:.2f}%")
            df_hist.at[0, 'Crecimiento Anual (%)'] = "-" 
            st.table(df_hist.set_index('Año'))

        with st.expander("📄 Ver Tabla de Proyección Futura (2025-2045)", expanded=False):
            df_tabla = pd.DataFrame({'TMDA Proyectado': pred.values}, index=pred.index)
            serie_completa_calc = pd.concat([pd.Series([tmda_24], index=[2024]), pred])
            crecimiento_pct = serie_completa_calc.pct_change() * 100
            df_tabla['Crecimiento Anual (%)'] = crecimiento_pct.loc[2025:]
            df_tabla['TMDA Proyectado'] = df_tabla['TMDA Proyectado'].astype(int)
            df_tabla['Crecimiento Anual (%)'] = df_tabla['Crecimiento Anual (%)'].apply(lambda x: f"{x:.2f}%")
            st.table(df_tabla)

        st.subheader("📋 Diagnóstico Técnico y Criterios de Diseño")
        col_diag, col_crit = st.columns([1.3, 1])

        with col_diag:
            st.markdown("#### 📢 Estado del Proyecto")
            calzada_up = calzada_info.upper()
            es_doble_via = "DOBLE" in calzada_up or "DOBLE" in rodadura_maestra

            if es_granular:
                if tmda_24 > 300: st.error(f"🔴 **PRIORIDAD ALTA:** Camino granular con {int(tmda_24)} veh/día. Supera norma (300). **Se recomienda Pavimentación.**")
                else: st.success(f"🟢 **CONSERVACIÓN:** Tránsito bajo ({int(tmda_24)} veh/día). Mantener perfilado.")
            else:
                if not es_doble_via:
                    if tmda_24 > 5000: st.error(f"🔴 **SATURACIÓN VIGENTE (2024):** Vía simple. **Estudio Segunda Calzada.**")
                    elif anio_saturacion and anio_saturacion > 2024: st.warning(f"🟡 **ALERTA FUTURA:** Saturación en {anio_saturacion}. **Planificar ampliación.**")
                    else: st.success("🟢 **OPERACIÓN NORMAL:** Capacidad suficiente.")
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

        st.markdown("<br><hr><div style='text-align: center; color: #888;'><small>Creado por José Tapia - Tesis Ingeniería Civil</small></div>", unsafe_allow_html=True)

    # =========================================================================
    # PESTAÑA 2: DISEÑO ESTRUCTURAL (ACTUALIZADA)
    # =========================================================================
    if mostrar_diseno:
        with tab_diseno:
            st.header("📏 Dimensionamiento Estructural (Método AASHTO 93)")
            st.write(f"Cálculo de conversión de camino **{carpeta}** a Pavimento Flexible.")
            st.markdown("---")

            # --- SECCIÓN 1: DATOS DE ENTRADA (Tránsito y Subrasante) ---
            st.subheader("1. Datos de Entrada")
            col_ent1, col_ent2 = st.columns(2)
            
            with col_ent1:
                eeq_diseno = info_inv['EEq 2045']
                st.metric("Tráfico en Ejes Equivalentes (EEq 2045)", f"{eeq_diseno:,.0f}")
                # El usuario puede modificar el CBR del suelo natural
                cbr_subrasante = st.number_input("C.B.R. de la Subrasante (%)", min_value=1.0, max_value=100.0, value=4.0, step=0.1)
                
            with col_ent2:
                precip = info_inv['Precipitacion promedio Mensual (mm)']
                st.metric("Precipitación Promedio", f"{precip} mm/mes")
                # Cálculo de drenaje basado en tu macro
                m_coef = 0.8 if precip > 80 else (1.0 if precip > 40 else 1.1)
                st.info(f"**Coeficientes de Drenaje ($m_2, m_3$):** `{m_coef}` (Calculado por clima regional)")

            st.markdown("<br>", unsafe_allow_html=True)

            # --- SECCIÓN 2: PROPIEDADES DE LOS MATERIALES ---
            st.subheader("2. Propiedades de los Materiales")
            col_mat1, col_mat2 = st.columns(2)
            
            with col_mat1:
                # Controles interactivos como en tu imagen
                cbr_base = st.slider("C.B.R. Base Granular (%)", min_value=40, max_value=100, value=80, step=5)
                cbr_subbase = st.slider("C.B.R. Subbase Granular (%)", min_value=15, max_value=60, value=30, step=5)

            with col_mat2:
                # Fórmulas extraídas exactamente de tu Excel para calcular a2 y a3
                a1 = 0.17  # Fijo para carpeta asfáltica
                a2 = min(a1, 0.032 * (cbr_base ** 0.32))
                a3 = min(a2, 0.058 * (cbr_subbase ** 0.19))
                
                st.markdown("**Coeficientes Estructurales ($a$):**")
                st.write(f"- Carpeta Asfáltica ($a_1$): `{a1:.3f}`")
                st.write(f"- Base Granular ($a_2$): `{a2:.3f}`")
                st.write(f"- Subbase Granular ($a_3$): `{a3:.3f}`")

            st.markdown("---")

            # --- SECCIÓN 3: CÁLCULO DE NÚMERO ESTRUCTURAL Y ESPESORES ---
            st.subheader("3. Propuesta de Espesores")
            
            # Cálculo del SN Requerido (Fórmula AASHTO simplificada)
            sn_req = 0.47 * np.log10(eeq_diseno + 1) * (1.2 / (cbr_subrasante**0.15))
            
            # Lógica de cálculo de espesores (en cm)
            d1_cm = 5.0 if eeq_diseno < 500000 else (7.0 if eeq_diseno < 1500000 else 10.0)
            d2_cm = 20.0
            sn_aportado_parcial = (a1 * d1_cm) + (a2 * d2_cm * m_coef)
            d3_cm = (sn_req - sn_aportado_parcial) / (a3 * m_coef)
            d3_cm = max(15.0, round(d3_cm, 0)) # Mínimo constructivo 15cm

            # Mostrar Cajas de Espesores
            res1, res2, res3 = st.columns(3)
            res1.info(f"**$D_1$ - Carpeta Asfáltica**\n\n### {d1_cm} cm")
            res2.info(f"**$D_2$ - Base Granular**\n\n### {d2_cm} cm")
            res3.info(f"**$D_3$ - Subbase**\n\n### {d3_cm} cm")

            # Validación final del Diseño (Igual al cuadro verde de "OK" de tu imagen)
            sn_calculado = (a1 * d1_cm) + (a2 * d2_cm * m_coef) + (a3 * d3_cm * m_coef)
            
            st.markdown("<br>", unsafe_allow_html=True)
            col_v1, col_v2 = st.columns(2)
            
            with col_v1:
                st.metric("NE Requerido", f"{sn_req:.2f}")
            with col_v2:
                if sn_calculado >= sn_req:
                    st.markdown(f"<div class='verdict-ok'>✅ OK! SN Calculado: {sn_calculado:.2f}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='verdict-bad'>⚠️ INSUFICIENTE. SN Calculado: {sn_calculado:.2f}</div>", unsafe_allow_html=True)
