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
        
        # LIMPIEZA DE DATOS (Incluyendo 'Sector')
        cols_limpiar = ['ROL', 'ROL NUEVO', 'NOMBRE DEL CAMINO', 'Sector', 'TIPO DE CARPETA', 'CLASIFICACIÓN', 'ESTACIÓN', 'CALZADA']
        
        for col in cols_limpiar:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # CORRECCIÓN 115 CANALES
        errores_115 = ['115 Canales', '115 CANALES', '115-Canales', '115 CH', '115-CH']
        if 'ROL' in df.columns:
            df['ROL'] = df['ROL'].replace(errores_115, 'Ruta 115 CH')
        if 'ROL NUEVO' in df.columns:
            df['ROL NUEVO'] = df['ROL NUEVO'].replace(errores_115, 'Ruta 115 CH')

        return df
    except FileNotFoundError:
        st.error(f"❌ Error Crítico: No encuentro el archivo '{archivo}'. Asegúrate de que esté en la misma carpeta.")
        st.stop()

df = cargar_datos()

# --- 3. MENÚ LATERAL ---
st.sidebar.header("🔍 Panel de Control")

# Selector de Rol Oficial
roles = sorted(df['ROL NUEVO'].unique())
rol_sel = st.sidebar.selectbox("Seleccione Rol Oficial:", roles)

# Filtramos por Rol
df_rol = df[df['ROL NUEVO'] == rol_sel]

# Etiqueta para el selector
df_rol['ETIQUETA'] = df_rol['NOMBRE DEL CAMINO'] + " (" + df_rol['ESTACIÓN'] + ")"
tramo_sel = st.sidebar.selectbox("Seleccione Sector:", df_rol['ETIQUETA'].tolist())

st.sidebar.markdown("---")
btn_calc = st.sidebar.button("Generar Informe Técnico 🚀")

# --- ESTILOS CSS ---
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
    # Usamos 'Sector' directamente porque así se llama en tu Excel
    sector_especifico = fila['Sector'] if 'Sector' in fila else "Sector No Especificado"
    
    rol_oficial = fila['ROL NUEVO']
    carpeta = fila['TIPO DE CARPETA']
    clasificacion = fila['CLASIFICACIÓN']
    calzada_info = fila['CALZADA'] if 'CALZADA' in fila else "No Inf"
    
    # Título y Subtítulo
    st.title(f"📍 {nombre}")
    st.markdown(f"<div class='subtitle-sector'>Sector: {sector_especifico}</div>", unsafe_allow_html=True)
    
    # Tarjetas
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='info-card'><div class='info-label'>Rol Oficial</div><div class='info-value'>{rol_oficial}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='info-card'><div class='info-label'>Tipo de Carpeta</div><div class='info-value'>{carpeta}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='info-card'><div class='info-label'>Clasificación</div><div class='info-value'>{clasificacion}</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='info-card'><div class='info-label'>Calzada</div><div class='info-value'>{calzada_info}</div></div>", unsafe_allow_html=True)
    
    # --- CÁLCULOS MATEMÁTICOS ---
    anios_censo = [2015, 2017, 2018, 2020, 2022, 2024]
    vals_censo = fila[[f'TMDA {a}' for a in anios_censo]].values.flatten().astype(float)
    datos_reales = pd.Series(vals_censo, index=anios_censo).sort_index()
    
    # Interpolación
    serie_completa = {}
    for i in range(len(anios_censo) - 1):
        a_inicio = anios_censo[i]
        a_fin = anios_censo[i+1]
        v_inicio = datos_reales[a_inicio]
        v_fin = datos_reales[a_fin]
        serie_completa[a_inicio] = v_inicio
        n_anios = a_fin - a_inicio
        if n_anios > 1:
            if v_inicio > 0:
                r = (v_fin / v_inicio) ** (1/n_anios) - 1
            else:
                r = 0
            for k in range(1, n_anios):
                serie_completa[a_inicio + k] = v_inicio * ((1 + r) ** k)
    serie_completa[anios_censo[-1]] = datos_reales[anios_censo[-1]]
    serie = pd.Series(serie_completa).sort_index()
    
    # Proyección Holt
    try:
        try:
            modelo = ExponentialSmoothing(serie, trend='mul', seasonal=None, damped_trend=True).fit(damping_trend=0.92)
        except:
            modelo = ExponentialSmoothing(serie, trend='add', seasonal=None, damped_trend=True).fit(damping_trend=0.92)
            
        anios_fut = np.arange(2025, 2046)
        pred_raw = modelo.forecast(len(anios_fut))
        pred_raw = pd.Series(pred_raw.values, index=anios_fut)
        
        # Anclaje
        if pred_raw.iloc[0] > 0 and pred_raw.iloc[1] > 0:
            tasa_crecimiento_inicial = pred_raw.iloc[1] / pred_raw.iloc[0]
        else:
            tasa_crecimiento_inicial = 1.0
        
        base_teorica_modelo = pred_raw.iloc[0] / tasa_crecimiento_inicial
        ultimo_real = serie.iloc[-1]
        factor_ajuste = ultimo_real / base_teorica_modelo if base_teorica_modelo > 0 else 1.0
        pred_escalada = pred_raw * factor_ajuste
        
        # Safety Net
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
    
    # Tasas
    tasa_24_26 = ((tmda_26 / tmda_24) ** (1/2) - 1) * 100 if tmda_24 > 0 and tmda_26 > 0 else 0
    tasa_26_45 = ((tmda_45 / tmda_26) ** (1/19) - 1) * 100 if tmda_26 > 0 and tmda_45 > 0 else 0

    # KPI
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

    # --- GRÁFICO ---
    st.subheader("Evolución de la Demanda y Umbrales")
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Historia
    x_interp = [a for a in serie.index if a not in anios_censo]
    y_interp = [serie[a] for a in x_interp]
    x_real = anios_censo
    y_real = [serie[a] for a in x_real if a in serie.index]
    
    ax.plot(serie.index, serie.values, '-', color='gray', alpha=0.4, linewidth=1)
    ax.scatter(x_interp, y_interp, color='#fd7e14', s=40, label='Interpolado (Geométrico)', zorder=5)
    ax.scatter(x_real, y_real, color='black', s=60, label='Censo Oficial', zorder=10)
    
    # Proyección
    x_proyeccion = [2024] + list(pred.index)
    y_proyeccion = [serie[2024]] + list(pred.values)
    ax.plot(x_proyeccion, y_proyeccion, '--.', color='#2ca02c', linewidth=1, markersize=4, label='Proyección (Holt Multiplicativo)')
    
    ax.axhline(5000, color='gray', linestyle=':', alpha=0.5, label='Umbral 5.000')
    
    # Lógica de Saturación
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

    # --- TABLA HISTÓRICA ---
    with st.expander("📅 Ver Histórico de Tránsito y Tasas Reales (2015-2024)", expanded=False):
        st.write("Calculado a partir de los Censos disponibles:")
        
        datos_hist = {
            'Año': anios_censo,
            'TMDA Real': vals_censo.astype(int)
        }
        df_hist = pd.DataFrame(datos_hist)
        
        crecimiento = [0.0]
        for i in range(1, len(df_hist)):
            v_actual = df_hist.iloc[i]['TMDA Real']
            v_ant = df_hist.iloc[i-1]['TMDA Real']
            anio_actual = df_hist.iloc[i]['Año']
            anio_ant = df_hist.iloc[i-1]['Año']
            
            n_anios = anio_actual - anio_ant
            
            if v_ant > 0 and n_anios > 0:
                tasa = ((v_actual / v_ant) ** (1/n_anios) - 1) * 100
            else:
                tasa = 0.0
            crecimiento.append(tasa)
            
        df_hist['Crecimiento Anual (%)'] = crecimiento
        df_hist['Crecimiento Anual (%)'] = df_hist['Crecimiento Anual (%)'].apply(lambda x: f"{x:.2f}%")
        df_hist.at[0, 'Crecimiento Anual (%)'] = "-" 
        
        st.table(df_hist.set_index('Año'))

    # --- TABLA DE PROYECCIÓN (ACTUALIZADA) ---
    with st.expander("📄 Ver Tabla de Proyección Futura (2025-2045)", expanded=False):
        df_tabla = pd.DataFrame({'TMDA Proyectado': pred.values}, index=pred.index)
        serie_completa_calc = pd.concat([pd.Series([tmda_24], index=[2024]), pred])
        crecimiento_pct = serie_completa_calc.pct_change() * 100
        
        # Nombre de columna unificado
        df_tabla['Crecimiento Anual (%)'] = crecimiento_pct.loc[2025:]
        
        df_tabla['TMDA Proyectado'] = df_tabla['TMDA Proyectado'].astype(int)
        df_tabla['Crecimiento Anual (%)'] = df_tabla['Crecimiento Anual (%)'].apply(lambda x: f"{x:.2f}%")
        st.table(df_tabla)

    # --- SECCIÓN FINAL: DIAGNÓSTICO ---
    st.subheader("📋 Diagnóstico Técnico y Criterios de Diseño")
    
    col_diag, col_crit = st.columns([1.3, 1])

    with col_diag:
        st.markdown("#### 📢 Estado del Proyecto")
        
        carpeta_up = carpeta.upper()
        calzada_up = calzada_info.upper()
        es_no_pavimentado = any(x in carpeta_up for x in ["TIERRA", "RIPIO", "GRAVA", "SUELO"])
        es_pavimentado = not es_no_pavimentado
        es_doble_via = "DOBLE" in calzada_up or "DOBLE" in carpeta_up

        if es_no_pavimentado:
            if tmda_24 > 300:
                st.error(f"🔴 **PRIORIDAD ALTA:** Camino granular con {int(tmda_24)} veh/día. Supera norma (300). **Se recomienda Pavimentación.**")
            else:
                st.success(f"🟢 **CONSERVACIÓN:** Tránsito bajo ({int(tmda_24)} veh/día). Mantener perfilado.")
                
        elif es_pavimentado:
            if not es_doble_via:
                if tmda_24 > 5000:
                    st.error(f"🔴 **SATURACIÓN VIGENTE (2024):** Vía simple con {int(tmda_24)} veh/día. Supera capacidad. **Se sugiere Estudio de Segunda Calzada.**")
                elif anio_saturacion and anio_saturacion > 2024:
                    st.warning(f"🟡 **ALERTA FUTURA:** Se proyecta saturación para el año {anio_saturacion}. **Planificar ampliación antes de esa fecha.**")
                else:
                    st.success("🟢 **OPERACIÓN NORMAL:** Capacidad suficiente durante todo el periodo de proyección.")
            else:
                st.success("🟢 **ESTÁNDAR ADECUADO:** Doble Calzada acorde al flujo.")

    with col_crit:
        st.markdown("#### 📏 Referencia Manual de Carreteras (Vol. 3)")
        st.markdown("""
        <table class="ref-table">
            <thead>
                <tr>
                    <th>TMDA (veh/día)</th>
                    <th>Categoría</th>
                    <th>Intervención Sugerida</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><b>&lt; 300</b></td>
                    <td>Tránsito Bajo</td>
                    <td>Mantener Carpeta Granular</td>
                </tr>
                <tr>
                    <td><b>300 – 5.000</b></td>
                    <td>Tránsito Medio</td>
                    <td>Pavimentación (Sello/Asfalto)</td>
                </tr>
                <tr>
                    <td><b>&gt; 5.000</b></td>
                    <td>Saturación</td>
                    <td>Estudio de Segunda Calzada</td>
                </tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)

    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; color: #888;'><small>Creado por José Tapia - Tesis Ingeniería Civil</small></div>", unsafe_allow_html=True)