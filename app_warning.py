# ========================================================================
# APP STREAMLIT: ANÁLISIS DE INDICADORES MACRO
# ========================================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuración de página
st.set_page_config(layout="wide")

# Configuración visual
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# ========================================================================
# FUNCIÓN PRINCIPAL
# ========================================================================

def run_complete_analysis():
    st.title("🚀 WARNING SIGNAL")
    st.write("Investigando 9 indicadores leading para detectar recesiones (2000-2025)")

    # Sidebar para API Key
    st.sidebar.header("⚙️ Configuración")
    api_key = st.sidebar.text_input("FRED API Key", type="password")

    st.sidebar.markdown(
        """
        [🔑 Obtener API Key de FRED](https://fred.stlouisfed.org/)

        **Instrucciones:**
        1. Date de alta (es gratis)  
        2. Obtén tu API Key
        """
    )

    run_button = st.sidebar.button("▶️ Ejecutar Análisis")
    
    if not api_key:
        st.warning("⚠️ Por favor, introduce tu FRED API Key en el sidebar para continuar.")
        st.stop()
    
    if not run_button:
        st.info("👈 Pulsa el botón 'Ejecutar Análisis' en el sidebar para comenzar.")
        st.stop()

    # Configuración APIs
    fred = Fred(api_key=api_key)

    # Fechas de análisis
    start_date = "2000-01-01"
    end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    progress_bar = st.progress(0)
    status_text = st.empty()

    # ====================================================================
    # 1. DESCARGA DE DATOS
    # ====================================================================
    status_text.text("🔄 Descargando datos FRED y Yahoo Finance...")
    progress_bar.progress(10)

    # FRED
    gs10 = fred.get_series("GS10", start=start_date, end=end_date)
    gs2 = fred.get_series("GS2", start=start_date, end=end_date)
    gs3m = fred.get_series("GS3M", start=start_date, end=end_date)
    jobless_claims = fred.get_series("ICSA", start=start_date, end=end_date)
    credit_spreads = fred.get_series("BAMLH0A0HYM2", start=start_date, end=end_date)
    new_orders = fred.get_series("NEWORDER", start=start_date, end=end_date)

    # Spreads
    yield_curve_10y2y = gs10 - gs2
    yield_curve_10y3m = gs10 - gs3m

    # Yahoo Finance (descargamos todos juntos para evitar descuadres)
    prices = yf.download(["^VIX", "HG=F", "SPY", "XLF", "XLY", "XLP", "TLT", "GLD"],
                         start=start_date, end=end_date, progress=False)["Close"]

    vix = prices["^VIX"]
    copper = prices["HG=F"]
    spy = prices["SPY"]
    xlf = prices["XLF"]
    xly = prices["XLY"]
    xlp = prices["XLP"]
    tlt = prices["TLT"]
    gld = prices["GLD"]

    # Ratios
    xlf_spy_ratio = (xlf / spy).dropna()
    xly_xlp_ratio = (xly / xlp).dropna()

    # ====================================================================
    # 2. CONSTRUCCIÓN DEL DATASET
    # ====================================================================
    status_text.text("🔧 Construyendo dataset...")
    progress_bar.progress(40)

    # Crear índice mensual común
    idx = pd.date_range(start=start_date, end=end_date, freq="M")

    data = pd.DataFrame(index=idx)
    data["yield_curve_10y2y"] = yield_curve_10y2y.resample("M").last().reindex(idx)
    data["yield_curve_10y3m"] = yield_curve_10y3m.resample("M").last().reindex(idx)
    data["jobless_claims"] = jobless_claims.resample("M").last().reindex(idx)
    data["credit_spreads"] = credit_spreads.resample("M").last().reindex(idx)
    data["vix"] = vix.resample("M").last().reindex(idx)
    data["copper"] = copper.resample("M").last().reindex(idx)
    data["xlf_spy_ratio"] = xlf_spy_ratio.resample("M").last().reindex(idx)
    data["xly_xlp_ratio"] = xly_xlp_ratio.resample("M").last().reindex(idx)
    data["new_orders"] = new_orders.resample("M").last().reindex(idx)
    data["spy_price"] = spy.resample("M").last().reindex(idx)
    data["tlt_price"] = tlt.resample("M").last().reindex(idx)
    data["gld_price"] = gld.resample("M").last().reindex(idx)

    # Limpiar
    data = data.dropna()

    st.success(f"✅ Dataset construido: {len(data)} observaciones mensuales")

    # ====================================================================
    # 3. CREACIÓN DE TARGETS
    # ====================================================================
    status_text.text("📈 Creando targets...")
    progress_bar.progress(60)

    def calculate_forward_returns(prices, periods):
        returns = pd.Series(index=prices.index, dtype=float)
        for i in range(len(prices) - periods):
            if i + periods < len(prices):
                current_price = prices.iloc[i]
                future_price = prices.iloc[i + periods]
                returns.iloc[i] = (future_price - current_price) / current_price * 100
        return returns

    data["target_30d"] = calculate_forward_returns(data["spy_price"], 1)
    data["target_60d"] = calculate_forward_returns(data["spy_price"], 2)
    data["target_90d"] = calculate_forward_returns(data["spy_price"], 3)
    data = data.dropna()

    # ====================================================================
    # 4. ANÁLISIS DE ESTACIONARIEDAD
    # ====================================================================
    status_text.text("🔬 Test de estacionariedad...")
    progress_bar.progress(75)

    def test_stationarity(series, name):
        result = adfuller(series.dropna())
        return result[1] < 0.05

    indicators = ["yield_curve_10y2y", "yield_curve_10y3m", "jobless_claims",
                  "credit_spreads", "vix", "copper", "xlf_spy_ratio",
                  "xly_xlp_ratio", "new_orders"]

    stationarity_results = {ind: test_stationarity(data[ind], ind) for ind in indicators}

    data_diff = pd.DataFrame(index=data.index)
    for ind in indicators:
        if stationarity_results[ind]:
            data_diff[ind] = data[ind]
        else:
            data_diff[ind] = data[ind].diff()

    data_diff["target_30d"] = data["target_30d"]
    data_diff["target_60d"] = data["target_60d"]
    data_diff["target_90d"] = data["target_90d"]
    data_diff = data_diff.dropna()

    # ====================================================================
    # 5. PCA
    # ====================================================================
    status_text.text("🧮 Ejecutando PCA...")
    progress_bar.progress(85)

    X = data_diff[indicators].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    pca_result = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(pca_result, columns=[f"PC{i+1}" for i in range(len(indicators))], index=data_diff.index)

    data_diff["synthetic_indicator"] = pca_df["PC1"]

    # ====================================================================
    # 6. CÁLCULO DE MÉTRICAS Y SEMÁFORO
    # ====================================================================
    status_text.text("📊 Calculando métricas...")
    progress_bar.progress(90)

    # Calcular drawdowns
    spy_dd = pd.Series(index=data_diff.index, dtype=float)
    peak = data["spy_price"].iloc[0]
    for i in range(len(data)):
        current_price = data["spy_price"].iloc[i]
        if current_price > peak:
            peak = current_price
        drawdown = (current_price - peak) / peak * 100
        if i < len(spy_dd):
            spy_dd.iloc[i] = drawdown

    # Umbrales fijos
    threshold_signal = 2
    threshold_dd = -15
    
    # Calcular semáforo histórico para cada punto
    semaforo_historico = []
    for i in range(len(data_diff)):
        sig_val = data_diff["synthetic_indicator"].iloc[i]
        dd_val = spy_dd.iloc[i]
        
        cond_sig = sig_val > threshold_signal
        cond_dd = dd_val < threshold_dd
        
        if cond_sig and cond_dd:
            semaforo_historico.append("red")
        elif cond_sig or cond_dd:
            semaforo_historico.append("orange")
        else:
            semaforo_historico.append("green")
    
    data_diff["semaforo"] = semaforo_historico
    
    # Obtener valores actuales
    current_signal = data_diff["synthetic_indicator"].iloc[-1]
    current_dd = spy_dd.iloc[-1]
    
    # Lógica del semáforo actual
    condition_signal = current_signal > threshold_signal
    condition_dd = current_dd < threshold_dd
    
    if condition_signal and condition_dd:
        semaforo_color = "🔴 ROJO"
        semaforo_status = "MÁXIMA ALERTA"
        color_hex = "#FF0000"
    elif condition_signal or condition_dd:
        semaforo_color = "🟠 NARANJA"
        semaforo_status = "ALERTA MODERADA"
        color_hex = "#FFA500"
    else:
        semaforo_color = "🟢 VERDE"
        semaforo_status = "TODO OK"
        color_hex = "#00FF00"

    status_text.text("✅ Análisis completado")
    progress_bar.progress(100)
    
    # ====================================================================
    # 7. VISUALIZACIÓN EN TABS
    # ====================================================================
    
    # Crear tabs
    tab1, tab2, tab3 = st.tabs(["📊 SIGNAL", "🚦 DECISION", "🔎 DETAILS"])
    
    with tab1:
        fig, ax1 = plt.subplots(figsize=(16, 8))
        ax1.plot(data_diff.index, data_diff["synthetic_indicator"], color="purple", linewidth=2, label="WARNING SIGNAL")
        ax1.axhline(y=threshold_signal, color="purple", linestyle="--", linewidth=2, alpha=0.6, 
                    label=f"Umbral Signal ({threshold_signal})")
        ax1.set_ylabel("WARNING SIGNAL", color="purple")
        ax1.tick_params(axis="y", labelcolor="purple")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.fill_between(data_diff.index, spy_dd, 0, color="red", alpha=0.3, label="Drawdown S&P 500")
        ax2.axhline(y=threshold_dd, color="red", linestyle="--", linewidth=2, alpha=0.6, 
                    label=f"Umbral DD ({threshold_dd}%)")
        ax2.set_ylabel("Drawdown S&P 500 (%)", color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        ax2.legend(loc="upper right")

        plt.title("📉 Variable Sintética vs Drawdowns Históricos S&P 500", fontsize=14, pad=20)
        plt.xticks(rotation=45)
        plt.tight_layout()

        st.pyplot(fig)
    
    with tab2:
        st.markdown(f"<h1 style='text-align: center; color: {color_hex};'>{semaforo_color}</h1>", 
                    unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align: center;'>{semaforo_status}</h2>", 
                    unsafe_allow_html=True)
        
        st.write("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("WARNING SIGNAL Actual", f"{current_signal:.2f}")
            st.metric("Umbral", f"{threshold_signal}")
            if condition_signal:
                st.error("⚠️ Signal por ENCIMA del umbral")
            else:
                st.success("✅ Signal por debajo del umbral")
        
        with col2:
            st.metric("Drawdown Actual", f"{current_dd:.2f}%")
            st.metric("Umbral", f"{threshold_dd}%")
            if condition_dd:
                st.error("⚠️ DD por DEBAJO del umbral")
            else:
                st.success("✅ DD por encima del umbral")
        
        st.write("---")
        st.write("**Criterios del semáforo:**")
        st.write("🔴 **ROJO**: WARNING SIGNAL > 2 Y Drawdown < -15%")
        st.write("🟠 **NARANJA**: Se cumple solo UNA condición")
        st.write("🟢 **VERDE**: No se cumple ninguna condición")

    with tab3:
        st.subheader("🔎 Evolución de los 9 Indicadores")

        indicator_list = {
            "yield_curve_10y2y": "Curva 10Y-2Y",
            "yield_curve_10y3m": "Curva 10Y-3M",
            "jobless_claims": "Solicitudes Subsidio Desempleo",
            "credit_spreads": "Credit Spreads HY",
            "vix": "Volatilidad (VIX)",
            "copper": "Cobre (HG=F)",
            "xlf_spy_ratio": "XLF/SPY",
            "xly_xlp_ratio": "XLY/XLP",
            "new_orders": "New Orders"
        }

        explanations = {
            "yield_curve_10y2y": "Cuando el rendimiento de los bonos a 2 años supera al de los 10 años (curva invertida), suele anticipar recesiones. Refleja expectativas de tipos futuros más bajos por desaceleración económica.",
            "yield_curve_10y3m": "La curva 10Y-3M es uno de los indicadores más fiables de recesión. Su inversión ha precedido casi todas las recesiones en EE.UU. desde los años 60.",
            "jobless_claims": "Un aumento sostenido de las solicitudes de subsidio de desempleo indica deterioro del mercado laboral y suele coincidir con el inicio de recesiones.",
            "credit_spreads": "La ampliación de los spreads de bonos High Yield respecto a los bonos del Tesoro refleja mayor percepción de riesgo y tensiones financieras, típicos de fases de recesión.",
            "vix": "El VIX, conocido como índice del miedo, se dispara en periodos de estrés financiero y recesión, mostrando la incertidumbre de los inversores.",
            "copper": "El cobre, llamado 'Dr. Copper', es muy sensible a la actividad industrial. Caídas en su precio suelen anticipar desaceleraciones económicas globales.",
            "xlf_spy_ratio": "Cuando el sector financiero (XLF) se comporta peor que el mercado en general (SPY), suele ser señal de debilidad estructural y riesgo de recesión.",
            "xly_xlp_ratio": "El ratio entre consumo discrecional (XLY) y consumo básico (XLP) refleja confianza del consumidor. Si cae, indica que los hogares reducen gasto no esencial, anticipando recesiones.",
            "new_orders": "Las caídas en los nuevos pedidos de bienes duraderos o industriales son un indicador temprano de contracción en la producción y de recesiones."
        }

        for key, label in indicator_list.items():
            st.markdown(f"### {label}")
            fig, ax = plt.subplots(figsize=(16, 4))
            ax.plot(data.index, data[key], color="blue", linewidth=2)
            ax.set_ylabel("Valor", fontsize=11)
            ax.set_xlabel("Fecha", fontsize=11)
            ax.set_title(label, fontsize=13, pad=15)
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

            # Añadir explicación
            st.info(explanations[key])
            st.write("---")


# ========================================================================
# RUN APP
# ========================================================================
if __name__ == "__main__":
    run_complete_analysis()