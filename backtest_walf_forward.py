import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf

# ==============================================================================
# 1. DESCARGA DE DATOS HISTÓRICOS REALES (2008 - 2026)
# ==============================================================================
# Tickers globales con amplio historial bursátil
TICKERS = ["MELI", "ADBE", "SPY", "QQQ", "TLT", "GLD", "KO"]

START_DATE = "2008-01-01"
END_DATE = "2026-01-01"
RISK_FREE_RATE = 0.045

print("=" * 65)
print(" FASE 3: WALK-FORWARD BACKTEST (VALIDACIÓN OUT-OF-SAMPLE)")
print("=" * 65)
print("Descargando precios históricos de Yahoo Finance...")

data = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True)['Close']
returns_df = data.pct_change().dropna()


# ==============================================================================
# 2. OPTIMIZADOR EN VENTANA MÓVIL (IN-SAMPLE)
# ==============================================================================
def optimize_weights_in_sample(window_returns, max_weight=0.30):
    """Calcula el vector de pesos que maximizó Sharpe en la ventana de entrenamiento."""
    n_assets = window_returns.shape[1]

    exp_returns = window_returns.mean() * 252
    cov_matrix = window_returns.cov() * 252

    def negative_sharpe(weights):
        p_ret = np.dot(weights, exp_returns)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        if p_vol == 0:
            return 0
        return -(p_ret - RISK_FREE_RATE) / p_vol

    init_weights = np.array([1.0 / n_assets] * n_assets)
    bounds = tuple((0.0, max_weight) for _ in range(n_assets))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    res = minimize(
        negative_sharpe,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return res.x if res.success else init_weights


# ==============================================================================
# 3. BUCLE DEL WALK-FORWARD BACKTEST
# ==============================================================================
TRAIN_WINDOW_DAYS = 252 * 2  # 2 años bursátiles para calibrar
TEST_WINDOW_DAYS = 63  # ~3 meses (rebalanceo trimestral fuera de muestra)

portfolio_out_of_sample_returns = []
total_days = len(returns_df)
current_step = TRAIN_WINDOW_DAYS

print("\nEjecutando simulación en ventanas móviles...")

while current_step + TEST_WINDOW_DAYS <= total_days:
    # Ventana In-Sample (Pasado)
    in_sample_data = returns_df.iloc[current_step - TRAIN_WINDOW_DAYS: current_step]

    # Pesos óptimos calculados con datos del pasado
    optimal_weights = optimize_weights_in_sample(in_sample_data)

    # Ventana Out-of-Sample (Futuro inmediato sin rebalancear)
    out_of_sample_data = returns_df.iloc[current_step: current_step + TEST_WINDOW_DAYS]

    # Retorno diario real obtenido por la cartera
    daily_portfolio_ret = (out_of_sample_data * optimal_weights).sum(axis=1)
    portfolio_out_of_sample_returns.append(daily_portfolio_ret)

    # Avanzar la ventana 3 meses
    current_step += TEST_WINDOW_DAYS

# Consolidación de la serie fuera de muestra
backtest_series = pd.concat(portfolio_out_of_sample_returns)
cumulative_returns = (1 + backtest_series).cumprod()

# ==============================================================================
# 4. MÉTRICAS FINALES DE PERFORMANCE HISTÓRICA
# ==============================================================================
total_ret = cumulative_returns.iloc[-1] - 1
annual_ret = (1 + total_ret) ** (252 / len(backtest_series)) - 1
annual_vol = backtest_series.std() * np.sqrt(252)
sharpe_ratio = (annual_ret - RISK_FREE_RATE) / annual_vol

peak = cumulative_returns.cummax()
drawdown = (cumulative_returns - peak) / peak
max_drawdown = drawdown.min()

print("\n" + "=" * 65)
print(" RESULTADOS DEL WALK-FORWARD BACKTEST (2008 - 2026)")
print("=" * 65)
print(
    f" Período Evaluado        : {backtest_series.index[0].strftime('%Y-%m-%d')} a {backtest_series.index[-1].strftime('%Y-%m-%d')}")
print(f" Retorno Acumulado Real   : {total_ret * 100:.2f}%")
print(f" Retorno Anualizado       : {annual_ret * 100:.2f}%")
print(f" Volatilidad Anualizada   : {annual_vol * 100:.2f}%")
print(f" Sharpe Ratio Real        : {sharpe_ratio:.2f}")
print(f" Caída Máxima (Drawdown)  : {max_drawdown * 100:.2f}%")
print("=" * 65)