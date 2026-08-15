"""
Simulación Monte Carlo + optimización de mínima varianza (GMV) de la
cartera REAL del usuario: ADBE, BABA, BIDU, LCID, LEDE, LOMA, MELI, NU,
PBR, SUPV
==========================================================================

DIAGNÓSTICO PREVIO (léase antes de mirar los números):
Esta cartera es equity puro, sin renta fija, y está concentrada en TRES
bloques de riesgo EM + un outlier de altísimo riesgo idiosincrático:
  - Argentina doméstico:  LEDE (BCBA, peso-denominada), LOMA, SUPV
  - Brasil:                PBR, MELI (ingresos fuerte exposición BRL), NU
  - China (ADRs):          BABA, BIDU
  - Outlier de riesgo:     LCID (EV en modo "turnaround", rango de 52
                            semanas de USD 2,37 a USD 25,23 - volatilidad
                            extrema, riesgo de going-concern no descartado
                            del todo por el propio mercado)
  - Único "ancla" de bajo riesgo relativo: ADBE

NO hay ningún activo defensivo, de duración o refugio en el set. Antes
de optimizar pesos DENTRO de este universo, tené en cuenta que el
universo en sí mismo ya viene con un sesgo de concentración fuerte -
la optimización de mínima varianza puede reducir el riesgo relativo
DENTRO de estos 10 nombres, pero no puede inventar diversificación que
no está en el universo que elegiste.

METODOLOGÍA (consistente con los scripts anteriores):
- GBM (Black-Scholes) bajo medida neutral al riesgo: drift = tasa libre
  de riesgo, NO un retorno esperado subjetivo (mismo criterio que
  siempre: mu no es un input confiable, no se inventa).
- Matriz de covarianza por clase de activo (factores: ar_domestic,
  brazil, china_adr, us_tech, ev_distressed). Acá, a diferencia del
  script con bonos, la covarianza SÍ tiene forma cerrada (analítica:
  cov_ij = sigma_i * sigma_j * rho_ij) porque todos los activos son GBM
  puro - no hace falta simular para calibrarla como con los bonos
  (donde el default y el proceso OU rompían la forma cerrada).
- Optimización de mínima varianza global (GMV): SOLO usa la matriz de
  covarianza, no retorno esperado.

IMPORTANTE — PRECIOS Y VOLATILIDADES:
Precios de referencia ~11/12-ago-2026 tomados de fuentes públicas
(Yahoo Finanzas, Investing.com, TradingView, Robinhood). Los de BABA,
LOMA y PBR son estimaciones aproximadas -- varias fuentes daban datos
desactualizados o inconsistentes al momento de armar esto. Las
volatilidades son estimaciones de referencia (no implícitas de
mercado). ACTUALIZÁ ambos antes de usar esto para decidir algo real.
El precio de LEDE está expresado como equivalente en USD -- LEDE
cotiza en pesos en BCBA, así que hay riesgo cambiario ARS/USD implícito
que este script NO modela explícitamente (ver nota al final).
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.optimize import minimize

# ----------------------------------------------------------------------
# 1. DEFINICIÓN DEL UNIVERSO
# ----------------------------------------------------------------------

@dataclass
class Stock:
    ticker: str
    emisor: str
    asset_class: str
    initial_price: float
    volatility: float
    # Sin expected_return a propósito: drift = tasa libre de riesgo
    # (medida neutral al riesgo). Ver docstring del módulo.


RISK_FREE_RATE = 0.045

# Precios ~ago-2026 (ver nota de calidad de dato en el docstring)
STOCKS = [
    Stock(ticker="ADBE", emisor="Adobe", asset_class="us_tech",
          initial_price=261.70, volatility=0.32),

    Stock(ticker="BABA", emisor="Alibaba", asset_class="china_adr",
          initial_price=130.00, volatility=0.42),   # precio aproximado - VERIFICAR

    Stock(ticker="BIDU", emisor="Baidu", asset_class="china_adr",
          initial_price=106.10, volatility=0.40),

    Stock(ticker="LCID", emisor="Lucid Group", asset_class="ev_distressed",
          initial_price=6.69, volatility=0.75),      # volatilidad extrema, ver docstring

    Stock(ticker="LEDE", emisor="Ledesma S.A.A.I.", asset_class="ar_domestic",
          initial_price=8.50, volatility=0.45),       # equivalente USD aprox - VERIFICAR, FX no modelado

    Stock(ticker="LOMA", emisor="Loma Negra", asset_class="ar_domestic",
          initial_price=9.00, volatility=0.40),        # precio aproximado - VERIFICAR

    Stock(ticker="MELI", emisor="MercadoLibre", asset_class="brazil",
          initial_price=1650.00, volatility=0.38),

    Stock(ticker="NU", emisor="Nu Holdings", asset_class="brazil",
          initial_price=13.60, volatility=0.45),

    Stock(ticker="PBR", emisor="Petrobras", asset_class="brazil",
          initial_price=14.50, volatility=0.35),      # precio aproximado - VERIFICAR

    Stock(ticker="SUPV", emisor="Grupo Supervielle", asset_class="ar_domestic",
          initial_price=9.62, volatility=0.55),
]

N_ASSETS = len(STOCKS)
TICKERS = [s.ticker for s in STOCKS]
ASSET_CLASSES = [s.asset_class for s in STOCKS]

# ------------------------------------------------------------------
# Correlación por clase. Los tres bloques EM (ar_domestic, brazil,
# china_adr) tienen correlación cruzada moderada-alta entre sí (todos
# reaccionan a apetito de riesgo EM global / dólar), pero cada uno
# tiene también su propio driver local. LCID es un outlier de alto beta
# de mercado US pero con riesgo idiosincrático dominante.
# ------------------------------------------------------------------
SAME_CLASS_CORR = {
    "us_tech": 1.0, "china_adr": 0.70, "ar_domestic": 0.60,
    "brazil": 0.55, "ev_distressed": 1.0,
}
CROSS_CLASS_CORR = {
    ("ar_domestic", "brazil"): 0.35,
    ("ar_domestic", "china_adr"): 0.15,
    ("ar_domestic", "ev_distressed"): 0.15,
    ("ar_domestic", "us_tech"): 0.15,
    ("brazil", "china_adr"): 0.25,
    ("brazil", "ev_distressed"): 0.25,
    ("brazil", "us_tech"): 0.30,
    ("china_adr", "ev_distressed"): 0.20,
    ("china_adr", "us_tech"): 0.30,
    ("ev_distressed", "us_tech"): 0.35,
}


def class_corr(c1, c2):
    if c1 == c2:
        return SAME_CLASS_CORR[c1]
    return CROSS_CLASS_CORR[tuple(sorted([c1, c2]))]


CORR_MATRIX = np.array([[class_corr(c1, c2) for c2 in ASSET_CLASSES] for c1 in ASSET_CLASSES])
np.fill_diagonal(CORR_MATRIX, 1.0)
CHOL = np.linalg.cholesky(CORR_MATRIX)

VOLS = np.array([s.volatility for s in STOCKS])

# ----------------------------------------------------------------------
# 2. COVARIANZA ANALÍTICA (válida porque todos los activos son GBM puro)
# ----------------------------------------------------------------------

COV_MATRIX = np.outer(VOLS, VOLS) * CORR_MATRIX

# ----------------------------------------------------------------------
# 3. OPTIMIZACIÓN DE MÍNIMA VARIANZA GLOBAL (GMV)
# ----------------------------------------------------------------------

MAX_WEIGHT_PER_ASSET = 0.25   # tope de concentración por activo
MAX_WEIGHT_PER_CLASS = 0.45   # tope de concentración por bloque de riesgo


def optimize_min_variance(cov_matrix):
    def portfolio_variance(w):
        return w @ cov_matrix @ w

    class_masks = {c: np.array([1.0 if ac == c else 0.0 for ac in ASSET_CLASSES])
                   for c in set(ASSET_CLASSES)}

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    for c, mask in class_masks.items():
        constraints.append({
            "type": "ineq",
            "fun": (lambda w, m=mask: MAX_WEIGHT_PER_CLASS - np.sum(w * m))
        })

    bounds = [(0.0, MAX_WEIGHT_PER_ASSET) for _ in range(N_ASSETS)]
    w0 = np.full(N_ASSETS, 1 / N_ASSETS)

    result = minimize(portfolio_variance, w0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"maxiter": 500, "ftol": 1e-12})
    if not result.success:
        raise RuntimeError(f"Optimización falló: {result.message}")
    return result.x


# ----------------------------------------------------------------------
# 4. SIMULACIÓN MONTE CARLO (GBM correlacionado, vectorizado)
# ----------------------------------------------------------------------

def simulate(weights, horizon_years, n_simulations, seed):
    rng = np.random.default_rng(seed=seed)
    n_steps = int(horizon_years * 12)
    dt = 1 / 12

    z = rng.standard_normal((n_simulations, n_steps, N_ASSETS))
    correlated_z = np.einsum("ijk,lk->ijl", z, CHOL)

    drift = (RISK_FREE_RATE - 0.5 * VOLS ** 2) * dt
    diffusion = VOLS * np.sqrt(dt) * correlated_z
    log_returns = drift + diffusion
    cum_log_returns = np.cumsum(log_returns, axis=1)

    prices0 = np.array([s.initial_price for s in STOCKS])
    prices = prices0 * np.exp(cum_log_returns)  # (n_sim, n_steps, N_ASSETS)

    capital_per_asset = INITIAL_CAPITAL * weights
    num_shares = capital_per_asset / prices0
    asset_values = prices * num_shares  # (n_sim, n_steps, N_ASSETS)

    # agregar el punto inicial (t=0)
    initial_values = np.tile(capital_per_asset, (n_simulations, 1, 1))
    asset_values = np.concatenate([initial_values, asset_values], axis=1)

    portfolio_value = asset_values.sum(axis=2)
    return portfolio_value, asset_values


INITIAL_CAPITAL = 10_000.0
N_SIMULATIONS = 20_000
HORIZON_YEARS = 1


if __name__ == "__main__":
    print(f"Universo: {N_ASSETS} activos — {', '.join(TICKERS)}")
    print("Clases:", ", ".join(f"{t}({c})" for t, c in zip(TICKERS, ASSET_CLASSES)))

    print(f"\nDesvío estándar anual estimado (equal-weight naive, 1/{N_ASSETS} c/u):")
    naive_weights = np.full(N_ASSETS, 1 / N_ASSETS)
    naive_var = naive_weights @ COV_MATRIX @ naive_weights
    print(f"  {np.sqrt(naive_var)*100:.2f}%")

    print(f"\nOptimizando pesos de mínima varianza (GMV, tope {MAX_WEIGHT_PER_ASSET:.0%} "
          f"por activo, tope {MAX_WEIGHT_PER_CLASS:.0%} por bloque de riesgo)...")
    optimal_weights = optimize_min_variance(COV_MATRIX)
    optimal_var = optimal_weights @ COV_MATRIX @ optimal_weights

    print("\nPesos óptimos (mínima varianza):")
    for t, cl, w in zip(TICKERS, ASSET_CLASSES, optimal_weights):
        print(f"  {t:6s} ({cl:14s}): {w*100:>6.2f}%")
    print(f"\nDesvío estándar anual estimado (óptimo GMV): {np.sqrt(optimal_var)*100:.2f}%")

    print("\nExposición por bloque de riesgo (pesos óptimos):")
    for c in sorted(set(ASSET_CLASSES)):
        exposure = sum(w for w, ac in zip(optimal_weights, ASSET_CLASSES) if ac == c)
        print(f"  {c:14s}: {exposure*100:>6.2f}%")

    print(f"\nCorriendo simulación completa a {HORIZON_YEARS} años "
          f"({N_SIMULATIONS:,} simulaciones) con pesos óptimos...")
    portfolio_value, asset_values = simulate(
        optimal_weights, horizon_years=HORIZON_YEARS, n_simulations=N_SIMULATIONS, seed=42
    )
    final_values = portfolio_value[:, -1]
    returns = (final_values / INITIAL_CAPITAL) - 1

    print("\n" + "=" * 62)
    print(f"RESULTADOS — Horizonte: {HORIZON_YEARS} años | Capital inicial: USD {INITIAL_CAPITAL:,.0f}")
    print("=" * 62)
    print(f"Valor final promedio:      USD {final_values.mean():>12,.0f}")
    print(f"Retorno promedio:          {returns.mean()*100:>12.2f}%")
    print(f"Retorno mediano:           {np.median(returns)*100:>12.2f}%")
    print(f"Desvío estándar retorno:   {returns.std()*100:>12.2f}%")
    print("-" * 62)

    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(final_values, p)
        ret = np.percentile(returns, p)
        print(f"Percentil {p:>2}:  USD {val:>10,.0f}   ({ret*100:>6.2f}%)")

    print("-" * 62)
    var_95 = INITIAL_CAPITAL - np.percentile(final_values, 5)
    cvar_95 = INITIAL_CAPITAL - final_values[final_values <= np.percentile(final_values, 5)].mean()
    print(f"VaR 95%:   USD {var_95:>10,.0f}")
    print(f"CVaR 95%:  USD {cvar_95:>10,.0f}")
    print(f"Probabilidad de pérdida de capital nominal: {(final_values < INITIAL_CAPITAL).mean()*100:>6.2f}%")

    # ------------------------------------------------------------------
    # Comparación: cartera óptima vs. equal-weight (misma corrida MC)
    # ------------------------------------------------------------------
    portfolio_naive, _ = simulate(naive_weights, horizon_years=HORIZON_YEARS,
                                   n_simulations=N_SIMULATIONS, seed=42)
    final_naive = portfolio_naive[:, -1]
    print("-" * 62)
    print("Comparación con equal-weight (misma simulación, mismos shocks):")
    print(f"  Óptima GMV  -> desvío simulado: {returns.std()*100:.2f}%  "
          f"| VaR95: USD {var_95:,.0f}  | prob. pérdida: {(final_values < INITIAL_CAPITAL).mean()*100:.1f}%")
    ret_naive = (final_naive / INITIAL_CAPITAL) - 1
    var95_naive = INITIAL_CAPITAL - np.percentile(final_naive, 5)
    print(f"  Equal-weight-> desvío simulado: {ret_naive.std()*100:.2f}%  "
          f"| VaR95: USD {var95_naive:,.0f}  | prob. pérdida: {(final_naive < INITIAL_CAPITAL).mean()*100:.1f}%")

    # ------------------------------------------------------------------
    # Gráficos
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(final_values, bins=80, color="#2b6cb0", alpha=0.85, edgecolor="white", label="GMV")
    axes[0].hist(final_naive, bins=80, color="#dd6b20", alpha=0.45, edgecolor="white", label="Equal-weight")
    axes[0].axvline(INITIAL_CAPITAL, color="black", linestyle="--", linewidth=1)
    axes[0].set_title(f"Distribución del valor de cartera a {HORIZON_YEARS} años")
    axes[0].set_xlabel("Valor de cartera (USD)")
    axes[0].legend()

    rng_plot = np.random.default_rng(7)
    sample_idx = rng_plot.choice(N_SIMULATIONS, size=150, replace=False)
    time_axis = np.linspace(0, HORIZON_YEARS, portfolio_value.shape[1])
    for idx in sample_idx:
        axes[1].plot(time_axis, portfolio_value[idx], color="#2b6cb0", alpha=0.08, linewidth=0.8)
    axes[1].plot(time_axis, portfolio_value.mean(axis=0), color="black", linewidth=2, label="Promedio")
    axes[1].plot(time_axis, np.percentile(portfolio_value, 5, axis=0), color="crimson", linestyle="--", label="P5")
    axes[1].plot(time_axis, np.percentile(portfolio_value, 95, axis=0), color="green", linestyle="--", label="P95")
    axes[1].set_title("Trayectorias de cartera (pesos GMV)")
    axes[1].set_xlabel("Años")
    axes[1].legend()

    x = np.arange(N_ASSETS)
    width = 0.35
    axes[2].bar(x - width/2, naive_weights * 100, width, label="Equal-weight", color="#dd6b20", alpha=0.7)
    axes[2].bar(x + width/2, optimal_weights * 100, width, label="GMV óptimo", color="#2b6cb0")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(TICKERS, rotation=45)
    axes[2].set_title("Composición: equal-weight vs. GMV")
    axes[2].set_ylabel("Peso (%)")
    axes[2].legend()

    plt.tight_layout()
    output_path = "monte_carlo_portafolio_actual_resultado.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nGráfico guardado en {output_path}")