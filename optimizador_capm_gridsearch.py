from enum import Enum
import numpy as np
from scipy.optimize import minimize

# ==============================================================================
# 1. DEFINICIÓN DE CLASES DE ACTIVOS PERMITIDAS (VALIDACIÓN / BLINDAJE)
# ==============================================================================
class AssetClass(Enum):
    TECH = "tech"
    FINANCIAL = "financial"
    INDUSTRIAL = "industrial"
    EV = "ev"
    GOLD = "gold"
    DEFENSIVE = "defensive"
    DURATION = "duration"
    INDEX = "index"

# Set derivado directamente del Enum para búsquedas O(1) y validación rápida
ALLOWED_ASSET_CLASSES = {ac.value for ac in AssetClass}


def validate_portfolio_inputs(assets, asset_classes, volatilities, betas):
    """
    Blindaje de datos: Verifica consistencia de dimensiones y clases permitidas.
    """
    n = len(assets)
    if not (len(asset_classes) == len(volatilities) == len(betas) == n):
        raise ValueError(
            f"  ERROR DE DIMENSIÓN: 'ASSETS' ({n}), 'ASSET_CLASSES' ({len(asset_classes)}), "
            f"'VOLATILITIES' ({len(volatilities)}) y 'BETAS' ({len(betas)}) deben tener la misma longitud."
        )

    invalid_classes = [ac for ac in asset_classes if ac not in ALLOWED_ASSET_CLASSES]
    if invalid_classes:
        raise ValueError(
            f"   CLASE DE ACTIVO NO PERMITIDA: {set(invalid_classes)}\n"
            f"   Clases válidas aceptadas: {ALLOWED_ASSET_CLASSES}"
        )
# ==============================================================================
# 1. PARÁMETROS MACRO Y UNIVERSO DE ACTIVOS
# ==============================================================================
RISK_FREE_RATE = 0.045     # Tasa libre de riesgo (4.5%)
MARKET_PREMIUM = 0.055     # Prima de riesgo del mercado (5.5%)

# Activos de inversion
ASSETS = ["MELI", "ADBE", "BABA", "SUPV", "LOMA", "LCID", "GLD", "KO", "TLT", "SPY"]

ASSET_CLASSES = ["tech", "tech", "tech", "financial", "industrial", "ev", "gold", "defensive", "duration", "index"]
N_ASSETS = len(ASSETS)

# Matriz de Volatilidades Anualizadas Teóricas/Históricas
VOLATILITIES = np.array([0.32, 0.29, 0.36, 0.45, 0.38, 0.65, 0.15, 0.14, 0.16, 0.16])

# Betas Estimados frente al Índice de Mercado (SPY)
BETAS = np.array([1.35, 1.15, 1.20, 1.50, 1.30, 1.80, 0.10, 0.55, -0.20, 1.00])

# Retornos Esperados según la Ecuación del CAPM: E(Ri) = Rf + Beta * (Rm - Rf)
EXPECTED_RETURNS_CAPM = RISK_FREE_RATE + BETAS * MARKET_PREMIUM

# Matriz de Correlación Estructural por Clase de Activo
CORR_MATRIX = np.eye(N_ASSETS) + 0.25 * (1 - np.eye(N_ASSETS))

# Matriz de Covarianza Anualizada = D * Corr * D
diag_vol = np.diag(VOLATILITIES)
COV_MATRIX = diag_vol @ CORR_MATRIX @ diag_vol

# Índices para identificar instrumentos de cobertura/renta fija (GLD, TLT)
DEFENSIVE_INDICES = [6, 8]  # GLD (idx 6), TLT (idx 8)


# ==============================================================================
# 2. FUNCIONES DE OPTIMIZACIÓN DE MARKOWITZ
# ==============================================================================
def portfolio_performance(weights, expected_returns, cov_matrix):
    """Calcula retorno esperado y volatilidad del portafolio."""
    p_ret = np.dot(weights, expected_returns)
    p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return p_ret, p_vol


def negative_sharpe(weights, expected_returns, cov_matrix, rf_rate):
    """Función objetivo a minimizar (-Sharpe Ratio)."""
    p_ret, p_vol = portfolio_performance(weights, expected_returns, cov_matrix)
    if p_vol < 1e-6:
        return 0.0
    return -(p_ret - rf_rate) / p_vol


def optimize_portfolio(max_asset_weight=0.30, min_defensive_weight=0.10):
    """
    Maximiza el Ratio de Sharpe ajustando límites de concentración individual
    y pisos de activos defensivos/renta fija.
    """
    w0 = np.full(N_ASSETS, 1.0 / N_ASSETS)
    bounds = tuple((0.0, max_asset_weight) for _ in range(N_ASSETS))

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: sum(w[i] for i in DEFENSIVE_INDICES) - min_defensive_weight},
    ]

    res = minimize(
        negative_sharpe,
        w0,
        args=(EXPECTED_RETURNS_CAPM, COV_MATRIX, RISK_FREE_RATE),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 500}
    )

    return res.x if res.success else w0


# ==============================================================================
# 3. EJECUCIÓN GRID SEARCH PARA DEFINIR POLITICA DE INVERSIÓN
# ==============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print(" FASE 1: OPTIMIZACIÓN CAPM + MARKOWITZ (GRID SEARCH)")
    print("=" * 65)

    max_weights_grid = [0.20, 0.30, 0.40]
    min_defensive_grid = [0.10, 0.20]

    best_weights = None
    best_sharpe = -np.inf

    for max_w in max_weights_grid:
        for min_def in min_defensive_grid:
            w = optimize_portfolio(max_asset_weight=max_w, min_defensive_weight=min_def)
            p_ret, p_vol = portfolio_performance(w, EXPECTED_RETURNS_CAPM, COV_MATRIX)
            sharpe = (p_ret - RISK_FREE_RATE) / p_vol

            print(f"Tope Max: {max_w*100:2.0f}% | Piso Defensivo: {min_def*100:2.0f}% ==> "
                  f"Retorno: {p_ret*100:5.2f}% | Vol: {p_vol*100:5.2f}% | Sharpe: {sharpe:4.2f}")

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = w

    print("\n" + "=" * 65)
    print(" ASIGNACIÓN ÓPTIMA RESULTANTE (MAX SHARPE)")
    print("=" * 65)
    for asset, asset_cls, weight in zip(ASSETS, ASSET_CLASSES, best_weights):
        print(f"  • {asset:6s} ({asset_cls:10s}): {weight * 100:6.2f}%")
    print("=" * 65)