"""Risk measurement and holdout scoring."""

from __future__ import annotations

import numpy as np


def portfolio_losses(asset_returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return -(asset_returns @ weights)


def var_es(losses: np.ndarray, alpha: float) -> tuple[float, float]:
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    es = float(np.mean(tail))
    return var, es


def quantile_loss(realized_losses: np.ndarray, forecast_var: float, alpha: float) -> float:
    error = realized_losses - forecast_var
    return float(np.mean(np.maximum(alpha * error, (alpha - 1) * error)))


def holdout_metrics(
    realized_losses: np.ndarray,
    var: float,
    es: float,
    alpha: float,
) -> dict[str, float]:
    exceed = realized_losses > var
    realized_tail = realized_losses[exceed]
    return {
        "exceptions": int(exceed.sum()),
        "observations": int(len(realized_losses)),
        "exception_rate": float(exceed.mean()),
        "quantile_loss": quantile_loss(realized_losses, var, alpha),
        "tail_mean_if_exceeded": float(realized_tail.mean()) if len(realized_tail) else float("nan"),
        "es_forecast": float(es),
    }
