"""Typed domain results shared by analysis, reporting, and serialization."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


Matrix = tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class RiskEstimate:
    """Value at Risk and Expected Shortfall at one confidence level."""

    var: float
    expected_shortfall: float


@dataclass(frozen=True)
class BacktestResult:
    """Holdout diagnostics for one fixed VaR and ES forecast."""

    exceptions: int
    observations: int
    exception_rate: float
    quantile_loss: float
    tail_mean_if_exceeded: float
    es_forecast: float


@dataclass(frozen=True)
class ScaleResult:
    """All reported estimates for one wavelet investment horizon."""

    scale: str
    empirical_tail: float
    t_tail: float
    t_df: float
    mean_corr: float
    t_risk: RiskEstimate
    gaussian_risk: RiskEstimate
    t_backtest: BacktestResult
    gaussian_backtest: BacktestResult

    @property
    def es_change_pct(self) -> float:
        return 100 * (
            self.t_risk.expected_shortfall
            / self.gaussian_risk.expected_shortfall
            - 1
        )

    def to_record(self) -> dict[str, float | int | str]:
        """Return the stable flat schema used by the submission CSV."""
        return {
            "scale": self.scale,
            "empirical_tail": self.empirical_tail,
            "t_tail": self.t_tail,
            "t_df": self.t_df,
            "mean_corr": self.mean_corr,
            "t_var": self.t_risk.var,
            "t_es": self.t_risk.expected_shortfall,
            "gaussian_var": self.gaussian_risk.var,
            "gaussian_es": self.gaussian_risk.expected_shortfall,
            "test_n": self.t_backtest.observations,
            "t_exceptions": self.t_backtest.exceptions,
            "t_exception_rate": self.t_backtest.exception_rate,
            "t_quantile_loss": self.t_backtest.quantile_loss,
            "gaussian_exceptions": self.gaussian_backtest.exceptions,
            "gaussian_exception_rate": self.gaussian_backtest.exception_rate,
            "gaussian_quantile_loss": self.gaussian_backtest.quantile_loss,
            "es_change_pct": self.es_change_pct,
        }


@dataclass(frozen=True)
class HorizonModelParameters:
    """Serializable Student-t fit metadata for one horizon."""

    scale: str
    degrees_of_freedom: float
    log_likelihood: float
    correlation: Matrix

    def to_record(self) -> dict[str, object]:
        return {
            "t_df": self.degrees_of_freedom,
            "t_loglik": self.log_likelihood,
            "t_correlation": [list(row) for row in self.correlation],
        }


@dataclass(frozen=True)
class AnalysisResult:
    """Complete typed output of the multi-horizon model analysis."""

    scales: tuple[ScaleResult, ...]
    benchmark_correlation: Matrix
    model_parameters: tuple[HorizonModelParameters, ...]

    def csv_records(self) -> list[dict[str, float | int | str]]:
        return [scale.to_record() for scale in self.scales]

    def model_parameter_records(self) -> dict[str, dict[str, object]]:
        return {
            parameters.scale: parameters.to_record()
            for parameters in self.model_parameters
        }


def matrix_to_tuple(matrix: Iterable[Iterable[float]]) -> Matrix:
    """Freeze a two-dimensional numeric array without importing NumPy here."""
    return tuple(tuple(float(value) for value in row) for row in matrix)
