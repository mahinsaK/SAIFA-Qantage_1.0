"""End-to-end orchestration for all numbers, figures, tables and report."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from .copulas import (
    CopulaModel,
    GaussianCopula,
    TCopula,
    empirical_inverse_margins,
    empirical_lower_tail_dependence,
)
from .config import AnalysisConfig
from .results import (
    AnalysisResult,
    HorizonModelParameters,
    ScaleResult,
    matrix_to_tuple,
)
from .risk import holdout_metrics, portfolio_losses, var_es


class MarketDataGateway(Protocol):
    """Market-data operations required by the pipeline."""

    def load_prices(self, refresh: bool = False) -> pd.DataFrame: ...

    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame: ...


@dataclass(frozen=True)
class OutputPaths:
    """Filesystem destinations produced by one pipeline run."""

    figures: Path
    tables: Path
    pdfs: Path


@dataclass(frozen=True)
class PreparedData:
    """Validated inputs needed by horizon-level analysis."""

    prices: pd.DataFrame
    returns: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
    train_components: dict[str, pd.DataFrame]
    test_components: dict[str, pd.DataFrame]
    weights: np.ndarray


ResultWriter = Callable[
    [OutputPaths, PreparedData, AnalysisResult, AnalysisConfig],
    None,
]
ArtifactPublisher = Callable[
    [OutputPaths, PreparedData, AnalysisResult, AnalysisConfig],
    None,
]
WaveletDecomposer = Callable[
    [pd.DataFrame, AnalysisConfig],
    dict[str, pd.DataFrame],
]


@dataclass(frozen=True)
class PipelineDependencies:
    """External collaborators selected by the application composition root."""

    market_data: MarketDataGateway
    decomposer: WaveletDecomposer
    primary_model: CopulaModel[TCopula]
    benchmark_model: CopulaModel[GaussianCopula]
    result_writer: ResultWriter
    artifact_publisher: ArtifactPublisher


def _mean_off_diagonal(matrix: np.ndarray) -> float:
    return float(matrix[np.triu_indices_from(matrix, k=1)].mean())


def _create_output_paths(root: Path) -> OutputPaths:
    output = root / "output"
    paths = OutputPaths(
        figures=output / "figures",
        tables=output / "tables",
        pdfs=output / "pdf",
    )
    for directory in (paths.figures, paths.tables, paths.pdfs):
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def _prepare_data(
    config: AnalysisConfig,
    refresh_data: bool,
    dependencies: PipelineDependencies,
) -> PreparedData:
    """Load, split, transform, and validate inputs for model analysis."""
    prices = dependencies.market_data.load_prices(refresh_data)
    returns = dependencies.market_data.calculate_returns(prices)
    split = int(config.train_fraction * len(returns))
    train = returns.iloc[:split].copy()
    test = returns.iloc[split:].copy()
    return PreparedData(
        prices=prices,
        returns=returns,
        train=train,
        test=test,
        train_components=dependencies.decomposer(train, config),
        test_components=dependencies.decomposer(test, config),
        weights=np.asarray(config.weights),
    )


def _analyze_horizon(
    scale: str,
    train_values: np.ndarray,
    test_values: np.ndarray,
    weights: np.ndarray,
    primary_model: CopulaModel[TCopula],
    benchmark_model: CopulaModel[GaussianCopula],
    benchmark_copula: GaussianCopula,
    primary_seed: np.random.SeedSequence,
    benchmark_seed: np.random.SeedSequence,
    config: AnalysisConfig,
) -> tuple[ScaleResult, HorizonModelParameters]:
    """Fit, simulate, and score both models for one investment horizon."""
    fitted_primary = primary_model.fit(train_values)

    t_u = primary_model.simulate_uniforms(
        fitted_primary,
        config.simulations,
        np.random.default_rng(primary_seed),
    )
    g_u = benchmark_model.simulate_uniforms(
        benchmark_copula,
        config.simulations,
        np.random.default_rng(benchmark_seed),
    )
    t_sim = empirical_inverse_margins(t_u, train_values)
    g_sim = empirical_inverse_margins(g_u, train_values)
    t_risk = var_es(
        portfolio_losses(t_sim, weights),
        config.confidence_level,
    )
    gaussian_risk = var_es(
        portfolio_losses(g_sim, weights),
        config.confidence_level,
    )
    realized = portfolio_losses(test_values, weights)
    t_holdout = holdout_metrics(
        realized,
        t_risk,
        config.confidence_level,
    )
    gaussian_holdout = holdout_metrics(
        realized,
        gaussian_risk,
        config.confidence_level,
    )

    primary_correlation = primary_model.correlation(fitted_primary)
    result = ScaleResult(
        scale=scale,
        empirical_tail=empirical_lower_tail_dependence(
            train_values,
            config.tail_quantile,
        ),
        t_tail=primary_model.lower_tail_dependence(fitted_primary),
        t_df=fitted_primary.df,
        mean_corr=_mean_off_diagonal(primary_correlation),
        t_risk=t_risk,
        gaussian_risk=gaussian_risk,
        t_backtest=t_holdout,
        gaussian_backtest=gaussian_holdout,
    )
    parameters = HorizonModelParameters(
        scale=scale,
        degrees_of_freedom=fitted_primary.df,
        log_likelihood=fitted_primary.loglik,
        correlation=matrix_to_tuple(fitted_primary.corr),
    )
    return result, parameters


def _analyze_horizons(
    data: PreparedData,
    config: AnalysisConfig,
    dependencies: PipelineDependencies,
) -> AnalysisResult:
    """Analyze all configured horizons with deterministic child seeds."""
    seed_sequence = np.random.SeedSequence(config.seed)
    child_seeds = seed_sequence.spawn(len(config.ordered_scale_labels) * 2)
    primary_model = dependencies.primary_model
    benchmark_model = dependencies.benchmark_model
    benchmark_copula = benchmark_model.fit(data.train.to_numpy())

    scale_results: list[ScaleResult] = []
    model_parameters: list[HorizonModelParameters] = []
    for index, scale in enumerate(config.ordered_scale_labels):
        result, horizon_parameters = _analyze_horizon(
            scale=scale,
            train_values=data.train_components[scale].to_numpy(),
            test_values=data.test_components[scale].to_numpy(),
            weights=data.weights,
            primary_model=primary_model,
            benchmark_model=benchmark_model,
            benchmark_copula=benchmark_copula,
            primary_seed=child_seeds[2 * index],
            benchmark_seed=child_seeds[2 * index + 1],
            config=config,
        )
        scale_results.append(result)
        model_parameters.append(horizon_parameters)

    return AnalysisResult(
        scales=tuple(scale_results),
        benchmark_correlation=matrix_to_tuple(
            benchmark_model.correlation(benchmark_copula)
        ),
        model_parameters=tuple(model_parameters),
    )


def _print_summary(
    paths: OutputPaths,
    data: PreparedData,
    config: AnalysisConfig,
) -> None:
    """Print the small command-line completion summary."""
    print("Quant Edge pipeline completed")
    print(
        f"Observations: {len(data.returns)} "
        f"({data.train.index.min().date()} to {data.test.index.max().date()})"
    )
    print(f"Report: {paths.pdfs / config.report_filename}")
    print(f"Results: {paths.tables / 'risk_results.csv'}")


def run_pipeline(
    root: Path,
    config: AnalysisConfig,
    dependencies: PipelineDependencies,
    refresh_data: bool = False,
) -> None:
    """Coordinate the analysis stages without implementing their details."""
    paths = _create_output_paths(root)
    data = _prepare_data(config, refresh_data, dependencies)
    results = _analyze_horizons(data, config, dependencies)
    dependencies.result_writer(paths, data, results, config)
    dependencies.artifact_publisher(paths, data, results, config)
    _print_summary(paths, data, config)
