"""End-to-end orchestration for all numbers, figures, tables and report."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .copulas import (
    CopulaModel,
    GaussianCopula,
    GaussianCopulaModel,
    StudentTCopulaModel,
    TCopula,
    empirical_inverse_margins,
    empirical_lower_tail_dependence,
)
from .config import AnalysisConfig
from .data import create_market_data_service
from .plots import plot_backtest, plot_risk_comparison, plot_tail_dependence
from .report import build_report
from .risk import holdout_metrics, portfolio_losses, var_es
from .wavelets import decompose


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
    root: Path,
    config: AnalysisConfig,
    refresh_data: bool,
) -> PreparedData:
    """Load, split, transform, and validate inputs for model analysis."""
    market_data = create_market_data_service(root, config)
    prices = market_data.load_prices(refresh_data)
    returns = market_data.calculate_returns(prices)
    split = int(config.train_fraction * len(returns))
    train = returns.iloc[:split].copy()
    test = returns.iloc[split:].copy()
    return PreparedData(
        prices=prices,
        returns=returns,
        train=train,
        test=test,
        train_components=decompose(train, config),
        test_components=decompose(test, config),
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
) -> tuple[dict[str, float | int | str], dict[str, object]]:
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
    t_var, t_es = var_es(
        portfolio_losses(t_sim, weights),
        config.confidence_level,
    )
    g_var, g_es = var_es(
        portfolio_losses(g_sim, weights),
        config.confidence_level,
    )
    realized = portfolio_losses(test_values, weights)
    t_holdout = holdout_metrics(
        realized,
        t_var,
        t_es,
        config.confidence_level,
    )
    g_holdout = holdout_metrics(
        realized,
        g_var,
        g_es,
        config.confidence_level,
    )

    primary_parameters = primary_model.parameters(fitted_primary)
    primary_correlation = primary_model.correlation(fitted_primary)
    result = {
        "scale": scale,
        "empirical_tail": empirical_lower_tail_dependence(
            train_values,
            config.tail_quantile,
        ),
        "t_tail": primary_model.lower_tail_dependence(fitted_primary),
        "t_df": float(primary_parameters["t_df"]),
        "mean_corr": _mean_off_diagonal(primary_correlation),
        "t_var": t_var,
        "t_es": t_es,
        "gaussian_var": g_var,
        "gaussian_es": g_es,
        "test_n": t_holdout["observations"],
        "t_exceptions": t_holdout["exceptions"],
        "t_exception_rate": t_holdout["exception_rate"],
        "t_quantile_loss": t_holdout["quantile_loss"],
        "gaussian_exceptions": g_holdout["exceptions"],
        "gaussian_exception_rate": g_holdout["exception_rate"],
        "gaussian_quantile_loss": g_holdout["quantile_loss"],
    }
    return result, primary_parameters


def _analyze_horizons(
    data: PreparedData,
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Analyze all configured horizons with deterministic child seeds."""
    seed_sequence = np.random.SeedSequence(config.seed)
    child_seeds = seed_sequence.spawn(len(config.ordered_scale_labels) * 2)
    primary_model = StudentTCopulaModel(config.t_df_bounds)
    benchmark_model = GaussianCopulaModel()
    benchmark_copula = benchmark_model.fit(data.train.to_numpy())

    rows: list[dict[str, float | int | str]] = []
    models: dict[str, object] = {}
    for index, scale in enumerate(config.ordered_scale_labels):
        result, model_parameters = _analyze_horizon(
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
        rows.append(result)
        models[scale] = model_parameters

    results = pd.DataFrame(rows)
    results["es_change_pct"] = 100 * (
        results["t_es"] / results["gaussian_es"] - 1
    )
    parameters: dict[str, object] = {
        "config": asdict(config),
        "asset_order": list(config.asset_codes),
        "weights": data.weights.tolist(),
        "seed": config.seed,
        "simulations_per_scale": config.simulations,
        "benchmark_correlation": benchmark_model.correlation(
            benchmark_copula
        ).tolist(),
        "models": models,
    }
    return results, parameters


def _save_results(
    paths: OutputPaths,
    data: PreparedData,
    results: pd.DataFrame,
    parameters: dict[str, object],
) -> None:
    """Persist machine-readable results without creating presentation artifacts."""
    results.to_csv(
        paths.tables / "risk_results.csv",
        index=False,
        float_format="%.10f",
    )
    data.returns.to_csv(
        paths.tables / "aligned_log_returns.csv",
        float_format="%.10f",
    )
    with (paths.tables / "model_parameters.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(parameters, handle, indent=2)


def _create_artifacts(
    paths: OutputPaths,
    data: PreparedData,
    results: pd.DataFrame,
    config: AnalysisConfig,
) -> None:
    """Create plots and the PDF report from completed analysis results."""
    plot_tail_dependence(
        results,
        paths.figures / "tail_dependence.png",
        config.tail_quantile,
    )
    plot_risk_comparison(
        results,
        paths.figures / "risk_comparison.png",
        config.confidence_level,
    )
    plot_backtest(
        results,
        paths.figures / "backtest.png",
        config.confidence_level,
    )
    build_report(
        paths.pdfs / config.report_filename,
        paths.figures,
        results,
        data.prices,
        str(data.train.index[-1].date()),
        str(data.test.index[0].date()),
        config,
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
    refresh_data: bool = False,
) -> None:
    """Coordinate the analysis stages without implementing their details."""
    paths = _create_output_paths(root)
    data = _prepare_data(root, config, refresh_data)
    results, parameters = _analyze_horizons(data, config)
    _save_results(paths, data, results, parameters)
    _create_artifacts(paths, data, results, config)
    _print_summary(paths, data, config)
