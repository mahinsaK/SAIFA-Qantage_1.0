"""End-to-end orchestration for all numbers, figures, tables and report."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .copulas import (
    empirical_inverse_margins,
    empirical_lower_tail_dependence,
    fit_gaussian_correlation,
    fit_t_copula,
    simulate_gaussian_uniforms,
    simulate_t_uniforms,
    t_lower_tail_dependence,
)
from .config import AnalysisConfig
from .data import load_prices, make_returns
from .plots import plot_backtest, plot_risk_comparison, plot_tail_dependence
from .report import build_report
from .risk import holdout_metrics, portfolio_losses, var_es
from .wavelets import decompose


def _mean_off_diagonal(matrix: np.ndarray) -> float:
    return float(matrix[np.triu_indices_from(matrix, k=1)].mean())


def run_pipeline(
    root: Path,
    config: AnalysisConfig,
    refresh_data: bool = False,
) -> None:
    output = root / "output"
    figures = output / "figures"
    tables = output / "tables"
    pdfs = output / "pdf"
    for directory in (figures, tables, pdfs):
        directory.mkdir(parents=True, exist_ok=True)

    prices = load_prices(root, config, refresh_data)
    returns = make_returns(prices)
    split = int(config.train_fraction * len(returns))
    train = returns.iloc[:split].copy()
    test = returns.iloc[split:].copy()
    train_components = decompose(train, config)
    test_components = decompose(test, config)
    weights = np.asarray(config.weights)
    benchmark_corr = fit_gaussian_correlation(train.to_numpy())

    rows: list[dict[str, float | int | str]] = []
    parameters: dict[str, object] = {
        "config": asdict(config),
        "asset_order": list(config.asset_codes),
        "weights": weights.tolist(),
        "seed": config.seed,
        "simulations_per_scale": config.simulations,
        "benchmark_correlation": benchmark_corr.tolist(),
        "models": {},
    }

    seed_sequence = np.random.SeedSequence(config.seed)
    child_seeds = seed_sequence.spawn(len(config.ordered_scale_labels) * 2)
    for k, scale in enumerate(config.ordered_scale_labels):
        train_values = train_components[scale].to_numpy()
        test_values = test_components[scale].to_numpy()
        model = fit_t_copula(train_values, config.t_df_bounds)

        t_u = simulate_t_uniforms(
            model,
            config.simulations,
            np.random.default_rng(child_seeds[2 * k]),
        )
        g_u = simulate_gaussian_uniforms(
            benchmark_corr,
            config.simulations,
            np.random.default_rng(child_seeds[2 * k + 1]),
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

        rows.append(
            {
                "scale": scale,
                "empirical_tail": empirical_lower_tail_dependence(
                    train_values,
                    config.tail_quantile,
                ),
                "t_tail": t_lower_tail_dependence(model),
                "t_df": model.df,
                "mean_corr": _mean_off_diagonal(model.corr),
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
        )
        parameters["models"][scale] = {
            "t_df": model.df,
            "t_loglik": model.loglik,
            "t_correlation": model.corr.tolist(),
        }

    results = pd.DataFrame(rows)
    results["es_change_pct"] = 100 * (results["t_es"] / results["gaussian_es"] - 1)
    results.to_csv(tables / "risk_results.csv", index=False, float_format="%.10f")
    returns.to_csv(tables / "aligned_log_returns.csv", float_format="%.10f")
    with (tables / "model_parameters.json").open("w", encoding="utf-8") as handle:
        json.dump(parameters, handle, indent=2)

    plot_tail_dependence(
        results,
        figures / "tail_dependence.png",
        config.tail_quantile,
    )
    plot_risk_comparison(
        results,
        figures / "risk_comparison.png",
        config.confidence_level,
    )
    plot_backtest(
        results,
        figures / "backtest.png",
        config.confidence_level,
    )
    build_report(
        pdfs / config.report_filename,
        figures,
        results,
        prices,
        str(train.index[-1].date()),
        str(test.index[0].date()),
        config,
    )
    print("Quant Edge pipeline completed")
    print(f"Observations: {len(returns)} ({train.index.min().date()} to {test.index.max().date()})")
    print(f"Report: {pdfs / config.report_filename}")
    print(f"Results: {tables / 'risk_results.csv'}")
