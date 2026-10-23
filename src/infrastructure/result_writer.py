"""Persistence adapter for machine-readable pipeline results."""

from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd

from ..application.pipeline import OutputPaths, PreparedData
from ..domain.config import AnalysisConfig
from ..domain.results import AnalysisResult


def save_results(
    paths: OutputPaths,
    data: PreparedData,
    results: AnalysisResult,
    config: AnalysisConfig,
) -> None:
    """Persist CSV and JSON results at the output boundary."""
    pd.DataFrame(results.csv_records()).to_csv(
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
        parameters: dict[str, object] = {
            "config": asdict(config),
            "asset_order": list(config.asset_codes),
            "weights": data.weights.tolist(),
            "seed": config.seed,
            "simulations_per_scale": config.simulations,
            "benchmark_correlation": [
                list(row) for row in results.benchmark_correlation
            ],
            "models": results.model_parameter_records(),
        }
        json.dump(parameters, handle, indent=2)
