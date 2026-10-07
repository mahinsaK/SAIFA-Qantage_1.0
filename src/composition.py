"""Production composition root for pipeline dependencies."""

from __future__ import annotations

from pathlib import Path

from .artifacts import publish_artifacts
from .config import AnalysisConfig
from .copulas import GaussianCopulaModel, StudentTCopulaModel
from .data import create_market_data_service
from .outputs import save_results
from .pipeline import PipelineDependencies
from .wavelets import decompose


def create_pipeline_dependencies(
    root: Path,
    config: AnalysisConfig,
) -> PipelineDependencies:
    """Select concrete production adapters for one pipeline run."""
    return PipelineDependencies(
        market_data=create_market_data_service(root, config),
        decomposer=decompose,
        primary_model=StudentTCopulaModel(config.t_df_bounds),
        benchmark_model=GaussianCopulaModel(),
        result_writer=save_results,
        artifact_publisher=publish_artifacts,
    )
