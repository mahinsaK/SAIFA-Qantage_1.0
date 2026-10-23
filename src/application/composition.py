"""Production composition root for pipeline dependencies."""

from __future__ import annotations

from pathlib import Path

from ..data.service import create_market_data_service
from ..domain.config import AnalysisConfig
from ..infrastructure.result_writer import save_results
from ..models.copulas import GaussianCopulaModel, StudentTCopulaModel
from ..models.wavelets import decompose
from ..reporting.artifacts import publish_artifacts
from .pipeline import PipelineDependencies


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
