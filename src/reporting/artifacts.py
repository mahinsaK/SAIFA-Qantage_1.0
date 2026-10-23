"""Publication adapter for figures and the PDF report."""

from __future__ import annotations

from ..application.pipeline import OutputPaths, PreparedData
from ..domain.config import AnalysisConfig
from ..domain.results import AnalysisResult
from .content import build_report_content
from .pdf import build_report
from .plots import plot_backtest, plot_risk_comparison, plot_tail_dependence


def publish_artifacts(
    paths: OutputPaths,
    data: PreparedData,
    results: AnalysisResult,
    config: AnalysisConfig,
) -> None:
    """Create plots and the PDF report from completed analysis results."""
    plot_tail_dependence(
        results.scales,
        paths.figures / "tail_dependence.png",
        config.tail_quantile,
    )
    plot_risk_comparison(
        results.scales,
        paths.figures / "risk_comparison.png",
        config.confidence_level,
    )
    plot_backtest(
        results.scales,
        paths.figures / "backtest.png",
        config.confidence_level,
    )
    content = build_report_content(
        results.scales,
        data.prices,
        str(data.train.index[-1].date()),
        str(data.test.index[0].date()),
        config,
    )
    build_report(
        paths.pdfs / config.report_filename,
        paths.figures,
        content,
    )
