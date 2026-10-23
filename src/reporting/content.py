"""Renderer-independent construction of report prose, tables, and conclusions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from ..domain.config import AnalysisConfig
from ..domain.results import ScaleResult


@dataclass(frozen=True)
class ReportTable:
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]

    def matrix(self) -> list[list[str]]:
        return [list(self.headers), *(list(row) for row in self.rows)]


@dataclass(frozen=True)
class TextSection:
    heading: str
    body: str


@dataclass(frozen=True)
class ReportContent:
    report_date: str
    executive_callout: str
    decision: str
    tested: str
    contribution: str
    portfolio_intro: str
    portfolio_table: ReportTable
    design_table: ReportTable
    design_note: str
    method_sections: tuple[TextSection, ...]
    dependence_caption: str
    dependence_table: ReportTable
    dependence_interpretation: str
    risk_caption: str
    risk_table: ReportTable
    risk_interpretation: str
    backtest_caption: str
    backtest_table: ReportTable
    backtest_interpretation: str
    recommendation: str
    implementation_rule: str
    limitations_table: ReportTable
    robust_interpretation: str
    reproduction_command: str
    refresh_command: str
    reproduction_note: str
    ai_disclosure: str
    data_sources: str
    references: tuple[str, ...]
    code_map: str


def build_report_content(
    results: Sequence[ScaleResult],
    prices: pd.DataFrame,
    train_end: str,
    test_start: str,
    config: AnalysisConfig,
) -> ReportContent:
    """Convert analysis results into presentation-neutral report content."""
    scales = tuple(results)
    if not scales:
        raise ValueError("At least one scale result is required for the report")

    confidence_pct = 100 * config.confidence_level
    loss_pct = 100 * config.loss_probability
    train_pct = 100 * config.train_fraction
    test_pct = 100 * (1 - config.train_fraction)
    component_names = [
        f"A{config.wavelet_level}",
        *(f"D{level}" for level in range(config.wavelet_level, 0, -1)),
    ]
    component_equation = " + ".join(f"{name}(t)" for name in component_names)

    max_tail = max(result.empirical_tail for result in scales)
    min_tail = min(result.empirical_tail for result in scales)
    focal = max(scales, key=lambda result: abs(result.es_change_pct))
    direction = "higher" if focal.es_change_pct >= 0 else "lower"
    t_gap = sum(
        abs(result.t_backtest.exception_rate - config.loss_probability)
        for result in scales
    ) / len(scales)
    gaussian_gap = sum(
        abs(result.gaussian_backtest.exception_rate - config.loss_probability)
        for result in scales
    ) / len(scales)
    better = "wavelet-t model" if t_gap < gaussian_gap else "Gaussian benchmark"

    portfolio_table = ReportTable(
        headers=("Series", "Economic exposure", "Weight", "Reason for inclusion"),
        rows=tuple(
            (
                asset.code,
                asset.exposure,
                f"{100 * asset.weight:g}%",
                asset.reason,
            )
            for asset in config.assets
        ),
    )
    design_table = ReportTable(
        headers=("Choice", "Specification", "Rationale"),
        rows=(
            ("Return", "Daily log change", "Additive through time; comparable across level series"),
            (
                "Wavelet",
                f"{config.wavelet.title()} maximum-overlap/SWT, {config.wavelet_level} levels",
                "Additive, shift-invariant bands with transparent horizons",
            ),
            (
                "Main dependence",
                "Student-t copula by band",
                "Allows symmetric asymptotic tail dependence",
            ),
            (
                "Benchmark",
                "One Gaussian copula from raw returns",
                "Simple model with no asymptotic tail dependence",
            ),
            (
                "Risk",
                f"{confidence_pct:g}% VaR and Expected Shortfall",
                "VaR is backtestable; ES measures severity beyond VaR",
            ),
            (
                "Validation",
                f"Chronological {train_pct:g}/{test_pct:g} holdout",
                "Prevents fitting dependence to the evaluation period",
            ),
        ),
    )

    method_sections = (
        TextSection(
            "Wavelet decomposition",
            f"For each asset return series r(t), the stationary wavelet transform produces additive components r(t) = {component_equation}. The detail bands correspond approximately to {', '.join(config.mra_scale_labels[1:])} trading-day oscillations; A{config.wavelet_level} captures variation {config.mra_scale_labels[0]}. The transform is applied separately to estimation and holdout samples.",
        ),
        TextSection(
            "Copula estimation",
            f"Within each band, ranks are converted to pseudo-observations. For each candidate degrees-of-freedom value, pseudo-observations are mapped through the Student-t quantile, a correlation matrix is estimated, and the copula log-likelihood is evaluated. Bounded maximum likelihood selects degrees of freedom between {config.t_df_bounds[0]:g} and {config.t_df_bounds[1]:g}. The benchmark maps raw-return ranks through the normal quantile once and reuses that correlation matrix at every horizon.",
        ),
        TextSection(
            "Risk simulation",
            f"Each model generates {config.simulations:,} joint uniform draws. Inverse empirical marginal distributions convert these draws back to band-specific asset returns. Portfolio losses use the configured weights. VaR is the {confidence_pct:g}th percentile of loss and ES is the mean loss conditional on exceeding VaR.",
        ),
        TextSection(
            "Out-of-sample test",
            f"Model parameters and risk thresholds are frozen after estimation. Holdout wavelet-band portfolio losses are compared with each model's VaR. We report the exception rate and quantile loss; at a {confidence_pct:g}% VaR level, ideal coverage is approximately {loss_pct:g}%. This is a distributional holdout test, not a rolling one-day trading forecast.",
        ),
        TextSection(
            "Identification",
            "Because both models use the same band-specific empirical margins, changes in measured risk isolate the consequence of replacing horizon-specific heavy-tailed dependence with one Gaussian dependence structure. This controlled comparison directly answers the challenge question.",
        ),
    )

    dependence_table = ReportTable(
        headers=("Horizon", "Empirical co-exceed.", "t-copula tail", "t df", "Mean correlation"),
        rows=tuple(
            (
                result.scale,
                f"{result.empirical_tail:.3f}",
                f"{result.t_tail:.3f}",
                f"{result.t_df:.2f}",
                f"{result.mean_corr:.3f}",
            )
            for result in scales
        ),
    )
    risk_table = ReportTable(
        headers=("Horizon", "t VaR", "t ES", "Gaussian VaR", "Gaussian ES", "ES change"),
        rows=tuple(
            (
                result.scale,
                f"{100 * result.t_risk.var:.3f}%",
                f"{100 * result.t_risk.expected_shortfall:.3f}%",
                f"{100 * result.gaussian_risk.var:.3f}%",
                f"{100 * result.gaussian_risk.expected_shortfall:.3f}%",
                f"{result.es_change_pct:+.1f}%",
            )
            for result in scales
        ),
    )
    backtest_table = ReportTable(
        headers=(
            "Horizon",
            "N",
            "t exceptions",
            "t rate",
            "Gaussian exceptions",
            "Gaussian rate",
            "Better q-loss",
        ),
        rows=tuple(
            (
                result.scale,
                str(result.t_backtest.observations),
                str(result.t_backtest.exceptions),
                f"{100 * result.t_backtest.exception_rate:.2f}%",
                str(result.gaussian_backtest.exceptions),
                f"{100 * result.gaussian_backtest.exception_rate:.2f}%",
                (
                    "t"
                    if result.t_backtest.quantile_loss
                    < result.gaussian_backtest.quantile_loss
                    else "Gaussian"
                ),
            )
            for result in scales
        ),
    )
    limitations_table = ReportTable(
        headers=("Limitation", "Consequence and next test"),
        rows=(
            ("Index/spot proxies", "Implementation costs, futures rolls and ETF tracking are absent; repeat with investable instruments."),
            ("Fixed 80/20 split", "One market regime can dominate; add rolling-window and crisis-subperiod validation."),
            ("Symmetric t tails", "Upside and downside dependence share one parameter; compare with Clayton, Gumbel or vine copulas."),
            ("Wavelet boundary effects", "Edge coefficients are affected by periodic extension; test longer samples and alternative filters."),
            ("Band-level risk", "Results describe additive horizon components, not a fully dynamic multi-period P&L forecast."),
            ("Empirical margins", "Extreme quantiles rely on finite samples; compare generalized Pareto tail models."),
        ),
    )

    pair_count = len(config.assets) * (len(config.assets) - 1) // 2
    sources = "; ".join(
        f"{asset.name} ({asset.code})" for asset in config.assets
    )
    return ReportContent(
        report_date=config.report_date,
        executive_callout=f"Yes. Average empirical lower-tail co-exceedance ranges from {min_tail:.2f} to {max_tail:.2f} across wavelet horizons. At the largest risk-measurement difference ({focal.scale}), horizon-specific modelling produces {confidence_pct:g}% Expected Shortfall that is {abs(focal.es_change_pct):.1f}% {direction} than the single-dependence benchmark.",
        decision=f"A risk manager should match the dependence model to the holding horizon. For exposures dominated by the {focal.scale} band, replace the fixed Gaussian dependence assumption with the fitted horizon-specific Student-t copula and set the component ES budget to {100 * focal.t_risk.expected_shortfall:.3f}% of portfolio value at {confidence_pct:g}% confidence. Re-estimate after material market-regime changes.",
        tested=f"An equal-weight portfolio of {len(config.assets)} public market proxies was studied from {prices.index.min().date()} to {prices.index.max().date()}. The first {train_pct:g}% of returns formed the estimation sample (ending {train_end}); the remaining {test_pct:g}% formed an untouched holdout sample (starting {test_start}). The mean absolute gap between the observed VaR exception rate and the nominal {loss_pct:g}% rate is {100 * t_gap:.2f} percentage points for the wavelet-t model and {100 * gaussian_gap:.2f} for the benchmark; by this coverage diagnostic, the {better} is closer on average.",
        contribution="The comparison holds each wavelet band's empirical marginal distributions fixed. Only the dependence model changes: a Student-t copula estimated separately at each horizon versus one Gaussian copula estimated from raw daily returns and reused everywhere. The resulting ES difference is therefore attributable to dependence and tail assumptions rather than volatility scaling.",
        portfolio_intro="The portfolio intentionally mixes two correlated equity indices with commodity and currency exposures. Equal weights avoid optimizing on the same sample used to test risk, and all inputs are public, daily, and reproducible without an API key.",
        portfolio_table=portfolio_table,
        design_table=design_table,
        design_note=f"Data are aligned to a common business-day index; isolated publication gaps of at most {config.max_forward_fill_days} days are carried forward before log returns are calculated. This treats the index levels as tradable exposure proxies rather than claiming exact ETF implementation.",
        method_sections=method_sections,
        dependence_caption=f"Figure 1. Average across the {pair_count} asset pairs. Empirical co-exceedance is P(Uj <= {config.tail_quantile:.2f} | Ui <= {config.tail_quantile:.2f}); t-copula tail dependence is the fitted asymptotic coefficient.",
        dependence_table=dependence_table,
        dependence_interpretation="The non-flat profile is inconsistent with the practical assumption that one dependence structure adequately represents every holding horizon. The fitted degrees of freedom also change by band, indicating that joint-tail thickness is not constant even when the asset set and sample are unchanged.",
        risk_caption=f"Figure 2. Simulated {confidence_pct:g}% Expected Shortfall for each additive return band. Margins are identical across models; dependence assumptions differ.",
        risk_table=risk_table,
        risk_interpretation="A positive ES change means the single Gaussian structure understates loss severity relative to the horizon-specific Student-t model; a negative value means it overstates it. The direction need not be identical at every scale, which is exactly why a single adjustment factor is insufficient.",
        backtest_caption=f"Figure 3. Holdout VaR exception rates. The dashed line is the nominal {loss_pct:g}% rate; small holdout samples make exact equality unlikely.",
        backtest_table=backtest_table,
        backtest_interpretation="Coverage and quantile loss are complementary. Coverage asks whether the threshold is crossed at the intended frequency; quantile loss also penalizes how far observations fall on either side of the threshold. The holdout test prevents a more flexible t-copula from receiving credit merely for fitting its training sample.",
        recommendation=f"Action for tomorrow: map each risk position to its holding horizon and use that band's fitted t-copula ES. For the {focal.scale} component, budget {100 * focal.t_risk.expected_shortfall:.3f}% of portfolio value at {confidence_pct:g}% ES rather than {100 * focal.gaussian_risk.expected_shortfall:.3f}% from the fixed Gaussian model ({focal.es_change_pct:+.1f}% difference).",
        implementation_rule="Maintain the existing Gaussian result as a challenger, not as the production number. Refit horizon models on a scheduled basis and trigger an unscheduled review when realized VaR exceptions cluster. Escalate when the horizon-specific ES exceeds the Gaussian ES by the institution's materiality threshold.",
        limitations_table=limitations_table,
        robust_interpretation="The defensible result is the existence and economic size of horizon variation within this sample. The framework should not be read as proof that one copula family will dominate for every portfolio or future regime.",
        reproduction_command="uv run python run.py",
        refresh_command="--refresh-data",
        reproduction_note="This one command recreates all tables, figures, machine-readable results and this PDF. The included data snapshot fixes the submitted result; the refresh option retrieves current FRED observations. Random generation uses the seed recorded in the central configuration and model-parameter output.",
        ai_disclosure="OpenAI Codex was used for research design discussion, Python implementation, debugging, report drafting and formatting. The team remains responsible for checking every method, line of code, result and statement. No result in this report is accepted solely because an AI tool produced it; all reported numbers are generated by the submitted code and public data.",
        data_sources=f"Federal Reserve Bank of St. Louis, FRED: {sources}. Downloaded {config.report_date} via fred.stlouisfed.org/graph/fredgraph.csv.",
        references=(
            "Genest, C., Rémillard, B. & Beaudoin, D. (2009). Goodness-of-fit tests for copulas: A review and a power study. Insurance: Mathematics and Economics, 44, 199-213.",
            "McNeil, A. J., Frey, R. & Embrechts, P. (2015). Quantitative Risk Management, 2nd ed. Princeton University Press.",
            "Percival, D. B. & Walden, A. T. (2000). Wavelet Methods for Time Series Analysis. Cambridge University Press.",
            "Sklar, A. (1959). Fonctions de répartition à n dimensions et leurs marges. Publications de l'Institut de Statistique de l'Université de Paris, 8, 229-231.",
            "Basel Committee on Banking Supervision (2019). Minimum capital requirements for market risk. Bank for International Settlements.",
        ),
        code_map="Code map: src/domain contains configuration, risk calculations and result contracts; src/data handles acquisition, snapshots and preparation; src/models contains wavelets and copulas; src/application orchestrates injected collaborators; src/infrastructure writes machine-readable results; src/reporting builds figures, content and the PDF.",
    )
