"""Generate the submission PDF from analysis outputs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from .config import AnalysisConfig
from .results import ScaleResult


NAVY = colors.HexColor("#173A63")
BLUE = colors.HexColor("#2C6AA0")
PALE_BLUE = colors.HexColor("#EAF2F8")
PALE_GOLD = colors.HexColor("#FFF3D6")
TEXT = colors.HexColor("#273445")
MUTED = colors.HexColor("#66778A")
LINE = colors.HexColor("#D7E0E8")


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("CoverKicker", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=10, textColor=BLUE, leading=13, alignment=TA_CENTER, spaceAfter=10))
    styles.add(ParagraphStyle("CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=27, leading=32, textColor=NAVY, alignment=TA_CENTER, spaceAfter=13))
    styles.add(ParagraphStyle("CoverSub", parent=styles["Normal"], fontSize=12, leading=17, textColor=TEXT, alignment=TA_CENTER))
    styles.add(ParagraphStyle("H1x", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=NAVY, spaceBefore=4, spaceAfter=10))
    styles.add(ParagraphStyle("H2x", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=14, textColor=NAVY, spaceBefore=9, spaceAfter=5))
    styles.add(ParagraphStyle("Bodyx", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.2, leading=13, textColor=TEXT, spaceAfter=7))
    styles.add(ParagraphStyle("Smallx", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.8, leading=10.5, textColor=TEXT, spaceAfter=4))
    styles.add(ParagraphStyle("Callout", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=10.5, leading=15, textColor=NAVY, spaceAfter=0))
    styles.add(ParagraphStyle("Captionx", parent=styles["BodyText"], fontName="Helvetica-Oblique", fontSize=7.5, leading=10, textColor=MUTED, spaceAfter=7))
    return styles


def _header_footer(canvas, doc):
    canvas.saveState()
    if doc.page > 1:
        canvas.setStrokeColor(LINE)
        canvas.line(18 * mm, A4[1] - 15 * mm, A4[0] - 18 * mm, A4[1] - 15 * mm)
        canvas.setFont("Helvetica-Bold", 7.5)
        canvas.setFillColor(NAVY)
        canvas.drawString(18 * mm, A4[1] - 11.5 * mm, "SAIFA QUANT EDGE 1.0 | ROUND 1")
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(A4[0] - 18 * mm, A4[1] - 11.5 * mm, "RISK ACROSS TAILS AND TIMESCALES")
        canvas.line(18 * mm, 14 * mm, A4[0] - 18 * mm, 14 * mm)
        canvas.drawString(18 * mm, 9.5 * mm, "Wavelet-copula market-risk framework")
        canvas.drawRightString(A4[0] - 18 * mm, 9.5 * mm, f"Page {doc.page - 1}")
    canvas.restoreState()


def _table(data, widths, font_size=7.4):
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("LEADING", (0, 0), (-1, -1), font_size + 2.5),
                ("GRID", (0, 0), (-1, -1), 0.35, LINE),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7FA")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _callout(text: str, styles, color=PALE_GOLD):
    table = Table([[Paragraph(text, styles["Callout"])]] , colWidths=[166 * mm])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), color), ("BOX", (0, 0), (-1, -1), 0.8, BLUE), ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9)]))
    return table


def build_report(
    output_path: Path,
    figures_dir: Path,
    results: Sequence[ScaleResult],
    prices: pd.DataFrame,
    train_end: str,
    test_start: str,
    config: AnalysisConfig,
) -> None:
    styles = _styles()
    confidence_pct = 100 * config.confidence_level
    loss_pct = 100 * config.loss_probability
    tail_pct = 100 * config.tail_quantile
    train_pct = 100 * config.train_fraction
    test_pct = 100 * (1 - config.train_fraction)
    component_names = [
        f"A{config.wavelet_level}",
        *(f"D{level}" for level in range(config.wavelet_level, 0, -1)),
    ]
    component_equation = " + ".join(f"{name}(t)" for name in component_names)
    doc = BaseDocTemplate(
        str(output_path), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=20 * mm, bottomMargin=18 * mm, title="Risk Across Tails and Timescales",
        author="SAIFA Quant Edge 1.0 Team Submission",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates(PageTemplate(id="all", frames=frame, onPage=_header_footer))
    story = []

    # Cover (excluded from the competition's ten-page body limit).
    story += [Spacer(1, 38 * mm), Paragraph("SAIFA QUANT EDGE 1.0", styles["CoverKicker"]), Paragraph("Risk Across Tails<br/>and Timescales", styles["CoverTitle"]), Paragraph("A wavelet-copula framework for horizon-aware portfolio risk", styles["CoverSub"]), Spacer(1, 18 * mm)]
    story.append(_callout("Research question: Does tail dependence change with the investment horizon, and what does ignoring this do to a portfolio's measured risk?", styles, PALE_BLUE))
    story += [Spacer(1, 42 * mm), Paragraph("ROUND 1 - INITIAL SCREENING CHALLENGE", styles["CoverKicker"]), Paragraph(f"Reproducible analysis | {config.report_date}", styles["CoverSub"]), PageBreak()]

    # Page 1: executive answer.
    results = tuple(results)
    max_tail = max(result.empirical_tail for result in results)
    min_tail = min(result.empirical_tail for result in results)
    focal = max(results, key=lambda result: abs(result.es_change_pct))
    direction = "higher" if focal.es_change_pct >= 0 else "lower"
    t_gap = sum(
        abs(result.t_backtest.exception_rate - config.loss_probability)
        for result in results
    ) / len(results)
    g_gap = sum(
        abs(result.gaussian_backtest.exception_rate - config.loss_probability)
        for result in results
    ) / len(results)
    better = "wavelet-t model" if t_gap < g_gap else "Gaussian benchmark"
    story += [Paragraph("Executive answer", styles["H1x"]), _callout(f"Yes. Average empirical lower-tail co-exceedance ranges from {min_tail:.2f} to {max_tail:.2f} across wavelet horizons. At the largest risk-measurement difference ({focal.scale}), horizon-specific modelling produces {confidence_pct:g}% Expected Shortfall that is {abs(focal.es_change_pct):.1f}% {direction} than the single-dependence benchmark.", styles), Spacer(1, 8)]
    story += [Paragraph("Decision", styles["H2x"]), Paragraph(f"A risk manager should match the dependence model to the holding horizon. For exposures dominated by the {focal.scale} band, replace the fixed Gaussian dependence assumption with the fitted horizon-specific Student-t copula and set the component ES budget to {100*focal.t_risk.expected_shortfall:.3f}% of portfolio value at {confidence_pct:g}% confidence. Re-estimate after material market-regime changes.", styles["Bodyx"])]
    story += [Paragraph("What was tested", styles["H2x"]), Paragraph(f"An equal-weight portfolio of {len(config.assets)} public market proxies was studied from {prices.index.min().date()} to {prices.index.max().date()}. The first {train_pct:g}% of returns formed the estimation sample (ending {train_end}); the remaining {test_pct:g}% formed an untouched holdout sample (starting {test_start}). The mean absolute gap between the observed VaR exception rate and the nominal {loss_pct:g}% rate is {100*t_gap:.2f} percentage points for the wavelet-t model and {100*g_gap:.2f} for the benchmark; by this coverage diagnostic, the {better} is closer on average.", styles["Bodyx"])]
    story += [Paragraph("Contribution", styles["H2x"]), Paragraph("The comparison holds each wavelet band's empirical marginal distributions fixed. Only the dependence model changes: a Student-t copula estimated separately at each horizon versus one Gaussian copula estimated from raw daily returns and reused everywhere. The resulting ES difference is therefore attributable to dependence and tail assumptions rather than volatility scaling.", styles["Bodyx"]), PageBreak()]

    # Page 2: design.
    story += [Paragraph("1. Portfolio and experimental design", styles["H1x"]), Paragraph("The portfolio intentionally mixes two correlated equity indices with commodity and currency exposures. Equal weights avoid optimizing on the same sample used to test risk, and all inputs are public, daily, and reproducible without an API key.", styles["Bodyx"])]
    portfolio_data = [["Series", "Economic exposure", "Weight", "Reason for inclusion"]]
    portfolio_data.extend(
        [asset.code, asset.exposure, f"{100 * asset.weight:g}%", asset.reason]
        for asset in config.assets
    )
    story += [_table(portfolio_data, [30*mm, 42*mm, 18*mm, 76*mm]), Spacer(1, 8)]
    design_data = [["Choice", "Specification", "Rationale"], ["Return", "Daily log change", "Additive through time; comparable across level series"], ["Wavelet", f"{config.wavelet.title()} maximum-overlap/SWT, {config.wavelet_level} levels", "Additive, shift-invariant bands with transparent horizons"], ["Main dependence", "Student-t copula by band", "Allows symmetric asymptotic tail dependence"], ["Benchmark", "One Gaussian copula from raw returns", "Simple model with no asymptotic tail dependence"], ["Risk", f"{confidence_pct:g}% VaR and Expected Shortfall", "VaR is backtestable; ES measures severity beyond VaR"], ["Validation", f"Chronological {train_pct:g}/{test_pct:g} holdout", "Prevents fitting dependence to the evaluation period"]]
    story += [_table(design_data, [29*mm, 58*mm, 79*mm]), Paragraph(f"Data are aligned to a common business-day index; isolated publication gaps of at most {config.max_forward_fill_days} days are carried forward before log returns are calculated. This treats the index levels as tradable exposure proxies rather than claiming exact ETF implementation.", styles["Captionx"]), PageBreak()]

    # Page 3: method.
    story += [Paragraph("2. Method", styles["H1x"]), Paragraph("Wavelet decomposition", styles["H2x"]), Paragraph(f"For each asset return series r(t), the stationary wavelet transform produces additive components r(t) = {component_equation}. The detail bands correspond approximately to {', '.join(config.mra_scale_labels[1:])} trading-day oscillations; A{config.wavelet_level} captures variation {config.mra_scale_labels[0]}. The transform is applied separately to estimation and holdout samples.", styles["Bodyx"]), Paragraph("Copula estimation", styles["H2x"]), Paragraph(f"Within each band, ranks are converted to pseudo-observations. For each candidate degrees-of-freedom value, pseudo-observations are mapped through the Student-t quantile, a correlation matrix is estimated, and the copula log-likelihood is evaluated. Bounded maximum likelihood selects degrees of freedom between {config.t_df_bounds[0]:g} and {config.t_df_bounds[1]:g}. The benchmark maps raw-return ranks through the normal quantile once and reuses that correlation matrix at every horizon.", styles["Bodyx"]), Paragraph("Risk simulation", styles["H2x"]), Paragraph(f"Each model generates {config.simulations:,} joint uniform draws. Inverse empirical marginal distributions convert these draws back to band-specific asset returns. Portfolio losses use the configured weights. VaR is the {confidence_pct:g}th percentile of loss and ES is the mean loss conditional on exceeding VaR.", styles["Bodyx"]), Paragraph("Out-of-sample test", styles["H2x"]), Paragraph(f"Model parameters and risk thresholds are frozen after estimation. Holdout wavelet-band portfolio losses are compared with each model's VaR. We report the exception rate and quantile loss; at a {confidence_pct:g}% VaR level, ideal coverage is approximately {loss_pct:g}%. This is a distributional holdout test, not a rolling one-day trading forecast.", styles["Bodyx"]), Paragraph("Identification", styles["H2x"]), Paragraph("Because both models use the same band-specific empirical margins, changes in measured risk isolate the consequence of replacing horizon-specific heavy-tailed dependence with one Gaussian dependence structure. This controlled comparison directly answers the challenge question.", styles["Bodyx"]), PageBreak()]

    # Page 4: dependence.
    pair_count = len(config.assets) * (len(config.assets) - 1) // 2
    story += [Paragraph("3. Tail dependence changes with horizon", styles["H1x"]), Image(str(figures_dir / "tail_dependence.png"), width=166*mm, height=78*mm), Paragraph(f"Figure 1. Average across the {pair_count} asset pairs. Empirical co-exceedance is P(Uj <= {config.tail_quantile:.2f} | Ui <= {config.tail_quantile:.2f}); t-copula tail dependence is the fitted asymptotic coefficient.", styles["Captionx"])]
    tail_table = [["Horizon", "Empirical co-exceed.", "t-copula tail", "t df", "Mean correlation"]]
    for result in results:
        tail_table.append([result.scale, f"{result.empirical_tail:.3f}", f"{result.t_tail:.3f}", f"{result.t_df:.2f}", f"{result.mean_corr:.3f}"])
    story += [_table(tail_table, [34*mm, 38*mm, 32*mm, 25*mm, 37*mm]), Spacer(1, 7), Paragraph("The non-flat profile is inconsistent with the practical assumption that one dependence structure adequately represents every holding horizon. The fitted degrees of freedom also change by band, indicating that joint-tail thickness is not constant even when the asset set and sample are unchanged.", styles["Bodyx"]), PageBreak()]

    # Page 5: risk.
    story += [Paragraph("4. Effect on measured portfolio risk", styles["H1x"]), Image(str(figures_dir / "risk_comparison.png"), width=166*mm, height=81*mm), Paragraph(f"Figure 2. Simulated {confidence_pct:g}% Expected Shortfall for each additive return band. Margins are identical across models; dependence assumptions differ.", styles["Captionx"])]
    risk_table = [["Horizon", "t VaR", "t ES", "Gaussian VaR", "Gaussian ES", "ES change"]]
    for result in results:
        risk_table.append([result.scale, f"{100*result.t_risk.var:.3f}%", f"{100*result.t_risk.expected_shortfall:.3f}%", f"{100*result.gaussian_risk.var:.3f}%", f"{100*result.gaussian_risk.expected_shortfall:.3f}%", f"{result.es_change_pct:+.1f}%"])
    story += [_table(risk_table, [31*mm, 25*mm, 25*mm, 31*mm, 29*mm, 25*mm]), Spacer(1, 6), Paragraph("A positive ES change means the single Gaussian structure understates loss severity relative to the horizon-specific Student-t model; a negative value means it overstates it. The direction need not be identical at every scale, which is exactly why a single adjustment factor is insufficient.", styles["Bodyx"]), PageBreak()]

    # Page 6: holdout.
    story += [Paragraph("5. Out-of-sample comparison", styles["H1x"]), Image(str(figures_dir / "backtest.png"), width=166*mm, height=78*mm), Paragraph(f"Figure 3. Holdout VaR exception rates. The dashed line is the nominal {loss_pct:g}% rate; small holdout samples make exact equality unlikely.", styles["Captionx"])]
    back_table = [["Horizon", "N", "t exceptions", "t rate", "Gaussian exceptions", "Gaussian rate", "Better q-loss"]]
    for result in results:
        winner = "t" if result.t_backtest.quantile_loss < result.gaussian_backtest.quantile_loss else "Gaussian"
        back_table.append([result.scale, str(result.t_backtest.observations), str(result.t_backtest.exceptions), f"{100*result.t_backtest.exception_rate:.2f}%", str(result.gaussian_backtest.exceptions), f"{100*result.gaussian_backtest.exception_rate:.2f}%", winner])
    story += [_table(back_table, [29*mm, 13*mm, 23*mm, 21*mm, 32*mm, 25*mm, 23*mm], font_size=6.9), Spacer(1, 6), Paragraph("Coverage and quantile loss are complementary. Coverage asks whether the threshold is crossed at the intended frequency; quantile loss also penalizes how far observations fall on either side of the threshold. The holdout test prevents a more flexible t-copula from receiving credit merely for fitting its training sample.", styles["Bodyx"]), PageBreak()]

    # Page 7: recommendation/limits.
    story += [Paragraph("6. Recommendation, limits and robustness", styles["H1x"]), _callout(f"Action for tomorrow: map each risk position to its holding horizon and use that band's fitted t-copula ES. For the {focal.scale} component, budget {100*focal.t_risk.expected_shortfall:.3f}% of portfolio value at {confidence_pct:g}% ES rather than {100*focal.gaussian_risk.expected_shortfall:.3f}% from the fixed Gaussian model ({focal.es_change_pct:+.1f}% difference).", styles), Spacer(1, 8), Paragraph("Implementation rule", styles["H2x"]), Paragraph("Maintain the existing Gaussian result as a challenger, not as the production number. Refit horizon models on a scheduled basis and trigger an unscheduled review when realized VaR exceptions cluster. Escalate when the horizon-specific ES exceeds the Gaussian ES by the institution's materiality threshold.", styles["Bodyx"]), Paragraph("Limitations", styles["H2x"])]
    limitations = [["Limitation", "Consequence and next test"], ["Index/spot proxies", "Implementation costs, futures rolls and ETF tracking are absent; repeat with investable instruments."], ["Fixed 80/20 split", "One market regime can dominate; add rolling-window and crisis-subperiod validation."], ["Symmetric t tails", "Upside and downside dependence share one parameter; compare with Clayton, Gumbel or vine copulas."], ["Wavelet boundary effects", "Edge coefficients are affected by periodic extension; test longer samples and alternative filters."], ["Band-level risk", "Results describe additive horizon components, not a fully dynamic multi-period P&L forecast."], ["Empirical margins", "Extreme quantiles rely on finite samples; compare generalized Pareto tail models."]]
    story += [_table(limitations, [42*mm, 124*mm]), Paragraph("Robust interpretation", styles["H2x"]), Paragraph("The defensible result is the existence and economic size of horizon variation within this sample. The framework should not be read as proof that one copula family will dominate for every portfolio or future regime.", styles["Bodyx"]), PageBreak()]

    # Page 8: reproducibility, AI and references.
    source_text = "; ".join(f"{asset.name} ({asset.code})" for asset in config.assets)
    source_text = source_text.replace("&", "&amp;")
    story += [Paragraph("7. Reproducibility, AI disclosure and references", styles["H1x"]), Paragraph("Reproduction", styles["H2x"]), Paragraph("From the project root run: <font name='Courier'>uv run python run.py</font>. This one command recreates all tables, figures, machine-readable results and this PDF. The included data snapshot fixes the submitted result; <font name='Courier'>--refresh-data</font> retrieves current FRED observations. Random generation uses the seed recorded in the central configuration and model-parameter output.", styles["Bodyx"]), Paragraph("Material use of AI tools", styles["H2x"]), Paragraph("OpenAI Codex was used for research design discussion, Python implementation, debugging, report drafting and formatting. The team remains responsible for checking every method, line of code, result and statement. No result in this report is accepted solely because an AI tool produced it; all reported numbers are generated by the submitted code and public data.", styles["Bodyx"]), Paragraph("Data sources", styles["H2x"]), Paragraph(f"Federal Reserve Bank of St. Louis, FRED: {source_text}. Downloaded {config.report_date} via fred.stlouisfed.org/graph/fredgraph.csv.", styles["Smallx"]), Paragraph("References", styles["H2x"])]
    refs = ["Genest, C., Rémillard, B. & Beaudoin, D. (2009). Goodness-of-fit tests for copulas: A review and a power study. Insurance: Mathematics and Economics, 44, 199-213.", "McNeil, A. J., Frey, R. & Embrechts, P. (2015). Quantitative Risk Management, 2nd ed. Princeton University Press.", "Percival, D. B. & Walden, A. T. (2000). Wavelet Methods for Time Series Analysis. Cambridge University Press.", "Sklar, A. (1959). Fonctions de répartition à n dimensions et leurs marges. Publications de l'Institut de Statistique de l'Université de Paris, 8, 229-231.", "Basel Committee on Banking Supervision (2019). Minimum capital requirements for market risk. Bank for International Settlements."]
    for ref in refs:
        story.append(Paragraph(ref, styles["Smallx"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Code map: src/config.py defines assumptions; src/data.py coordinates market data; src/data_source.py acquires FRED series; src/data_repository.py persists snapshots; src/data_preparation.py cleans levels and calculates returns; src/wavelets.py creates horizon bands; src/copulas.py defines the common model contract and implementations; src/risk.py scores risk; src/results.py defines immutable result contracts; src/pipeline.py orchestrates outputs; src/report.py builds this document.", styles["Captionx"]))

    doc.build(story)
