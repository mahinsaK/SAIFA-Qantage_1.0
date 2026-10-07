"""Render report content as the submission PDF."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)

from .report_content import ReportContent


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
    table.setStyle(TableStyle([
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
    ]))
    return table


def _callout(text, styles, background=PALE_GOLD):
    box = Table([[Paragraph(text, styles["Callout"])]], colWidths=[166 * mm])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), background),
        ("BOX", (0, 0), (-1, -1), 0.8, BLUE),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    return KeepTogether([box])


def _escaped(text: str) -> str:
    """Escape content passed to ReportLab's XML-like paragraph parser."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_report(output_path: Path, figures_dir: Path, content: ReportContent) -> None:
    """Lay out prebuilt report content without deriving analytical claims."""
    styles = _styles()
    doc = BaseDocTemplate(
        str(output_path), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=20 * mm, bottomMargin=18 * mm, title="Risk Across Tails and Timescales",
        author="SAIFA Quant Edge 1.0 Team Submission",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates(PageTemplate(id="all", frames=frame, onPage=_header_footer))
    story = []

    story += [Spacer(1, 38 * mm), Paragraph("SAIFA QUANT EDGE 1.0", styles["CoverKicker"]), Paragraph("Risk Across Tails<br/>and Timescales", styles["CoverTitle"]), Paragraph("A wavelet-copula framework for horizon-aware portfolio risk", styles["CoverSub"]), Spacer(1, 18 * mm)]
    story.append(_callout("Research question: Does tail dependence change with the investment horizon, and what does ignoring this do to a portfolio's measured risk?", styles, PALE_BLUE))
    story += [Spacer(1, 42 * mm), Paragraph("ROUND 1 - INITIAL SCREENING CHALLENGE", styles["CoverKicker"]), Paragraph(f"Reproducible analysis | {content.report_date}", styles["CoverSub"]), PageBreak()]

    story += [Paragraph("Executive answer", styles["H1x"]), _callout(content.executive_callout, styles), Spacer(1, 8)]
    story += [Paragraph("Decision", styles["H2x"]), Paragraph(content.decision, styles["Bodyx"])]
    story += [Paragraph("What was tested", styles["H2x"]), Paragraph(content.tested, styles["Bodyx"])]
    story += [Paragraph("Contribution", styles["H2x"]), Paragraph(content.contribution, styles["Bodyx"]), PageBreak()]

    story += [Paragraph("1. Portfolio and experimental design", styles["H1x"]), Paragraph(content.portfolio_intro, styles["Bodyx"])]
    story += [_table(content.portfolio_table.matrix(), [30*mm, 42*mm, 18*mm, 76*mm]), Spacer(1, 8)]
    story += [_table(content.design_table.matrix(), [29*mm, 58*mm, 79*mm]), Paragraph(content.design_note, styles["Captionx"]), PageBreak()]

    story += [Paragraph("2. Method", styles["H1x"])]
    for section in content.method_sections:
        story += [Paragraph(section.heading, styles["H2x"]), Paragraph(section.body, styles["Bodyx"])]
    story.append(PageBreak())

    story += [Paragraph("3. Tail dependence changes with horizon", styles["H1x"]), Image(str(figures_dir / "tail_dependence.png"), width=166*mm, height=78*mm), Paragraph(_escaped(content.dependence_caption), styles["Captionx"])]
    story += [_table(content.dependence_table.matrix(), [34*mm, 38*mm, 32*mm, 25*mm, 37*mm]), Spacer(1, 7), Paragraph(content.dependence_interpretation, styles["Bodyx"]), PageBreak()]

    story += [Paragraph("4. Effect on measured portfolio risk", styles["H1x"]), Image(str(figures_dir / "risk_comparison.png"), width=166*mm, height=81*mm), Paragraph(content.risk_caption, styles["Captionx"])]
    story += [_table(content.risk_table.matrix(), [31*mm, 25*mm, 25*mm, 31*mm, 29*mm, 25*mm]), Spacer(1, 6), Paragraph(content.risk_interpretation, styles["Bodyx"]), PageBreak()]

    story += [Paragraph("5. Out-of-sample comparison", styles["H1x"]), Image(str(figures_dir / "backtest.png"), width=166*mm, height=78*mm), Paragraph(content.backtest_caption, styles["Captionx"])]
    story += [_table(content.backtest_table.matrix(), [29*mm, 13*mm, 23*mm, 21*mm, 32*mm, 25*mm, 23*mm], font_size=6.9), Spacer(1, 6), Paragraph(content.backtest_interpretation, styles["Bodyx"]), PageBreak()]

    story += [Paragraph("6. Recommendation, limits and robustness", styles["H1x"]), _callout(content.recommendation, styles), Spacer(1, 8), Paragraph("Implementation rule", styles["H2x"]), Paragraph(content.implementation_rule, styles["Bodyx"]), Paragraph("Limitations", styles["H2x"])]
    story += [_table(content.limitations_table.matrix(), [42*mm, 124*mm]), Paragraph("Robust interpretation", styles["H2x"]), Paragraph(content.robust_interpretation, styles["Bodyx"]), PageBreak()]

    reproduction = (
        "From the project root run: "
        f"<font name='Courier'>{_escaped(content.reproduction_command)}</font>. "
        f"{_escaped(content.reproduction_note)} Refresh flag: "
        f"<font name='Courier'>{_escaped(content.refresh_command)}</font>."
    )
    story += [Paragraph("7. Reproducibility, AI disclosure and references", styles["H1x"]), Paragraph("Reproduction", styles["H2x"]), Paragraph(reproduction, styles["Bodyx"]), Paragraph("Material use of AI tools", styles["H2x"]), Paragraph(content.ai_disclosure, styles["Bodyx"]), Paragraph("Data sources", styles["H2x"]), Paragraph(_escaped(content.data_sources), styles["Smallx"]), Paragraph("References", styles["H2x"])]
    for reference in content.references:
        story.append(Paragraph(_escaped(reference), styles["Smallx"]))
    story += [Spacer(1, 4), Paragraph(_escaped(content.code_map), styles["Captionx"])]

    doc.build(story)
