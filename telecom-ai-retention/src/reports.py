"""
Executive PDF report generation.

Takes the output of `src/batch.py` (a scored customer dataframe + its
summary dict) and produces a branded, print-ready PDF: executive
summary, business KPIs, revenue at risk, a feature-importance chart,
a high-risk customer table, and grounded retention recommendations.

Deliberately does NOT compute per-customer SHAP for every row in the
report -- that's expensive at batch scale (hundreds/thousands of
customers) and belongs to the interactive Explain Prediction page for
drilling into one customer at a time. This report works at the
aggregate level: global feature importance plus the same
risk-driver-to-action rulebook used there, applied to the batch's
dominant risk factors rather than one customer's.
"""

import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from src.recommendations import RETENTION_RULEBOOK, humanize_feature

BRAND_NAME = "RetainAI"
BRAND_TAGLINE = "Customer Retention Intelligence Platform"
BRAND_COLOR = colors.HexColor("#1d4ed8")
BRAND_COLOR_LIGHT = colors.HexColor("#eff6ff")
DANGER_COLOR = colors.HexColor("#dc2626")
MUTED_COLOR = colors.HexColor("#6b7280")


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "BrandTitle", parent=styles["Title"], textColor=BRAND_COLOR, fontSize=22, spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        "BrandTagline", parent=styles["Normal"], textColor=MUTED_COLOR, fontSize=10, spaceAfter=18,
    ))
    styles.add(ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"], textColor=BRAND_COLOR, spaceBefore=16, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        "Body", parent=styles["Normal"], fontSize=10.5, leading=15,
    ))
    return styles


def _kpi_table(summary: dict) -> Table:
    rows = [
        ["Total customers scored", f"{summary['total_customers']:,}"],
        ["Predicted churners", f"{summary['predicted_churners']:,} "
                                 f"({summary['predicted_churn_rate']:.1%})"],
        ["Average churn probability", f"{summary['avg_churn_probability']:.1%}"],
    ]
    if summary.get("monthly_revenue_at_risk") is not None:
        rows.append(["Monthly revenue at risk", f"${summary['monthly_revenue_at_risk']:,.0f}"])
    rows.append(["Urgent priority customers", f"{summary['urgent_count']:,}"])
    rows.append(["High priority customers", f"{summary['high_count']:,}"])

    table = Table(rows, colWidths=[3.2 * inch, 2.3 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), BRAND_COLOR_LIGHT),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return table


def _feature_importance_chart(model, feature_names, top_n: int = 12) -> io.BytesIO:
    if not hasattr(model, "feature_importances_"):
        return None
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values().tail(top_n)

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.barh(importances.index.map(humanize_feature), importances.values, color="#1d4ed8")
    ax.set_xlabel("Importance", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def _high_risk_table(scored: pd.DataFrame, top_n: int = 20) -> Table:
    id_col = "customerID" if "customerID" in scored.columns else None
    display_cols = [c for c in [id_col, "Contract", "MonthlyCharges", "tenure"] if c and c in scored.columns]

    top = scored.sort_values("ChurnProbability", ascending=False).head(top_n)

    header = (["Customer"] if id_col else []) + [c for c in display_cols if c != id_col] + \
             ["Churn Prob.", "Priority"]
    rows = [header]
    for _, row in top.iterrows():
        line = []
        if id_col:
            line.append(str(row[id_col]))
        for c in display_cols:
            if c == id_col:
                continue
            val = row[c]
            line.append(f"{val:.0f}" if isinstance(val, float) else str(val))
        line.append(f"{row['ChurnProbability']:.1%}")
        line.append(row["Priority"])
        rows.append(line)

    table = Table(rows, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_COLOR),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    priority_col = len(header) - 1
    for i, row in enumerate(rows[1:], start=1):
        if row[priority_col] == "Urgent":
            style.append(("TEXTCOLOR", (priority_col, i), (priority_col, i), DANGER_COLOR))
            style.append(("FONTNAME", (priority_col, i), (priority_col, i), "Helvetica-Bold"))
    table.setStyle(TableStyle(style))
    return table


def _top_recommendations(scored: pd.DataFrame, max_recommendations: int = 4) -> list:
    """Aggregate-level recommendations for the batch, using the same
    rulebook as per-customer recommendations (src/recommendations.py)
    but driven by which risk factors are most common across the
    batch's high-priority customers, not one customer's SHAP values."""
    high_risk = scored[scored["Priority"].isin(["Urgent", "High"])]
    if high_risk.empty:
        return ["No urgent or high-priority customers in this batch — no immediate action needed."]

    signals = []
    if "Contract" in high_risk.columns:
        mtm_share = (high_risk["Contract"] == "Month-to-month").mean()
        if mtm_share >= 0.4:
            signals.append("ContractRiskLevel")
    if "PaymentMethod" in high_risk.columns:
        echeck_share = (high_risk["PaymentMethod"] == "Electronic check").mean()
        if echeck_share >= 0.3:
            signals.append("AutoPaymentUser")
    if "InternetService" in high_risk.columns:
        fiber_share = (high_risk["InternetService"] == "Fiber optic").mean()
        if fiber_share >= 0.4:
            signals.append("PremiumInternetUser")
    if "tenure" in high_risk.columns:
        low_tenure_share = (high_risk["tenure"] <= 12).mean()
        if low_tenure_share >= 0.3:
            signals.append("LowTenureFlag")

    actions = []
    for key in signals[:max_recommendations]:
        rule = RETENTION_RULEBOOK.get(key)
        if rule:
            actions.append(f"{rule['reason']} — {rule['actions'][0]}")

    if not actions:
        actions.append(
            "High-risk customers in this batch don't share one dominant driver — "
            "review individually via the Explain Prediction page for per-customer SHAP drivers."
        )
    return actions


def generate_pdf_report(
    scored: pd.DataFrame,
    summary: dict,
    model=None,
    feature_names=None,
    report_title: str = "Executive Retention Report",
) -> bytes:
    """Build the full PDF report and return it as bytes, ready to hand
    to a Streamlit download button or write to disk."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
    )
    styles = _styles()
    story = []

    story.append(Paragraph(BRAND_NAME, styles["BrandTitle"]))
    story.append(Paragraph(BRAND_TAGLINE, styles["BrandTagline"]))
    story.append(Paragraph(report_title, styles["Heading1"]))
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%B %d, %Y at %H:%M')} — "
        f"{summary['total_customers']:,} customers scored",
        styles["Body"],
    ))
    story.append(Spacer(1, 16))

    story.append(Paragraph("Executive summary", styles["SectionHeading"]))
    revenue_line = (
        f" representing approximately ${summary['monthly_revenue_at_risk']:,.0f} "
        f"in monthly recurring revenue at risk"
        if summary.get("monthly_revenue_at_risk") is not None else ""
    )
    story.append(Paragraph(
        f"Of {summary['total_customers']:,} customers scored, "
        f"{summary['predicted_churners']:,} ({summary['predicted_churn_rate']:.1%}) are "
        f"predicted to churn{revenue_line}. {summary['urgent_count']:,} customers are flagged "
        f"as urgent priority and should be prioritized for retention outreach; "
        f"{summary['high_count']:,} more are high priority.",
        styles["Body"],
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Business KPIs", styles["SectionHeading"]))
    story.append(_kpi_table(summary))

    if model is not None and feature_names is not None:
        chart_buf = _feature_importance_chart(model, feature_names)
        if chart_buf is not None:
            story.append(Paragraph("What's driving churn risk in this batch", styles["SectionHeading"]))
            story.append(Image(chart_buf, width=6.2 * inch, height=3.3 * inch))
            story.append(Paragraph(
                "Global feature importance from the production model. For why any single "
                "customer is flagged, see the Explain Prediction page's per-customer SHAP breakdown.",
                styles["Body"],
            ))

    story.append(PageBreak())
    story.append(Paragraph("Recommended actions", styles["SectionHeading"]))
    for rec in _top_recommendations(scored):
        story.append(Paragraph(f"• {rec}", styles["Body"]))
    story.append(Spacer(1, 14))

    story.append(Paragraph("Highest-risk customers", styles["SectionHeading"]))
    story.append(_high_risk_table(scored))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
