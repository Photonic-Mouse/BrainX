import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    HRFlowable
)


def create_probability_chart(models_data, class_names, output_chart_path):
    """Generates a high-res probability chart image for embedding in the PDF."""
    fig, ax = plt.subplots(figsize=(7, 3), dpi=200)
    
    x = np.arange(len(class_names))
    width = 0.25

    base_probs = [models_data["baseline"]["probabilities"][c] for c in class_names]
    he_probs = [models_data["he"]["probabilities"][c] for c in class_names]
    clahe_probs = [models_data["clahe"]["probabilities"][c] for c in class_names]

    r1 = ax.bar(x - width, base_probs, width, label="Baseline (Original)", color="#2563eb")
    r2 = ax.bar(x, he_probs, width, label="HE ResNet18", color="#dc2626")
    r3 = ax.bar(x + width, clahe_probs, width, label="CLAHE ResNet18", color="#16a34a")

    ax.set_ylabel("Softmax Probability", fontsize=9)
    ax.set_title("Probability Distribution Across Classes", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Format percentage annotations
    for rects in [r1, r2, r3]:
        for rect in rects:
            h = rect.get_height()
            if h > 0.05:
                ax.annotate(f"{h:.1%}",
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=6.5)

    plt.tight_layout()
    plt.savefig(output_chart_path, dpi=200, bbox_inches="tight")
    plt.close()


def generate_pdf_report(analysis_data: dict, output_pdf_path: str):
    """Generates a publication-grade, professional PDF analysis report."""
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=12
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155")
    )

    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#b45309")
    )

    story = []

    # 1. Header & Title
    story.append(Paragraph("BrainX: Multi-Model Brain MRI Analysis Report", title_style))
    story.append(Paragraph(
        "Deep Learning Decision Support & Explainability Audit Across Preprocessing Modalities",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#cbd5e1"), spaceAfter=10))

    # 2. Metadata Box
    run_dir = analysis_data.get("run_dir", "")
    metadata_data = [
        [
            Paragraph(f"<b>Run ID:</b> {analysis_data.get('run_id')}", body_style),
            Paragraph(f"<b>Timestamp:</b> {analysis_data.get('timestamp')}", body_style),
            Paragraph(f"<b>Device:</b> {analysis_data.get('device')}", body_style)
        ],
        [
            Paragraph(f"<b>File:</b> {analysis_data.get('filename')}", body_style),
            Paragraph(f"<b>Dimensions:</b> {analysis_data.get('image_dimensions')}", body_style),
            Paragraph("<b>Backbone:</b> ResNet18 (ImageNet)", body_style)
        ]
    ]
    meta_table = Table(metadata_data, colWidths=[180, 180, 180])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # 3. Model Agreement Banner
    agreement = analysis_data.get("agreement", {})
    agreement_color = "#16a34a" if agreement.get("unanimous") else ("#d97706" if agreement.get("level") == "medium" else "#dc2626")
    
    agreement_content = [
        [
            Paragraph(f"<b>MODEL CONSENSUS:</b> {agreement.get('status', 'Unknown')}", ParagraphStyle(
                "AgreeStatus", parent=body_style, fontName="Helvetica-Bold", fontSize=11, textColor=colors.HexColor(agreement_color)
            )),
        ],
        [
            Paragraph(agreement.get("summary", ""), body_style)
        ]
    ]
    agree_table = Table(agreement_content, colWidths=[540])
    agree_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor(agreement_color)),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(agree_table)
    story.append(Spacer(1, 10))

    # 4. Predictions Table
    story.append(Paragraph("1. Model Predictions & Probability Breakdown", section_heading))
    
    models = analysis_data["models"]
    class_names = analysis_data["class_names"]

    pred_table_data = [
        ["Model Architecture", "Preprocessing", "Predicted Class", "Confidence", "Benchmark Acc", "Benchmark Macro F1"]
    ]
    for key in ["baseline", "he", "clahe"]:
        m = models[key]
        pred_table_data.append([
            m["name"],
            m["preprocessing"],
            m["predicted_class"],
            f"{m['confidence_pct']}%",
            m["benchmark_acc"],
            m["benchmark_macro_f1"]
        ])

    pred_table = Table(pred_table_data, colWidths=[130, 80, 110, 70, 75, 75])
    pred_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ALIGN", (3, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(pred_table)
    story.append(Spacer(1, 10))

    # 5. Probability Chart
    chart_path = os.path.join(run_dir, "prob_chart.png")
    create_probability_chart(models, class_names, chart_path)
    if os.path.exists(chart_path):
        story.append(RLImage(chart_path, width=540, height=210))
    story.append(Spacer(1, 12))

    # 6. Preprocessing Comparison Section
    story.append(Paragraph("2. MRI Preprocessing Comparison (Raw vs HE vs CLAHE)", section_heading))
    raw_img = os.path.join(run_dir, "original_raw.png")
    he_img = os.path.join(run_dir, "prep_he.png")
    clahe_img = os.path.join(run_dir, "prep_clahe.png")

    if os.path.exists(raw_img) and os.path.exists(he_img) and os.path.exists(clahe_img):
        prep_table_data = [
            [
                Paragraph("<b>Original MRI</b>", body_style),
                Paragraph("<b>Histogram Equalization</b>", body_style),
                Paragraph("<b>CLAHE</b>", body_style)
            ],
            [
                RLImage(raw_img, width=150, height=150),
                RLImage(he_img, width=150, height=150),
                RLImage(clahe_img, width=150, height=150)
            ],
            [
                Paragraph("Unenhanced grayscale axial MRI", disclaimer_style),
                Paragraph("Global intensity stretching", disclaimer_style),
                Paragraph("Adaptive local contrast (8x8 tiles)", disclaimer_style)
            ]
        ]
        prep_table = Table(prep_table_data, colWidths=[180, 180, 180])
        prep_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        story.append(prep_table)
    story.append(Spacer(1, 12))

    # 7. Grad-CAM Visualizations Section
    story.append(Paragraph("3. Explainability: Dynamic Grad-CAM Visualizations", section_heading))
    story.append(Paragraph(
        "Attribution heatmaps are extracted from ResNet18 Layer 4. Red/warm regions indicate brain structures positively driving the model's classification score.",
        body_style
    ))
    story.append(Spacer(1, 6))

    gradcam_table_data = [
        ["Model", "Preprocessed Image", "Grad-CAM Heatmap", "Heatmap Overlay", "Predicted Outcome"]
    ]
    for key in ["baseline", "he", "clahe"]:
        m = models[key]
        prep_f = os.path.join(run_dir, f"prep_{key}.png") if key != "baseline" else raw_img
        heat_f = os.path.join(run_dir, f"heatmap_{key}.png")
        over_f = os.path.join(run_dir, f"overlay_{key}.png")
        
        gradcam_table_data.append([
            Paragraph(f"<b>{m['name']}</b>", body_style),
            RLImage(prep_f, width=90, height=90) if os.path.exists(prep_f) else "N/A",
            RLImage(heat_f, width=90, height=90) if os.path.exists(heat_f) else "N/A",
            RLImage(over_f, width=90, height=90) if os.path.exists(over_f) else "N/A",
            Paragraph(f"<b>{m['predicted_class']}</b><br/>Confidence: {m['confidence_pct']}%", body_style)
        ])

    gradcam_table = Table(gradcam_table_data, colWidths=[110, 100, 100, 100, 130])
    gradcam_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(gradcam_table)
    story.append(Spacer(1, 14))

    # 8. Technical Discussion & Saliency Observations
    story.append(Paragraph("4. Technical Discussion & Saliency Interpretation", section_heading))
    discussion_text = (
        "<b>Neuroanatomical Correlation:</b> In Baseline and CLAHE ResNet18 models, high saliency gradients typically center "
        "on the lateral ventricles, third ventricle, and medial temporal lobe structures. Ventricular enlargement and hippocampal "
        "atrophy represent principal imaging hallmarks of neurodegeneration. Conversely, the global Histogram Equalization (HE) "
        "model frequently displays diffused attention toward outer skull contours and peripheral noise due to non-linear intensity "
        "over-saturation, directly explaining its lower test benchmark generalization (78.23% vs 95.10%)."
    )
    story.append(Paragraph(discussion_text, body_style))
    story.append(Spacer(1, 12))

    # 9. Academic & Clinical Safety Disclaimer
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0"), spaceAfter=8))
    disclaimer_box = [
        [
            Paragraph(
                "<b>RESEARCH SAFETY & REGULATORY DISCLAIMER:</b><br/>"
                "This analysis report is automatically generated by an experimental research platform (BrainX) for academic, "
                "scientific, and benchmarking purposes. It is <b>NOT</b> a medical device and does <b>NOT</b> provide a clinically "
                "certified diagnosis of Alzheimer's Disease or Mild Cognitive Impairment. Softmax probability percentages indicate "
                "internal algorithmic alignment to training feature distributions and must not be used as clinical diagnostic certainty. "
                "All patient care decisions require comprehensive neuropsychological testing, biomarker assessment, and expert radiological review.",
                disclaimer_style
            )
        ]
    ]
    disc_table = Table(disclaimer_box, colWidths=[540])
    disc_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fffbeb")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#f59e0b")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(disc_table)

    # Build PDF
    doc.build(story)
    return output_pdf_path
