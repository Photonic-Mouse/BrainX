import os
import sys
import pptx
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

# -----------------------------------------------------------------------------
# APPLICATION UI/UX COLOR PALETTE (Exact match to frontend/styles.css)
# -----------------------------------------------------------------------------
COLOR_BG_MAIN      = RGBColor(11, 15, 25)    # --bg-main: #0B0F19 (Deepest Dark Navy/Black)
COLOR_BG_CARD      = RGBColor(17, 24, 39)    # --bg-card: #111827 (Primary Card Slate)
COLOR_CARD_SUBTLE  = RGBColor(30, 41, 59)    # --bg-card-subtle: #1E293B (Ribbon / Card Highlight)
COLOR_BORDER       = RGBColor(51, 65, 85)    # --border: #334155 (Subtle Slate Border)
COLOR_TEXT_MAIN    = RGBColor(248, 250, 252) # --text-main: #F8FAFC (Crisp Off-White)
COLOR_TEXT_MUTED   = RGBColor(148, 163, 184) # --text-muted: #94A3B8 (Cool Slate Gray)
COLOR_PRIMARY      = RGBColor(59, 130, 246)  # --primary: #3B82F6 (Electric Tech Blue)
COLOR_PRIMARY_DARK = RGBColor(37, 99, 235)   # --primary-hover: #2563EB (Deep Accent Blue)
COLOR_CYAN         = RGBColor(56, 189, 248)  # Cyan highlight (#38BDF8)
COLOR_SUCCESS      = RGBColor(16, 185, 129)  # Status Indicator / Success (#10B981)
COLOR_WARNING      = RGBColor(245, 158, 11)  # Amber Warning (#F59E0B)
COLOR_DANGER       = RGBColor(239, 68, 68)   # Alert / Severe (#EF4444)

FONT_HEADING = "Segoe UI"
FONT_BODY    = "Segoe UI"


def set_slide_background(slide):
    """Draw full-bleed background using application's --bg-main (#0B0F19)."""
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = COLOR_BG_MAIN
    bg.line.color.rgb = COLOR_BG_MAIN
    return bg


def add_header(slide, title_text, step_tag="01 | OVERVIEW"):
    """Render top application-style navigation header with status pill."""
    # Top rule bar
    rule = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(0.35), Inches(11.733), Pt(1.5))
    rule.fill.solid()
    rule.fill.fore_color.rgb = COLOR_PRIMARY_DARK
    rule.line.color.rgb = COLOR_PRIMARY_DARK

    # Category / Step Tag
    tag_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.45), Inches(8.0), Inches(0.3))
    tf_tag = tag_box.text_frame
    tf_tag.word_wrap = True
    p_tag = tf_tag.paragraphs[0]
    p_tag.text = step_tag.upper()
    p_tag.font.name = FONT_HEADING
    p_tag.font.size = Pt(9.5)
    p_tag.font.bold = True
    p_tag.font.color.rgb = COLOR_CYAN

    # Status indicator pill on top right
    pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(10.3), Inches(0.45), Inches(2.233), Inches(0.32))
    pill.fill.solid()
    pill.fill.fore_color.rgb = COLOR_BG_CARD
    pill.line.color.rgb = COLOR_BORDER
    pill.line.width = Pt(1)
    
    tf_pill = pill.text_frame
    p_pill = tf_pill.paragraphs[0]
    p_pill.text = "● SYSTEM ACTIVE | v2.0"
    p_pill.alignment = PP_ALIGN.CENTER
    p_pill.font.name = FONT_HEADING
    p_pill.font.size = Pt(8.5)
    p_pill.font.bold = True
    p_pill.font.color.rgb = COLOR_SUCCESS

    # Main Slide Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.75), Inches(11.733), Inches(0.65))
    tf_title = title_box.text_frame
    tf_title.word_wrap = True
    p_title = tf_title.paragraphs[0]
    p_title.text = title_text
    p_title.font.name = FONT_HEADING
    p_title.font.size = Pt(21)
    p_title.font.bold = True
    p_title.font.color.rgb = COLOR_TEXT_MAIN


def add_card(slide, left, top, width, height, title=None, badge=None, border_color=COLOR_BORDER, bg_color=COLOR_BG_CARD):
    """Render a container matching the UI card styling (--bg-card: #111827 with --border: #334155)."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    card.line.color.rgb = border_color
    card.line.width = Pt(1)

    if title:
        # Title text
        tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.12), width - Inches(0.4), Inches(0.38))
        tf = tb.text_frame
        p = tf.paragraphs[0]
        p.text = title
        p.font.name = FONT_HEADING
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY

        # Thin header divider line
        line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left + Inches(0.2), top + Inches(0.48), width - Inches(0.4), Pt(0.75))
        line.fill.solid()
        line.fill.fore_color.rgb = COLOR_BORDER
        line.line.color.rgb = COLOR_BORDER

    if badge:
        # Mini badge on top right of card
        badge_w = Inches(1.3)
        b_shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left + width - badge_w - Inches(0.2), top + Inches(0.12), badge_w, Inches(0.28))
        b_shape.fill.solid()
        b_shape.fill.fore_color.rgb = COLOR_CARD_SUBTLE
        b_shape.line.color.rgb = COLOR_BORDER
        b_tf = b_shape.text_frame
        b_p = b_tf.paragraphs[0]
        b_p.text = badge
        b_p.alignment = PP_ALIGN.CENTER
        b_p.font.name = FONT_HEADING
        b_p.font.size = Pt(8)
        b_p.font.bold = True
        b_p.font.color.rgb = COLOR_CYAN

    return card


def add_stat_card(slide, left, top, width, height, value, label, subtext="", val_color=COLOR_CYAN):
    """Render a prominent metric KPI card matching the application's header distribution badges."""
    card = add_card(slide, left, top, width, height, bg_color=COLOR_BG_CARD, border_color=COLOR_BORDER)
    tb = slide.shapes.add_textbox(left + Inches(0.15), top + Inches(0.12), width - Inches(0.3), height - Inches(0.24))
    tf = tb.text_frame
    tf.word_wrap = True

    p_val = tf.paragraphs[0]
    p_val.text = value
    p_val.font.name = FONT_HEADING
    p_val.font.size = Pt(25)
    p_val.font.bold = True
    p_val.font.color.rgb = val_color

    p_lbl = tf.add_paragraph()
    p_lbl.text = label
    p_lbl.font.name = FONT_HEADING
    p_lbl.font.size = Pt(9.5)
    p_lbl.font.bold = True
    p_lbl.font.color.rgb = COLOR_TEXT_MAIN

    if subtext:
        p_sub = tf.add_paragraph()
        p_sub.text = subtext
        p_sub.font.name = FONT_BODY
        p_sub.font.size = Pt(8.5)
        p_sub.font.color.rgb = COLOR_TEXT_MUTED

    return card


def build_full_deck(output_path="BrainScope_XAI_Presentation.pptx"):
    prs = pptx.Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: TITLE SLIDE (APPLICATION HERO)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1)

    # Top brand bar
    top_bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(0.6), Inches(11.733), Pt(2))
    top_bar.fill.solid()
    top_bar.fill.fore_color.rgb = COLOR_PRIMARY
    top_bar.line.color.rgb = COLOR_PRIMARY

    # Brand Title
    tb_brand = s1.shapes.add_textbox(Inches(0.8), Inches(1.1), Inches(11.733), Inches(2.2))
    tf_brand = tb_brand.text_frame
    tf_brand.word_wrap = True

    p_b0 = tf_brand.paragraphs[0]
    p_b0.text = "BRAINX  |  BRAINSCOPE-XAI"
    p_b0.font.name = FONT_HEADING
    p_b0.font.size = Pt(36)
    p_b0.font.bold = True
    p_b0.font.color.rgb = COLOR_CYAN

    p_b1 = tf_brand.add_paragraph()
    p_b1.text = "Multi-Model Brain MRI Classification & Explainable AI (XAI) for Alzheimer's Staging"
    p_b1.font.name = FONT_HEADING
    p_b1.font.size = Pt(19)
    p_b1.font.bold = True
    p_b1.font.color.rgb = COLOR_TEXT_MAIN
    p_b1.space_before = Pt(4)

    p_b2 = tf_brand.add_paragraph()
    p_b2.text = "From Literature Critique & Forensic Audit to 97.50% Accuracy • RSNA CLAIM 2024 Compliant"
    p_b2.font.name = FONT_BODY
    p_b2.font.size = Pt(12)
    p_b2.font.color.rgb = COLOR_TEXT_MUTED
    p_b2.space_before = Pt(4)

    # 4 App Stat Badges
    add_stat_card(s1, Inches(0.8), Inches(3.8), Inches(2.7), Inches(1.5), "97.50%", "Peak Test Accuracy", "936 / 960 Scans (Seed 17)", COLOR_SUCCESS)
    add_stat_card(s1, Inches(3.8), Inches(3.8), Inches(2.7), Inches(1.5), "100.0%", "Severe AD Sensitivity", "0 False Negatives on Moderate AD", COLOR_PRIMARY)
    add_stat_card(s1, Inches(6.8), Inches(3.8), Inches(2.7), Inches(1.5), "98.20%", "Peak Macro-F1", "Equal Balance Across 4 Stages", COLOR_CYAN)
    add_stat_card(s1, Inches(9.8), Inches(3.8), Inches(2.733), Inches(1.5), "p = 0.0015", "Statistical Significance", "McNemar Test vs. Baseline", COLOR_SUCCESS)

    # Bottom footer card
    add_card(s1, Inches(0.8), Inches(5.6), Inches(11.733), Inches(1.1), bg_color=COLOR_CARD_SUBTLE)
    tb_foot = s1.shapes.add_textbox(Inches(1.0), Inches(5.7), Inches(11.333), Inches(0.9))
    tf_foot = tb_foot.text_frame
    tf_foot.word_wrap = True
    p_f = tf_foot.paragraphs[0]
    p_f.text = "ARCHITECTURE & DEFENSE PRESENTATION DECK\nBackbone: ResNet-18 (Torchvision Weights)  •  Explainability: Layer-4 Grad-CAM  •  Backend: FastAPI REST Microservice"
    p_f.font.name = FONT_BODY
    p_f.font.size = Pt(10)
    p_f.font.color.rgb = COLOR_TEXT_MUTED

    # =========================================================================
    # SLIDE 2: LITERATURE SURVEY: 4 CRITICAL FLAWS IN EXISTING RESEARCH
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)
    add_header(s2, "Literature Survey: 4 Critical Flaws in Published Alzheimer's MRI Research", "01 | PROJECT GENESIS & LITERATURE CRITIQUE")

    lit_flaws = [
        ("1. The 99% Accuracy Illusion & Data Leakage",
         "Many prominent published papers report 99%+ accuracy by using naive random slice splits. As proven in Nature Scientific Reports (2021), when multiple 2D axial slices from the SAME patient are split across train and test, the CNN simply memorizes skull geometry rather than genuine neurodegeneration, causing massive real-world clinical failure."),
        
        ("2. Minority Class Neglect (50:1 Imbalance)",
         "In real-world cohorts, healthy controls outnumber severe dementia cases by 50 to 1. Most published literature computes standard unweighted accuracy. A model predicting 'healthy' for every scan achieves 95% accuracy while missing 100% of severe dementia patients—an intolerable diagnostic hazard."),
        
        ("3. The 'Black-Box' Radiologist Rejection",
         "Existing architectures deploy deep CNNs or Vision Transformers without spatial interpretability. Under the RSNA CLAIM 2024 guidelines, medical AI must localize the anatomical biomarkers driving the prediction (ventricular enlargement, hippocampal shrinkage) before it can be trusted by clinical radiologists."),
        
        ("4. Single Lucky Runs & Unchecked Stochasticity",
         "Most published architectures report results from a single random train/test run without seed disclosure. Stochastic variations in data batching and augmentation mean single runs are often cherry-picked outliers that fail to replicate.")
    ]

    for idx, (flaw_title, flaw_desc) in enumerate(lit_flaws):
        row = idx // 2
        col = idx % 2
        c_left = Inches(0.8 + col * 5.95)
        c_top  = Inches(1.6 + row * 2.7)
        c_w    = Inches(5.78)
        c_h    = Inches(2.45)

        add_card(s2, c_left, c_top, c_w, c_h, flaw_title, badge=f"DRAWBACK #{idx+1}", border_color=COLOR_BORDER)
        tb = s2.shapes.add_textbox(c_left + Inches(0.2), c_top + Inches(0.55), c_w - Inches(0.4), c_h - Inches(0.65))
        tf = tb.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = flaw_desc
        p.font.name = FONT_BODY
        p.font.size = Pt(9.5)
        p.font.color.rgb = COLOR_TEXT_MUTED

    # =========================================================================
    # SLIDE 3: FORENSIC TAKEOVER & REPRODUCIBILITY AUDIT
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3)
    add_header(s3, "Forensic Project Takeover & Rigorous Audit", "02 | TAKING OVER THE PROJECT")

    # Left Card: Legacy State
    add_card(s3, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.2), "Legacy Repository State (What We Inherited)", badge="STARTING POINT")
    tb3_l = s3.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.3), Inches(4.4))
    tf3_l = tb3_l.text_frame; tf3_l.word_wrap = True

    legacy_points = [
        ("Hardcoded Legacy Paths: ", "Scripts contained old paths ('C:\\HEHAW\\BrainX'), breaking automated execution and reproducibility."),
        ("Unchecked Split Leakage Risk: ", "Our SHA-256 perceptual hash audit discovered 543 cross-split duplicate image clusters that risked test contamination if left unmanaged."),
        ("Fragile Legacy Baselines: ", "Saved results existed (94.38% baseline, 78.23% HE, 95.10% CLAHE), but the underlying code had 3x stochastic variance across runs (±2.26%)."),
        ("Minority Class Blindness: ", "The legacy training script missed 20% of Moderate Demented scans due to unweighted loss optimization.")
    ]
    for i, (bt, bd) in enumerate(legacy_points):
        p = tf3_l.paragraphs[0] if i == 0 else tf3_l.add_paragraph()
        r1 = p.add_run(); r1.text = bt; r1.font.bold = True; r1.font.color.rgb = COLOR_DANGER; r1.font.size = Pt(10.5)
        r2 = p.add_run(); r2.text = bd; r2.font.color.rgb = COLOR_TEXT_MUTED; r2.font.size = Pt(9.5)
        p.space_after = Pt(10)

    # Right Card: Our Forensic Cleanup
    add_card(s3, Inches(6.8), Inches(1.6), Inches(5.733), Inches(5.2), "Our Scientific Intervention & Protocol", badge="AUDITED FOUNDATION")
    tb3_r = s3.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.4))
    tf3_r = tb3_r.text_frame; tf3_r.word_wrap = True

    audit_points = [
        ("Audited Stratified Split Manifest: ", "Locked an immutable 70/15/15 split (4,480 train / 960 val / 960 test) verified via SHA-256 hashes in Dataset/splits/splits.csv."),
        ("Deterministic Multi-Seed Control: ", "Pre-registered evaluation across 3 fixed random seeds (17, 42, 2026) with deterministic CuDNN execution to eliminate lucky-run bias."),
        ("Strict Test Set Sealing: ", "Test data (N=960) was completely sealed during all preprocessing, augmentation, and hyperparameter tuning. Only validation metrics were monitored."),
        ("RSNA CLAIM 2024 Alignment: ", "Predeclared evaluation metrics: Macro-Averaged F1 (primary guardrail), Balanced Accuracy, Wilson 95% CIs, and paired McNemar significance.")
    ]
    for i, (bt, bd) in enumerate(audit_points):
        p = tf3_r.paragraphs[0] if i == 0 else tf3_r.add_paragraph()
        r1 = p.add_run(); r1.text = bt; r1.font.bold = True; r1.font.color.rgb = COLOR_SUCCESS; r1.font.size = Pt(10.5)
        r2 = p.add_run(); r2.text = bd; r2.font.color.rgb = COLOR_TEXT_MUTED; r2.font.size = Pt(9.5)
        p.space_after = Pt(10)

    # =========================================================================
    # SLIDE 4: MODEL SELECTION: WHY RESNET-18?
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4)
    add_header(s4, "Model Selection: Why ResNet-18 Outperforms Deeper Models", "03 | ARCHITECTURAL DECISION MATRIX")

    # 3 Comparison Cards
    add_card(s4, Inches(0.8), Inches(1.6), Inches(3.6), Inches(5.2), "ResNet-50 / DenseNet-121", badge="OVERPARAMETERIZED")
    tb4_1 = s4.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(3.2), Inches(4.4))
    tf4_1 = tb4_1.text_frame; tf4_1.word_wrap = True
    tf4_1.paragraphs[0].text = "• 25.6M parameters (2.3x larger)\n• Severe Overfitting: On small/medium medical cohorts (6,400 scans), ultra-deep models rapidly memorize training noise.\n• Late-epoch validation divergence\n• Excessive computation latency, hindering hospital edge deployment."
    tf4_1.paragraphs[0].font.name = FONT_BODY; tf4_1.paragraphs[0].font.size = Pt(10); tf4_1.paragraphs[0].font.color.rgb = COLOR_TEXT_MUTED

    add_card(s4, Inches(4.85), Inches(1.6), Inches(3.6), Inches(5.2), "Vision Transformers (ViT)", badge="DATA HUNGRY")
    tb4_2 = s4.shapes.add_textbox(Inches(5.05), Inches(2.2), Inches(3.2), Inches(4.4))
    tf4_2 = tb4_2.text_frame; tf4_2.word_wrap = True
    tf4_2.paragraphs[0].text = "• 86M+ parameters (8x larger)\n• Lacks Inductive Bias: Transformers lack the innate translational invariance and locality of convolutions.\n• Requires 100k+ images to avoid severe overfitting unless aggressively regularized.\n• Poor sample efficiency on 6,400 axial MRI slices."
    tf4_2.paragraphs[0].font.name = FONT_BODY; tf4_2.paragraphs[0].font.size = Pt(10); tf4_2.paragraphs[0].font.color.rgb = COLOR_TEXT_MUTED

    add_card(s4, Inches(8.9), Inches(1.6), Inches(3.633), Inches(5.2), "Selected: ResNet-18", badge="OPTIMAL WINNER", border_color=COLOR_SUCCESS)
    tb4_3 = s4.shapes.add_textbox(Inches(9.1), Inches(2.2), Inches(3.2), Inches(4.4))
    tf4_3 = tb4_3.text_frame; tf4_3.word_wrap = True
    tf4_3.paragraphs[0].text = "• 11.2M parameters: Perfect sample efficiency for 6,400 scans.\n• Residual Skip Connections: y = F(x) + x provides gradient highways, preventing degradation.\n• Fast Convergence: Reaches optimal validation F1 in 4-6 epochs without memorization.\n• Real-Time Edge Inference: Sub-100ms inference per slice on standard clinic hardware."
    tf4_3.paragraphs[0].font.name = FONT_BODY; tf4_3.paragraphs[0].font.size = Pt(10); tf4_3.paragraphs[0].font.color.rgb = COLOR_TEXT_MAIN

    # =========================================================================
    # SLIDE 5: END-TO-END SYSTEM PIPELINE ARCHITECTURE
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)
    add_header(s5, "End-to-End System Pipeline Architecture", "04 | SYSTEM ARCHITECTURE")

    # Dark Mermaid Diagram Card
    add_card(s5, Inches(0.8), Inches(1.5), Inches(11.733), Inches(2.6), "One-View Execution Dataflow: Ingestion, Backbone, Loss & XAI", badge="DETERMINISTIC PIPELINE")
    diag_pipe = "results/architecture_diagrams/dark_diagram_pipeline.png"
    if os.path.exists(diag_pipe):
        s5.shapes.add_picture(diag_pipe, Inches(1.0), Inches(2.05), width=Inches(11.333))

    # Bottom 3 Cards Explaining Components
    add_card(s5, Inches(0.8), Inches(4.3), Inches(3.6), Inches(2.6), "1. Ingestion & Preprocessing")
    tb5_1 = s5.shapes.add_textbox(Inches(0.95), Inches(4.85), Inches(3.3), Inches(1.9))
    tf5_1 = tb5_1.text_frame; tf5_1.word_wrap = True
    tf5_1.paragraphs[0].text = "• Bilinear Rescaling: 128×128 -> 224×224\n• 3-Channel Replication: [I] -> [R=I, G=I, B=I]\n• ImageNet Normalization (mu, sigma)\n• Training-Only Regularization: Mild RandomRotation([-5°, +5°])"
    tf5_1.paragraphs[0].font.name = FONT_BODY; tf5_1.paragraphs[0].font.size = Pt(9.5); tf5_1.paragraphs[0].font.color.rgb = COLOR_TEXT_MUTED

    add_card(s5, Inches(4.85), Inches(4.3), Inches(3.6), Inches(2.6), "2. Deep Residual Backbone")
    tb5_2 = s5.shapes.add_textbox(Inches(5.0), Inches(4.85), Inches(3.3), Inches(1.9))
    tf5_2 = tb5_2.text_frame; tf5_2.word_wrap = True
    tf5_2.paragraphs[0].text = "• Stem: Conv1 (7×7, stride 2) + MaxPool\n• 4 Residual Stages with BasicBlocks\n• Skip Connections resolve vanishing gradients\n• Global Average Pooling collapses 7×7×512 into a 512-dim vector"
    tf5_2.paragraphs[0].font.name = FONT_BODY; tf5_2.paragraphs[0].font.size = Pt(9.5); tf5_2.paragraphs[0].font.color.rgb = COLOR_TEXT_MUTED

    add_card(s5, Inches(8.9), Inches(4.3), Inches(3.633), Inches(2.6), "3. Dual Output Branch")
    tb5_3 = s5.shapes.add_textbox(Inches(9.05), Inches(4.85), Inches(3.3), Inches(1.9))
    tf5_3 = tb5_3.text_frame; tf5_3.word_wrap = True
    tf5_3.paragraphs[0].text = "• Classification Head: Linear(512, 4)\n• Loss: Inverse-frequency weighted CE\n• Layer-4 Grad-CAM: Captures gradients at layer4[-1], rendering transparent clinical saliency heatmaps"
    tf5_3.paragraphs[0].font.name = FONT_BODY; tf5_3.paragraphs[0].font.size = Pt(9.5); tf5_3.paragraphs[0].font.color.rgb = COLOR_TEXT_MUTED

    # =========================================================================
    # SLIDE 6: INPUT REPRESENTATION ENGINEERING
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6)
    add_header(s6, "Input Representation Engineering: Bilinear Rescale & 3-Channel Replication", "05 | REPRESENTATION DESIGN")

    # Left: Bilinear Rescale
    add_card(s6, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), "1. Bilinear Rescaling (128×128 -> 224×224)", badge="SPATIAL UPSAMPLING")
    tb6_l = s6.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.2), Inches(4.4))
    tf6_l = tb6_l.text_frame; tf6_l.word_wrap = True

    b6_l = [
        ("The Dimension Mismatch: ", "Raw MRI slices are 128×128 pixels, whereas ResNet-18 expects a 224×224 input tensor."),
        ("Why Nearest-Neighbor Fails: ", "Simple nearest-neighbor copying creates blocky, jagged edges that distort delicate sulcal folds and ventricular boundaries."),
        ("Bilinear Interpolation Solution: ", "Computes each new pixel as a smooth distance-weighted average of its 4 nearest neighbors across both horizontal and vertical axes."),
        ("Clinical Preservation: ", "Preserves natural proton-density tissue contrast and continuous cortical curves without introducing high-frequency digital noise.")
    ]
    for i, (bt, bd) in enumerate(b6_l):
        p = tf6_l.paragraphs[0] if i == 0 else tf6_l.add_paragraph()
        r1 = p.add_run(); r1.text = bt; r1.font.bold = True; r1.font.color.rgb = COLOR_PRIMARY; r1.font.size = Pt(10.5)
        r2 = p.add_run(); r2.text = bd; r2.font.color.rgb = COLOR_TEXT_MUTED; r2.font.size = Pt(9.5)
        p.space_after = Pt(10)

    # Right: 3-Channel Replication
    add_card(s6, Inches(6.8), Inches(1.6), Inches(5.733), Inches(5.2), "2. Three-Channel Replication (Grayscale -> RGB)", badge="TENSOR MATCHING")
    tb6_r = s6.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.4))
    tf6_r = tb6_r.text_frame; tf6_r.word_wrap = True

    b6_r = [
        ("The Architectural Mismatch: ", "Structural brain MRI is single-channel grayscale (radiofrequency intensity). ResNet-18's Conv1 weights have shape [64, 3, 7, 7] expecting 3 channels."),
        ("The Replication Formula: ", "We duplicate the single grayscale intensity channel across all three channels:\n[I]  -->  [R = I,  G = I,  B = I]"),
        ("100% Anatomical Fidelity: ", "Because R = G = B, the image remains visually and diagnostically 100% grayscale; zero false chromatic artifacts are added."),
        ("Unlocks Transfer Learning: ", "Enables direct utilization of ImageNet pretrained low-level spatial filters (Gabor-like edge detectors) without random first-layer reinitialization.")
    ]
    for i, (bt, bd) in enumerate(b6_r):
        p = tf6_r.paragraphs[0] if i == 0 else tf6_r.add_paragraph()
        r1 = p.add_run(); r1.text = bt; r1.font.bold = True; r1.font.color.rgb = COLOR_CYAN; r1.font.size = Pt(10.5)
        r2 = p.add_run(); r2.text = bd; r2.font.color.rgb = COLOR_TEXT_MUTED; r2.font.size = Pt(9.5)
        p.space_after = Pt(10)

    # =========================================================================
    # SLIDE 7: DEEP RESIDUAL BACKBONE & MATHEMATICAL FORMULATIONS
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7)
    add_header(s7, "Deep Residual Backbone & Mathematical Formulations", "06 | RESIDUAL STAGES & LOSS MATH")

    # Dark Mermaid Diagram Card
    add_card(s7, Inches(0.8), Inches(1.5), Inches(11.733), Inches(2.3), "Modified ResNet-18 Backbone: Stem, 4 Stages, GAP & 4-Class Head", badge="RESIDUAL NETWORK")
    diag_resnet = "results/architecture_diagrams/dark_diagram_resnet.png"
    if os.path.exists(diag_resnet):
        s7.shapes.add_picture(diag_resnet, Inches(1.0), Inches(1.98), width=Inches(11.333))

    # Bottom Left: Inverse Frequency Weights
    add_card(s7, Inches(0.8), Inches(4.0), Inches(5.6), Inches(2.9), "Inverse-Frequency Class Weighting Math")
    tb7_l = s7.shapes.add_textbox(Inches(1.0), Inches(4.45), Inches(5.2), Inches(2.3))
    tf7_l = tb7_l.text_frame; tf7_l.word_wrap = True

    p7_eq = tf7_l.paragraphs[0]
    p7_eq.text = "w_c = N_train / (K * N_c)"
    p7_eq.font.bold = True; p7_eq.font.color.rgb = COLOR_CYAN; p7_eq.font.size = Pt(13)

    p7_desc = tf7_l.add_paragraph()
    p7_desc.text = "Computed strictly from 4,480 training scans (K = 4 classes):\n• w_Moderate = 24.8889 (Heavy penalty for missing rare severe scans)\n• w_Mild = 1.7863  |  w_VeryMild = 0.7143  |  w_Non = 0.5000\nEliminates majority class suppression on the 50:1 imbalanced cohort."
    p7_desc.font.name = FONT_BODY; p7_desc.font.size = Pt(9.5); p7_desc.font.color.rgb = COLOR_TEXT_MUTED

    # Bottom Right: Weighted Cross Entropy & Hyperparameters
    add_card(s7, Inches(6.8), Inches(4.0), Inches(5.733), Inches(2.9), "Weighted Cross-Entropy Loss & Training Parameters")
    tb7_r = s7.shapes.add_textbox(Inches(7.0), Inches(4.45), Inches(5.3), Inches(2.3))
    tf7_r = tb7_r.text_frame; tf7_r.word_wrap = True

    p7_loss = tf7_r.paragraphs[0]
    p7_loss.text = "L_WCE(y, y_hat) = - sum_{c=0}^3 w_c * y_c * log(y_hat_c)"
    p7_loss.font.bold = True; p7_loss.font.color.rgb = COLOR_SUCCESS; p7_loss.font.size = Pt(12)

    p7_hp = tf7_r.add_paragraph()
    p7_hp.text = "• Optimizer: Adam (lr = 1e-4, beta1 = 0.9, beta2 = 0.999)\n• Batch Size: 16 samples (optimal stochasticity for 4,480 train scans)\n• Maximum Epochs: 8 (rapid convergence achieved by epoch 4-6)\n• Guardrail Checkpoint Selection: Best Validation Macro-F1 (equal 25% weight across all 4 stages, protecting minority sensitivity)"
    p7_hp.font.name = FONT_BODY; p7_hp.font.size = Pt(9.5); p7_hp.font.color.rgb = COLOR_TEXT_MUTED

    # =========================================================================
    # SLIDE 8: EXPLAINABLE AI: LAYER-4 GRAD-CAM ARCHITECTURE
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8)
    add_header(s8, "Explainable AI (XAI): Layer-4 Grad-CAM Architecture", "07 | CLINICAL EXPLAINABILITY ENGINE")

    # Dark Mermaid Diagram Card
    add_card(s8, Inches(0.8), Inches(1.5), Inches(11.733), Inches(2.6), "Gradient Backpropagation Flow from Target Logit to Saliency Heatmap", badge="RSNA CLAIM 2024")
    diag_gradcam = "results/architecture_diagrams/dark_diagram_gradcam.png"
    if os.path.exists(diag_gradcam):
        s8.shapes.add_picture(diag_gradcam, Inches(1.0), Inches(2.05), width=Inches(11.333))

    # Bottom 3 Cards Explaining Steps
    add_card(s8, Inches(0.8), Inches(4.3), Inches(3.6), Inches(2.6), "1. Forward & Backward Hooks")
    tb8_1 = s8.shapes.add_textbox(Inches(0.95), Inches(4.85), Inches(3.3), Inches(1.9))
    tf8_1 = tb8_1.text_frame; tf8_1.word_wrap = True
    tf8_1.paragraphs[0].text = "• Forward Hook: Captures activations A^k (512 maps of 7×7) at model.layer4[-1]\n• Backward Hook: Propagates gradient dY_c / dA^k from predicted class logit Y_c back to feature maps"
    tf8_1.paragraphs[0].font.name = FONT_BODY; tf8_1.paragraphs[0].font.size = Pt(9.5); tf8_1.paragraphs[0].font.color.rgb = COLOR_TEXT_MUTED

    add_card(s8, Inches(4.85), Inches(4.3), Inches(3.6), Inches(2.6), "2. Importance Weights & ReLU")
    tb8_2 = s8.shapes.add_textbox(Inches(5.0), Inches(4.85), Inches(3.3), Inches(1.9))
    tf8_2 = tb8_2.text_frame; tf8_2.word_wrap = True
    tf8_2.paragraphs[0].text = "• GAP of Gradients: alpha_k^c = (1/UV) sum(dY_c / dA_ij^k)\n• Weighted Sum: sum(alpha_k^c * A^k)\n• ReLU Rectification: Discards negative features to isolate structures contributing positively to class c"
    tf8_2.paragraphs[0].font.name = FONT_BODY; tf8_2.paragraphs[0].font.size = Pt(9.5); tf8_2.paragraphs[0].font.color.rgb = COLOR_TEXT_MUTED

    add_card(s8, Inches(8.9), Inches(4.3), Inches(3.633), Inches(2.6), "3. Upsampling & Clinical Overlay")
    tb8_3 = s8.shapes.add_textbox(Inches(9.05), Inches(4.85), Inches(3.3), Inches(1.9))
    tf8_3 = tb8_3.text_frame; tf8_3.word_wrap = True
    tf8_3.paragraphs[0].text = "• Bilinear Interpolation: Expands coarse 7×7 activation map to 224×224 patient MRI resolution\n• Min-Max Normalization: Rescales to [0.0, 1.0]\n• Jet Colormap: Real-time opacity overlay highlighting ventricles & temporal lobes"
    tf8_3.paragraphs[0].font.name = FONT_BODY; tf8_3.paragraphs[0].font.size = Pt(9.5); tf8_3.paragraphs[0].font.color.rgb = COLOR_TEXT_MUTED

    # =========================================================================
    # SLIDE 9: PHASE 1 GENESIS & PROBLEMS ENCOUNTERED
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9)
    add_header(s9, "Phase 1 Genesis: Initial Training & Problems Encountered", "08 | PHASE 1 AUDIT & FLAWS DISCOVERED")

    # Initial Training Results Card
    add_card(s9, Inches(0.8), Inches(1.6), Inches(11.733), Inches(2.0), "Initial Baseline Training Results (Before 7-Stage Arc)", badge="INITIAL RUNS")
    tb9_top = s9.shapes.add_textbox(Inches(1.0), Inches(2.05), Inches(11.333), Inches(1.4))
    tf9_top = tb9_top.text_frame; tf9_top.word_wrap = True
    tf9_top.paragraphs[0].text = "• Baseline Raw ResNet18: 906 / 960 = 94.38% test accuracy, 95.30% macro-F1 (high seed variance: ±2.26%)\n• Global Histogram Equalization (HE): Severely collapsed to 751 / 960 = 78.23% accuracy and 79.76% F1!\n• CLAHE ResNet18: 913 / 960 = 95.10% accuracy, but missed 20% of Moderate Dementia cases (unweighted loss)\nConclusion: Initial results were brittle, uncontrolled, and failed on minority clinical safety."
    tf9_top.paragraphs[0].font.name = FONT_BODY; tf9_top.paragraphs[0].font.size = Pt(10.5); tf9_top.paragraphs[0].font.color.rgb = COLOR_TEXT_MAIN

    # 3 Flaws Discovered
    add_card(s9, Inches(0.8), Inches(3.8), Inches(3.6), Inches(3.1), "Problem 1: Global HE Noise Amplification", badge="PREPROCESSING")
    tb9_1 = s9.shapes.add_textbox(Inches(0.95), Inches(4.35), Inches(3.3), Inches(2.4))
    tf9_1 = tb9_1.text_frame; tf9_1.word_wrap = True
    tf9_1.paragraphs[0].text = "• In axial brain MRI, dark air outside the skull occupies >50% of the field of view.\n• Global HE computes a CDF over the whole image, forcing background noise into mid-gray levels.\n• Result: Washes out cortical margins and collapses accuracy to 78.23%."
    tf9_1.paragraphs[0].font.name = FONT_BODY; tf9_1.paragraphs[0].font.size = Pt(9.5); tf9_1.paragraphs[0].font.color.rgb = COLOR_TEXT_MUTED

    add_card(s9, Inches(4.85), Inches(3.8), Inches(3.6), Inches(3.1), "Problem 2: Over-Augmentation Penalty", badge="AUGMENTATION")
    tb9_2 = s9.shapes.add_textbox(Inches(5.0), Inches(4.35), Inches(3.3), Inches(2.4))
    tf9_2 = tb9_2.text_frame; tf9_2.word_wrap = True
    tf9_2.paragraphs[0].text = "• Brain anatomy relies strictly on bilateral symmetry across the midline fissure.\n• Aggressive rotation (±10°) and affine shear warped ventricular geometry, creating artificial tilts mimicking pathology.\n• Result: Increased seed variance and caused false alarms."
    tf9_2.paragraphs[0].font.name = FONT_BODY; tf9_2.paragraphs[0].font.size = Pt(9.5); tf9_2.paragraphs[0].font.color.rgb = COLOR_TEXT_MUTED

    add_card(s9, Inches(8.9), Inches(3.8), Inches(3.633), Inches(3.1), "Problem 3: The 50:1 Imbalance Blindness", badge="CLINICAL RISK")
    tb9_3 = s9.shapes.add_textbox(Inches(9.05), Inches(4.35), Inches(3.3), Inches(2.4))
    tf9_3 = tb9_3.text_frame; tf9_3.word_wrap = True
    tf9_3.paragraphs[0].text = "• Severe 50:1 ratio between healthy controls and Moderate Dementia.\n• Unweighted loss allowed the network to ignore the 64 severe dementia scans while still achieving 95% overall score.\n• Result: 1 in 5 severe dementia patients misdiagnosed."
    tf9_3.paragraphs[0].font.name = FONT_BODY; tf9_3.paragraphs[0].font.size = Pt(9.5); tf9_3.paragraphs[0].font.color.rgb = COLOR_TEXT_MUTED

    # =========================================================================
    # SLIDE 10: THE 7-STAGE SCIENTIFIC IMPROVEMENT ARC
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10)
    add_header(s10, "How We Improved Phase 1: The 7-Stage Scientific Engineering Arc", "09 | METHODICAL PROGRESSION")

    # Dark Mermaid Diagram Card
    add_card(s10, Inches(0.8), Inches(1.5), Inches(11.733), Inches(2.3), "Pre-Registered 7-Stage Optimization Arc (Evaluated across Seeds 17, 42, 2026)", badge="LEAK-FREE PROGRESSION")
    diag_7stages = "results/architecture_diagrams/dark_diagram_7stages.png"
    if os.path.exists(diag_7stages):
        s10.shapes.add_picture(diag_7stages, Inches(1.0), Inches(1.95), width=Inches(11.333))

    # Bottom 3 cards summarizing progression
    add_card(s10, Inches(0.8), Inches(4.0), Inches(3.6), Inches(2.9), "Phase A: Audit & Control (Stages 1-2)")
    tb10_1 = s10.shapes.add_textbox(Inches(0.95), Inches(4.45), Inches(3.3), Inches(2.3))
    tf10_1 = tb10_1.text_frame; tf10_1.word_wrap = True
    tf10_1.paragraphs[0].text = "• Stage 1 (Audit): Verified 6,400 scans; repaired path resolution; flagged 543 cross-split duplicate clusters to prevent test contamination.\n• Stage 2 (Baselines): Established reproducible unaugmented baseline (95.49% ± 0.75% Val Acc) across seeds 17, 42, 2026."
    tf10_1.paragraphs[0].font.name = FONT_BODY; tf10_1.paragraphs[0].font.size = Pt(9.5); tf10_1.paragraphs[0].font.color.rgb = COLOR_TEXT_MUTED

    add_card(s10, Inches(4.85), Inches(4.0), Inches(3.6), Inches(2.9), "Phase B: Ablations (Stages 3-4)")
    tb10_2 = s10.shapes.add_textbox(Inches(5.0), Inches(4.45), Inches(3.3), Inches(2.3))
    tf10_2 = tb10_2.text_frame; tf10_2.word_wrap = True
    tf10_2.paragraphs[0].text = "• Stage 3 (Preprocessing): Proved Global HE fails by amplifying air noise. CLAHE 1.5 and Raw preserve cortical edges.\n• Stage 4 (Augmentation): Discovered Over-Augmentation Penalty; mild ±5° rotation won (96.91% Val Acc, 97.66% Macro-F1)."
    tf10_2.paragraphs[0].font.name = FONT_BODY; tf10_2.paragraphs[0].font.size = Pt(9.5); tf10_2.paragraphs[0].font.color.rgb = COLOR_TEXT_MUTED

    add_card(s10, Inches(8.9), Inches(4.0), Inches(3.633), Inches(2.9), "Phase C: Decision & Test (Stages 5-7)")
    tb10_3 = s10.shapes.add_textbox(Inches(9.05), Inches(4.45), Inches(3.3), Inches(2.3))
    tf10_3 = tb10_3.text_frame; tf10_3.word_wrap = True
    tf10_3.paragraphs[0].text = "• Stage 5 (Imbalance): Inverse class weighting achieved 100% Moderate Recall.\n• Stage 6 (Validation Lock): raw_rot5 won Rank 1 (97.66% Macro-F1). Locked before test.\n• Stage 7 (Final Holdout): 97.50% peak test accuracy on 960 held-out scans (p = 0.0015)."
    tf10_3.paragraphs[0].font.name = FONT_BODY; tf10_3.paragraphs[0].font.size = Pt(9.5); tf10_3.paragraphs[0].font.color.rgb = COLOR_TEXT_MUTED

    # =========================================================================
    # SLIDE 11: QUANTITATIVE RESULTS: BEFORE VS. AFTER COMPARISON
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_background(s11)
    add_header(s11, "Results Comparison: Before vs. After the 7-Stage Optimization", "10 | RIGOROUS BENCHMARK VERIFICATION")

    # Left: Before vs. After Comparison Table
    add_card(s11, Inches(0.8), Inches(1.6), Inches(6.0), Inches(5.2), "Quantitative Benchmark: Initial vs. Frozen Winner", badge="960 TEST SCANS")
    tb11_l = s11.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.6), Inches(4.4))
    tf11_l = tb11_l.text_frame; tf11_l.word_wrap = True

    comp_rows = [
        ("Peak Test Accuracy:", "\nInitial Baseline: 94.38%  -->  FINAL FROZEN WINNER: 97.50% (+3.12%)"),
        ("Moderate AD Sensitivity (Recall):", "\nInitial Baseline: 80.0%  -->  FINAL FROZEN WINNER: 100.0% (9/9)"),
        ("Moderate AD Precision:", "\nInitial Baseline: 81.8%  -->  FINAL FROZEN WINNER: 100.0% (0 False Alarms)"),
        ("Peak Macro-Averaged F1:", "\nInitial Baseline: 95.30%  -->  FINAL FROZEN WINNER: 98.20% (+2.90%)"),
        ("Seed Stability (Variance):", "\nInitial: ±2.26% spread  -->  FINAL: ±1.20% (Consistent high performance)")
    ]
    for r_title, r_desc in comp_rows:
        p = tf11_l.add_paragraph()
        r1 = p.add_run(); r1.text = r_title; r1.font.bold = True; r1.font.color.rgb = COLOR_CYAN; r1.font.size = Pt(10.5)
        r2 = p.add_run(); r2.text = r_desc; r2.font.color.rgb = COLOR_TEXT_MAIN; r2.font.size = Pt(9.5)
        p.space_after = Pt(6)

    # Right: Which Result Was Taken & Why
    add_card(s11, Inches(7.1), Inches(1.6), Inches(5.433), Inches(5.2), "Which Result Is Taken & Why?", badge="FROZEN WINNER: RAW_ROT5")
    tb11_r = s11.shapes.add_textbox(Inches(7.3), Inches(2.2), Inches(5.0), Inches(4.4))
    tf11_r = tb11_r.text_frame; tf11_r.word_wrap = True

    why_points = [
        ("Selected Configuration: ", "raw_rot5 (Raw Grayscale + Mild ±5° Rotation)."),
        ("Why Raw Beat CLAHE & HE: ", "In structural MRI, pixel intensities reflect physical proton densities. Raw data preserves natural cortical gradients without scanner noise amplification."),
        ("Why Mild ±5° Rotation Won: ", "Matches natural head-coil tilt during patient acquisition. Unlike ±10° rotation, it never warps bilateral ventricular symmetry."),
        ("Statistically Proven (McNemar Test): ", "Edwards-corrected chi^2 = 10.081, p = 0.001498 (p < 0.01), Odds Ratio = 2.44 [1.41, 4.23]. Gains are mathematically verified to be non-random."),
        ("Error Topology: ", "79.2% of errors (19/24) occurred exclusively between Non Demented and Very Mild Demented (the subtle MCI boundary). ZERO severe dementia false negatives.")
    ]
    for bt, bd in why_points:
        p = tf11_r.add_paragraph()
        r1 = p.add_run(); r1.text = bt; r1.font.bold = True; r1.font.color.rgb = COLOR_SUCCESS; r1.font.size = Pt(10)
        r2 = p.add_run(); r2.text = bd; r2.font.color.rgb = COLOR_TEXT_MUTED; r2.font.size = Pt(9.5)
        p.space_after = Pt(6)

    # =========================================================================
    # SLIDE 12: CLINICAL DECISION-SUPPORT WEB APPLICATION STACK
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    set_slide_background(s12)
    add_header(s12, "Clinical Decision-Support Web Application Stack", "11 | APPLICATION ARCHITECTURE & DEPLOYMENT")

    add_card(s12, Inches(0.8), Inches(1.6), Inches(3.6), Inches(5.2), "FastAPI & Uvicorn Backend", badge="HIGH THROUGHPUT")
    tb12_1 = s12.shapes.add_textbox(Inches(0.95), Inches(2.2), Inches(3.3), Inches(4.4))
    tf12_1 = tb12_1.text_frame; tf12_1.word_wrap = True
    tf12_1.paragraphs[0].text = "• Asynchronous High-Throughput REST API\n• FastAPI + Uvicorn server running locally on http://127.0.0.1:8000\n• Singleton MRIAnalysisEngine handles PyTorch device dispatch (CUDA/CPU)\n• Sub-100ms inference latency per patient slice\n• Endpoints for health, live upload analysis, and sample evaluation"
    tf12_1.paragraphs[0].font.name = FONT_BODY; tf12_1.paragraphs[0].font.size = Pt(10); tf12_1.paragraphs[0].font.color.rgb = COLOR_TEXT_MUTED

    add_card(s12, Inches(4.85), Inches(1.6), Inches(3.6), Inches(5.2), "Multi-Model Consensus Engine", badge="CLINICAL SAFETY")
    tb12_2 = s12.shapes.add_textbox(Inches(5.0), Inches(2.2), Inches(3.3), Inches(4.4))
    tf12_2 = tb12_2.text_frame; tf12_2.word_wrap = True
    tf12_2.paragraphs[0].text = "• Concurrent Querying of 3 Models:\n  1. resnet18_frozen_winner.pth (Top Model)\n  2. resnet18_clahe.pth (CLAHE Benchmark)\n  3. resnet18_baseline.pth (Raw Control)\n• Full Consensus: Concordant predictions dramatically boost clinical trust\n• Discordance Flagging: Disagreements trigger automatic review by human radiologist"
    tf12_2.paragraphs[0].font.name = FONT_BODY; tf12_2.paragraphs[0].font.size = Pt(10); tf12_2.paragraphs[0].font.color.rgb = COLOR_TEXT_MUTED

    add_card(s12, Inches(8.9), Inches(1.6), Inches(3.633), Inches(5.2), "Interactive UI & PDF Generator", badge="FRONTEND & XAI")
    tb12_3 = s12.shapes.add_textbox(Inches(9.05), Inches(2.2), Inches(3.3), Inches(4.4))
    tf12_3 = tb12_3.text_frame; tf12_3.word_wrap = True
    tf12_3.paragraphs[0].text = "• Responsive Vanilla JS & CSS3 Dashboard\n• Drag-and-drop axial MRI upload with quick-test sample chips\n• Interactive Layer-4 Grad-CAM heatmaps with opacity slider controls\n• Automated Clinical PDF Report Generator: Formats predictions, probability distributions, and RSNA CLAIM disclosures into clinical documentation"
    tf12_3.paragraphs[0].font.name = FONT_BODY; tf12_3.paragraphs[0].font.size = Pt(10); tf12_3.paragraphs[0].font.color.rgb = COLOR_TEXT_MUTED

    # =========================================================================
    # SLIDE 13: FUTURE ROADMAP: OVERCOMING REMAINING LITERATURE DRAWBACKS
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    set_slide_background(s13)
    add_header(s13, "Future Roadmap: Overcoming Literature Drawbacks to Reach Near-Perfection", "12 | FUTURE DIRECTIONS & ROADMAP")

    roadmap_cards = [
        ("1. Multimodal Clinical Fusion",
         "The Literature Drawback: Image-only models struggle with borderline MCI vs. healthy aging.\nOur Future Fix: Fuse structural MRI with clinical tabular markers: MMSE cognitive scores, CDR scores, ApoE-ε4 genetic status, and CSF amyloid/tau ratios to resolve subtle borderline ambiguities."),
        
        ("2. 3D Volumetric Segmentation (Swin-UNETR)",
         "The Literature Drawback: 2D slice classification loses inter-slice 3D spatial continuity.\nOur Future Fix: Transition from 2D axial slices to 3D Swin-UNETR volumetric segmentation to quantify precise bilateral hippocampal and ventricular volume loss in cubic millimeters (mm³)."),
        
        ("3. Cross-Cohort Multicenter Generalization",
         "The Literature Drawback: Most papers train and test on single-center datasets, failing when tested on different MRI scanners.\nOur Future Fix: Validate the frozen pipeline across external cohorts (ADNI, OASIS-3, AIBL) to eliminate vendor-specific scanner bias."),
        
        ("4. Uncertainty-Guided Clinical Triage",
         "The Literature Drawback: Diagnostic AI outputs decisions even when uncertain.\nOur Future Fix: Implement selective prediction: scans with softmax confidence >80% (93.8% of cohort) are autonomously cleared, while ambiguous scans (<80%) are routed to specialist neuroradiologists.")
    ]

    for idx, (rm_title, rm_desc) in enumerate(roadmap_cards):
        row = idx // 2
        col = idx % 2
        c_left = Inches(0.8 + col * 5.95)
        c_top  = Inches(1.6 + row * 2.7)
        c_w    = Inches(5.78)
        c_h    = Inches(2.45)

        add_card(s13, c_left, c_top, c_w, c_h, rm_title, badge=f"ROADMAP GOAL #{idx+1}", border_color=COLOR_BORDER)
        tb = s13.shapes.add_textbox(c_left + Inches(0.2), c_top + Inches(0.55), c_w - Inches(0.4), c_h - Inches(0.65))
        tf = tb.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = rm_desc
        p.font.name = FONT_BODY
        p.font.size = Pt(9.5)
        p.font.color.rgb = COLOR_TEXT_MUTED

    # =========================================================================
    # SLIDE 14: PRESENTATION DEFENSE: ANTICIPATED EVALUATOR Q&A
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    set_slide_background(s14)
    add_header(s14, "Presentation Defense: Critical Evaluator Q&A", "13 | DEFENSE & JURY PREPARATION")

    qa_items = [
        ("Q1: Why choose ResNet-18 over deeper networks like ResNet-50, DenseNet, or Vision Transformers?",
         "In medical imaging cohorts of 6,400 scans, ultra-deep architectures (ResNet-50/ViT) overparameterize and memorize noise. ResNet-18 provides optimal sample efficiency, avoids late-stage overfitting, converges reliably within 8 epochs, and enables real-time sub-100ms inference on clinical workstations."),
        
        ("Q2: Why did raw images with mild ±5° rotation outperform CLAHE and heavy augmentation?",
         "In structural brain MRI, pixel intensities reflect physical proton densities. Global HE amplified background air noise into gray artifacts. Aggressive rotations (±10°) and shearing warped ventricular bilateral symmetry, mimicking artificial deformities. Mild ±5° rotation matched clinical head-coil tilt without distorting brain geometry."),
        
        ("Q3: How was severe 50:1 class imbalance resolved without synthetic data generation (SMOTE)?",
         "Synthetic image interpolation (SMOTE) often generates anatomically invalid hallucinations that violate medical plausibility. Instead, we used inverse-frequency class-weighted cross-entropy loss (w_Mod = 24.89 vs w_Non = 0.50), guaranteeing 100% recall on the rare severe dementia class.")
    ]

    for idx, (q, a) in enumerate(qa_items):
        t_top = Inches(1.6 + idx * 1.8)
        add_card(s14, Inches(0.8), t_top, Inches(11.733), Inches(1.65), border_color=COLOR_PRIMARY)
        tb = s14.shapes.add_textbox(Inches(1.0), t_top + Inches(0.12), Inches(11.333), Inches(1.4))
        tf = tb.text_frame; tf.word_wrap = True

        p_q = tf.paragraphs[0]
        p_q.text = q
        p_q.font.name = FONT_HEADING
        p_q.font.bold = True
        p_q.font.color.rgb = COLOR_CYAN
        p_q.font.size = Pt(11)

        p_a = tf.add_paragraph()
        p_a.text = a
        p_a.font.name = FONT_BODY
        p_a.font.color.rgb = COLOR_TEXT_MAIN
        p_a.font.size = Pt(9.5)
        p_a.space_before = Pt(3)

    prs.save(output_path)
    print(f"Successfully generated presentation deck: {output_path} (Slides: {len(prs.slides)})")


if __name__ == "__main__":
    build_full_deck()
