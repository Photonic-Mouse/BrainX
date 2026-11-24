import os
import sys
from pathlib import Path

import reportlab
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.pdfgen import canvas

# ----------------------------------------------------------------------
# 1. CANVAS WITH RUNNING HEADER & DYNAMIC "PAGE X OF Y" FOOTER
# ----------------------------------------------------------------------
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        # Running header (Page 2 onwards)
        if self._pageNumber > 1:
            self.setFont("Helvetica-Bold", 7.5)
            self.setFillColor(colors.HexColor("#0F2C59"))
            self.drawString(36, 812, "BrainScope-XAI (BrainX): Phase 1 Engineering & Clinical Intelligence Report")
            self.setFont("Helvetica", 7.5)
            self.setFillColor(colors.HexColor("#64748B"))
            self.drawRightString(559, 812, "Forensic Audit & Phase 2 Blueprint")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(36, 806, 559, 806)

        # Running footer (All pages)
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(36, 22, "Confidential - Academic Review & Research Faculty Evaluation Only")
        self.drawRightString(559, 22, f"Page {self._pageNumber} of {page_count}")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(36, 32, 559, 32)
        self.restoreState()


# ----------------------------------------------------------------------
# 2. COLOR PALETTE & TYPOGRAPHY STYLES
# ----------------------------------------------------------------------
PRIMARY = colors.HexColor("#0F2C59")      # Deep Navy
SECONDARY = colors.HexColor("#1E40AF")    # Royal Blue
DARK_SLATE = colors.HexColor("#1E293B")   # Slate Black
BODY_COLOR = colors.HexColor("#334155")   # Slate Body
ACCENT_RED = colors.HexColor("#991B1B")   # Crimson (Gaps / Warnings)
ACCENT_GREEN = colors.HexColor("#065F46") # Emerald (Verified Fixes)
BG_LIGHT = colors.HexColor("#F8FAFC")     # Soft Background Gray
BG_ALT = colors.HexColor("#F1F5F9")       # Table Alternate Gray
BORDER_COLOR = colors.HexColor("#CBD5E1") # Divider Line Gray

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "DocTitle",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=18,
    leading=22,
    textColor=PRIMARY,
    alignment=TA_LEFT,
    spaceAfter=4
)

subtitle_style = ParagraphStyle(
    "DocSubtitle",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=9.5,
    leading=13,
    textColor=colors.HexColor("#475569"),
    alignment=TA_LEFT,
    spaceAfter=8
)

h1_style = ParagraphStyle(
    "Heading1_Custom",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=12,
    leading=15,
    textColor=PRIMARY,
    spaceBefore=10,
    spaceAfter=4,
    keepWithNext=True
)

h2_style = ParagraphStyle(
    "Heading2_Custom",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=10,
    leading=13,
    textColor=SECONDARY,
    spaceBefore=7,
    spaceAfter=3,
    keepWithNext=True
)

h3_style = ParagraphStyle(
    "Heading3_Custom",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=8.5,
    leading=11,
    textColor=DARK_SLATE,
    spaceBefore=5,
    spaceAfter=2,
    keepWithNext=True
)

body_style = ParagraphStyle(
    "Body_Custom",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=8,
    leading=11,
    textColor=BODY_COLOR,
    alignment=TA_JUSTIFY,
    spaceAfter=4
)

bullet_style = ParagraphStyle(
    "Bullet_Custom",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=8,
    leading=10.5,
    textColor=BODY_COLOR,
    leftIndent=12,
    firstLineIndent=-8,
    alignment=TA_JUSTIFY,
    spaceAfter=2
)

callout_style = ParagraphStyle(
    "CalloutText",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=7.5,
    leading=10,
    textColor=DARK_SLATE,
    alignment=TA_LEFT
)

caption_style = ParagraphStyle(
    "Caption_Custom",
    parent=styles["Normal"],
    fontName="Helvetica-Oblique",
    fontSize=7,
    leading=9,
    textColor=colors.HexColor("#64748B"),
    alignment=TA_CENTER,
    spaceBefore=2,
    spaceAfter=6
)

tbl_header = ParagraphStyle(
    "TableHeader",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=7,
    leading=8.5,
    textColor=colors.white,
    alignment=TA_CENTER
)

tbl_cell = ParagraphStyle(
    "TableCell",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=6.8,
    leading=8.5,
    textColor=DARK_SLATE,
    alignment=TA_LEFT
)

tbl_cell_center = ParagraphStyle(
    "TableCellCenter",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=6.8,
    leading=8.5,
    textColor=DARK_SLATE,
    alignment=TA_CENTER
)

tbl_cell_right = ParagraphStyle(
    "TableCellRight",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=6.8,
    leading=8.5,
    textColor=DARK_SLATE,
    alignment=TA_RIGHT
)

tbl_cell_bold = ParagraphStyle(
    "TableCellBold",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=6.8,
    leading=8.5,
    textColor=DARK_SLATE,
    alignment=TA_LEFT
)

def make_callout(text: str, border_color=SECONDARY, bg_color=BG_LIGHT):
    p = Paragraph(text, callout_style)
    t = Table([[p]], colWidths=[520])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg_color),
        ('BOX', (0,0), (-1,-1), 1, border_color),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    return t

print("Setup completed successfully.")
