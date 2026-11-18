"""merge_pdf.py — insert cover as page 0 of the body PDF, normalize to A4.
Portable: all paths are package-relative. Requires pypdf.
"""
import os
from pathlib import Path
from pypdf import PdfReader, PdfWriter

PACKAGE_DIR = Path(__file__).resolve().parent

A4_W, A4_H = 595.28, 841.89

def normalize(page):
    w, h = float(page.mediabox.width), float(page.mediabox.height)
    if abs(w - A4_W) > 0.1 or abs(h - A4_H) > 0.1:
        page.scale_to(A4_W, A4_H)
        page.mediabox.lower_left = (0, 0)
        page.mediabox.upper_right = (A4_W, A4_H)
    return page

writer = PdfWriter()
cover = PdfReader(str(PACKAGE_DIR / "cover.pdf")).pages[0]
writer.add_page(normalize(cover))
for p in PdfReader(str(PACKAGE_DIR / "report_body.pdf")).pages:
    writer.add_page(normalize(p))
writer.add_metadata({
    "/Title": "Quantitative Strategy Review - FVG Momentum Alpha Basket",
    "/Author": "Z.ai", "/Creator": "Z.ai",
    "/Subject": "Repository audit, runner-exit optimization, risk parity, meta-labeling, statistical validity",
})
out = str(PACKAGE_DIR / "FVG_Strategy_Quant_Review.pdf")
with open(out, "wb") as f:
    writer.write(f)
print("merged ->", out, "pages:", len(writer.pages))
