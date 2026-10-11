"""Render the original HTML diagrams directly, retaining searchable font text."""
from pathlib import Path

from weasyprint import HTML
from weasyprint.text.fonts import FontConfiguration

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT.parent / "figure/04_design"

for source, name in (("overview", "fig_overview"),
                     ("prefix_coherence", "fig_prefix_coherence")):
    path = OUTPUT / f"{name}.pdf"
    HTML(filename=str(ROOT / f"{source}.html")).write_pdf(
        path, font_config=FontConfiguration(), pdf_version="1.5")
    print(path)
