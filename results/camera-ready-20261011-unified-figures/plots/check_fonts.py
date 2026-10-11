"""List Type 3 or unembedded fonts; nonzero exit means a font issue remains."""
import argparse
from pathlib import Path

import pymupdf

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("path", type=Path, help="PDF or directory of PDFs")
args = parser.parse_args()
paths = sorted(args.path.rglob("*.pdf")) if args.path.is_dir() else [args.path]
if not paths or any(not path.is_file() for path in paths):
    parser.error("No PDFs found")

issues = 0
for path in paths:
    with pymupdf.open(path) as doc:
        fonts = {font[0]: font for page in doc for font in page.get_fonts(full=True)}
        for xref, font in fonts.items():
            if font[2] == "Type3" or not doc.extract_font(xref)[3]:
                print(f"{path}: {font[2]} {font[3]} (xref {xref})")
                issues += 1
print(f"Checked {len(paths)} PDFs; {issues} font issues.")
raise SystemExit(bool(issues))
