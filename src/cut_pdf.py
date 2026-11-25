# cut_pdf.py — cuts the corpus (chapters 1–5 without applied sections) out of the full
# textbook, plus one PDF per chapter. The chapter PDFs are then uploaded to Google Drive
# and parsed in Colab (colab/parse_mineru.ipynb).
#
#   python src/cut_pdf.py                    -> uses data/raw/Lay.pdf
#   python src/cut_pdf.py path/to/Lay.pdf
import sys
from pathlib import Path

from pypdf import PdfReader, PdfWriter

from config import RAW_PDF, PDF_DIR, CHAPTER_RANGES


def write_pages(reader, ranges, output: Path) -> int:
    writer = PdfWriter()
    for start, end in ranges:
        for page_num in range(start, end + 1):
            writer.add_page(reader.pages[page_num - 1])  # pypdf is 0-based
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        writer.write(f)
    return len(writer.pages)


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else RAW_PDF
    if not src.exists():
        sys.exit(f"{src} not found. Put the full textbook at data/raw/Lay.pdf (see data/README.md).")
    reader = PdfReader(str(src))
    print(f"Source: {src} ({len(reader.pages)} pages)")

    all_ranges = [r for ranges in CHAPTER_RANGES.values() for r in ranges]
    n = write_pages(reader, all_ranges, PDF_DIR / "lay_ch1-5_theory.pdf")
    print(f"  lay_ch1-5_theory.pdf: {n} pages")

    for ch, ranges in CHAPTER_RANGES.items():
        n = write_pages(reader, ranges, PDF_DIR / "chapters" / f"lay_{ch}.pdf")
        print(f"  chapters/lay_{ch}.pdf: {n} pages")


if __name__ == "__main__":
    main()
