from app.ingestion.pdf_parser import extract_text_from_pdf
from app.ingestion.section_detector import detect_sections


pdf_path = "data/raw_papers/paper1.pdf"


pages = extract_text_from_pdf(pdf_path)

section_pages = detect_sections(pages)


for page in section_pages:

    if page["page_number"] in [1, 4, 6, 8]:

        print("\n" + "=" * 70)

        print(f"PAGE: {page['page_number']}")

        print(f"SECTION: {page['section']}")

        print("=" * 70)

        print(page["text"][:2000])