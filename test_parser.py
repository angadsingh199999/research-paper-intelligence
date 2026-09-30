from app.ingestion.pdf_parser import extract_text_from_pdf
from app.ingestion.metadata_extractor import extract_metadata
from app.ingestion.section_detector import detect_sections


pdf_path = "data/raw_papers/paper1.pdf"


# -----------------------------------
# 1. Extract PDF
# -----------------------------------

pages = extract_text_from_pdf(pdf_path)

print(f"Total pages extracted: {len(pages)}")


# -----------------------------------
# 2. Extract metadata
# -----------------------------------

metadata = extract_metadata(pages, pdf_path)

print("\n" + "=" * 60)
print("PAPER METADATA")
print("=" * 60)

print(f"Paper ID: {metadata['paper_id']}")
print(f"Filename: {metadata['filename']}")
print(f"Title: {metadata['title']}")
print(f"Authors: {metadata['authors']}")
print(f"Total Pages: {metadata['total_pages']}")


# -----------------------------------
# 3. Detect sections
# -----------------------------------

section_pages = detect_sections(pages)


print("\n" + "=" * 60)
print("DETECTED SECTIONS")
print("=" * 60)


previous_section = None

for page in section_pages:

    section = page["section"]

    if section != previous_section:

        print(
            f"Page {page['page_number']} → {section}"
        )

        previous_section = section