from app.ingestion.pdf_parser import extract_text_from_pdf
from app.ingestion.section_blocks import create_section_blocks


pdf_path = "data/raw_papers/paper1.pdf"


pages = extract_text_from_pdf(pdf_path)

section_pages = create_section_blocks(pages)


print("=" * 80)
print("SECTION BLOCK INSPECTION")
print("=" * 80)


for page in section_pages:

    for block in page["blocks"]:

        if block["section"] == "3. Conceptual model":

            print("\n" + "-" * 80)

            print(
                f"PAGE: {page['page_number']}"
            )

            print(
                f"SECTION: {block['section']}"
            )

            print(
                f"WORDS: "
                f"{len(block['text'].split())}"
            )

            print("\nTEXT:")

            print(block["text"][:1500])