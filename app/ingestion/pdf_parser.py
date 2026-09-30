import pymupdf

from app.ingestion.text_cleaner import clean_text
from app.ingestion.content_filter import remove_publisher_content


def extract_text_from_pdf(pdf_path):

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(
        document,
        start=1
    ):

        raw_text = page.get_text()

        cleaned_text = clean_text(
            raw_text
        )

        cleaned_text = remove_publisher_content(
            cleaned_text
        )

        pages.append({
            "page_number": page_number,
            "text": cleaned_text
        })

    document.close()

    return pages