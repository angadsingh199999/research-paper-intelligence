import sys
import json

from app.ingestion.ingest import ingest_paper


def main():

    if len(sys.argv) != 2:

        print(
            "Usage: python ingest.py <pdf_path>"
        )

        return

    pdf_path = sys.argv[1]

    paper = ingest_paper(pdf_path)

    print("\n" + "=" * 70)
    print("INGESTION SUCCESSFUL")
    print("=" * 70)

    print(f"\nPaper ID: {paper.metadata.paper_id}")
    print(f"Title: {paper.metadata.title}")
    print(f"Authors: {paper.metadata.authors}")
    print(f"Pages: {paper.metadata.total_pages}")

    print("\n" + "=" * 70)
    print("DOCUMENT STRUCTURE")
    print("=" * 70)

    for page in paper.pages:

        print(
            f"\nPAGE {page.page_number}"
        )

        for block in page.blocks:

            print(
                f"  SECTION: {block.section}"
            )

            print(
                f"  TEXT: {block.text[:150]}..."
            )


if __name__ == "__main__":
    main()