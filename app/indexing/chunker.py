from typing import List

from app.models.schemas import (
    ResearchPaper,
    Chunk
)


TARGET_WORDS = 500
OVERLAP_WORDS = 75


def create_chunk(
    paper: ResearchPaper,
    chunk_number: int,
    section: str,
    paragraphs: list,
    pages: list
) -> Chunk:

    text = "\n\n".join(paragraphs)

    return Chunk(
        chunk_id=(
            f"{paper.metadata.paper_id}"
            f"_chunk_{chunk_number:04d}"
        ),

        paper_id=(
            paper.metadata.paper_id
        ),

        paper_title=(
            paper.metadata.title
        ),

        section=section,

        text=text,

        page_start=min(pages),

        page_end=max(pages),

        word_count=len(text.split())
    )


def create_chunks(
    paper: ResearchPaper
) -> List[Chunk]:

    chunks = []

    chunk_number = 1

    current_section = None
    current_text = []
    current_pages = []
    current_words = 0

    for page in paper.pages:

        for block in page.blocks:

            section = block.section
            text = block.text.strip()

            if not text:
                continue

            words = text.split()
            word_count = len(words)

            # Retain substantive research text even if section heading was not detected,
            # but skip trivial short header fragments (< 25 words).
            if section == "Unknown":
                if word_count < 25:
                    continue
                section = "Introduction"

            # ----------------------------------------
            # New section
            # ----------------------------------------

            if (
                current_section is not None
                and section != current_section
            ):

                if current_text:

                    chunks.append(
                        create_chunk(
                            paper,
                            chunk_number,
                            current_section,
                            current_text,
                            current_pages
                        )
                    )

                    chunk_number += 1

                current_text = []
                current_pages = []
                current_words = 0

            current_section = section

            # ----------------------------------------
            # Add text to current chunk
            # ----------------------------------------

            current_text.append(text)

            current_pages.append(
                block.page_number
            )

            current_words += word_count

            # ----------------------------------------
            # Split once target is reached
            # ----------------------------------------

            if current_words >= TARGET_WORDS:

                chunks.append(
                    create_chunk(
                        paper,
                        chunk_number,
                        current_section,
                        current_text,
                        current_pages
                    )
                )

                chunk_number += 1

                # ------------------------------------
                # Keep overlap from END
                # ------------------------------------

                overlap_text = []
                overlap_pages = []
                overlap_words = 0

                for i in range(
                    len(current_text) - 1,
                    -1,
                    -1
                ):

                    paragraph = current_text[i]

                    paragraph_words = len(
                        paragraph.split()
                    )

                    if (
                        overlap_words
                        + paragraph_words
                        > OVERLAP_WORDS
                    ):
                        break

                    overlap_text.insert(
                        0,
                        paragraph
                    )

                    overlap_pages.insert(
                        0,
                        current_pages[i]
                    )

                    overlap_words += (
                        paragraph_words
                    )

                current_text = overlap_text
                current_pages = overlap_pages
                current_words = overlap_words

    # ----------------------------------------------
    # Final chunk
    # ----------------------------------------------

    if current_text:

        chunks.append(
            create_chunk(
                paper,
                chunk_number,
                current_section,
                current_text,
                current_pages
            )
        )

    return chunks