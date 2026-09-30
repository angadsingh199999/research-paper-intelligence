import re


COMMON_HEADINGS = {
    "abstract",
    "introduction",
    "background",
    "theoretical background",
    "literature review",
    "related work",
    "methodology",
    "methods",
    "materials and methods",
    "research methodology",
    "results",
    "discussion",
    "general discussion",
    "conclusion",
    "limitations",
    "future work",
    "future research",
    "references",
    "appendix",
}


def is_heading(line):
    """
    Detect normal academic section headings.
    """

    line = line.strip()

    if not line:
        return False

    # Common headings
    if line.lower() in COMMON_HEADINGS:
        return True

    # Numbered headings
    pattern = (
        r"^\d+(?:\.\d+)*\.?\s+"
        r"[A-Z][^.!?]{0,120}$"
    )

    return bool(re.match(pattern, line))


def split_merged_heading(line):
    """
    Detect cases such as:

    4.3.1 Participants and design. Data was collected...
    """

    pattern = (
        r"^(\d+(?:\.\d+)*\.?)\s+"
        r"(.+?)\.\s+"
        r"([A-Z].*)$"
    )

    match = re.match(
        pattern,
        line.strip()
    )

    if not match:
        return None, None

    number = match.group(1)
    heading_text = match.group(2).strip()
    body_text = match.group(3).strip()

    if len(heading_text) > 120:
        return None, None

    heading = f"{number} {heading_text}"

    return heading, body_text


def create_section_blocks(pages):

    processed_pages = []

    current_section = "Unknown"

    for page in pages:

        lines = page["text"].splitlines()

        blocks = []

        current_paragraph = []

        def save_paragraph():

            if not current_paragraph:
                return

            text = " ".join(
                current_paragraph
            ).strip()

            if text:

                blocks.append({
                    "section": current_section,
                    "text": text,
                    "page_number": page["page_number"]

                })

            current_paragraph.clear()

        for line in lines:

            line = line.strip()

            # -----------------------------------------
            # Blank line
            # -----------------------------------------

            if not line:

                save_paragraph()

                continue

            # -----------------------------------------
            # Merged numbered heading + body
            # -----------------------------------------

            heading, body = split_merged_heading(
                line
            )

            if heading:

                save_paragraph()

                current_section = heading

                if body:
                    current_paragraph.append(
                        body
                    )

                continue

            # -----------------------------------------
            # Normal heading
            # -----------------------------------------

            if is_heading(line):

                save_paragraph()

                current_section = line

                continue

            # -----------------------------------------
            # Normal body
            # -----------------------------------------

            current_paragraph.append(line)

        # Save final paragraph
        save_paragraph()

        processed_pages.append({
            "page_number": page["page_number"],
            "blocks": blocks
        })

    return processed_pages