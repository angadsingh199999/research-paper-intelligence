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


def extract_numbered_heading(line):
    """
    Try to extract a numbered section heading from a line.

    Example:

        4.3.1 Participants and design. Data was collected...

    becomes:

        4.3.1 Participants and design
    """

    pattern = r"^(\d+(?:\.\d+)*\.?)\s+(.+?)\.\s+[A-Z]"

    match = re.match(pattern, line.strip())

    if match:

        number = match.group(1)
        heading_text = match.group(2).strip()

        # Heading should not be excessively long
        if len(heading_text) <= 120:

            heading = f"{number} {heading_text}"

            return heading

    return None


def is_numbered_heading(line):
    """
    Check whether a line is a clean numbered heading.
    """

    line = line.strip()

    pattern = r"^\d+(?:\.\d+)*\.?\s+[A-Z][^.!?]{0,120}$"

    return bool(re.match(pattern, line))


def is_common_heading(line):
    """
    Check for common unnumbered academic headings.
    """

    return line.strip().lower() in COMMON_HEADINGS


def detect_sections(pages):
    """
    Detect sections while preserving page-level text.
    """

    current_section = "Unknown"

    processed_pages = []

    for page in pages:

        lines = page["text"].splitlines()

        page_text = []

        for line in lines:

            line = line.strip()

            if not line:
                continue

            # -----------------------------------------
            # Case 1: Common unnumbered heading
            # -----------------------------------------

            if is_common_heading(line):

                current_section = line

                continue

            # -----------------------------------------
            # Case 2: Clean numbered heading
            # -----------------------------------------

            if is_numbered_heading(line):

                current_section = line

                continue

            # -----------------------------------------
            # Case 3: Heading merged with body text
            # -----------------------------------------

            extracted_heading = extract_numbered_heading(line)

            if extracted_heading:

                current_section = extracted_heading

                # Remove heading from the line.
                remaining_text = line[
                    len(extracted_heading):
                ].strip()

                # Remove leading punctuation
                remaining_text = remaining_text.lstrip(". ")

                if remaining_text:
                    page_text.append(remaining_text)

                continue

            # -----------------------------------------
            # Normal body text
            # -----------------------------------------

            page_text.append(line)

        processed_pages.append({
            "page_number": page["page_number"],
            "section": current_section,
            "text": "\n".join(page_text)
        })

    return processed_pages