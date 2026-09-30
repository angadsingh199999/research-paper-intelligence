import re


PUBLISHER_PATTERNS = [
    r"^\d{1,2}\s+[A-Za-z]+\s+\d{4}$",
    r"^Accepted\s+\d{1,2}\s+[A-Za-z]+\s+\d{4}$",
    r"^European Journal(?:\s+of\s+Innovation)?(?:\s+Management)?$",
    r"^of Innovation(?:\s+Management)?$",
    r"^Management$",
    r"^Vol\.\s+\d+",
    r"^pp\.\s+\d+",
    r"^Emerald Publishing",
    r"^e-ISSN:",
    r"^p-ISSN:",
    r"^DOI\s+",
    r"^Downloaded from",
    r"^EJIM\s+\d+[\s,]+\d+\s+\d+$",
    r"^.*?et al\.\s+Sustainable Futures\s+\d+.*$",
    r"^Sustainable Futures\s+\d+.*$",
    r"^Source:\s*(?:Compiled|Authors?['’]?(?:\s+Analysis)?)[^\n\r]*$",
    r"^\(?\s*\)?\s*\d{6}\s+\d{1,3}$",
]


# ================================================================
# INLINE ARTIFACT PATTERNS
#
# Strip running headers/footers that get welded onto content lines
# ================================================================

INLINE_ARTIFACT_PATTERNS = [
    r"\bEJIM\s+\d{1,3}\s*,\s*\d{1,3}\s+\d{1,4}\b",
    r"\bEJIM\s+\d{1,3}\s+\d{1,4}\b",
    r"\b(?:The\s+)?European Journal(?:\s+of\s+Innovation)?(?:\s+Management)?\s+\d{1,4}\b",
    r"\b[A-Z]\.?\s*[A-Za-z]+.*?\bet al\.\s+Sustainable Futures\s+\d+.*?(\(\d{4}\))?.*?\d{1,4}\b",
    r"\bSustainable Futures\s+\d+\s*(?:\(\d{4}\))?\s*\d*\b",
    r"Source:\s*(?:Compiled\s+by\s+the\s+Author|Authors?['’]?(?:\s+Analysis)?)\s*\)?\s*\d{4,8}\s+\d{1,3}",
    r"Source:\s*(?:Compiled\s+by\s+the\s+Author|Authors?['’]?(?:\s+Analysis)?)[^\n.]*",
    r"\)\s*\d{6}\s+\d{1,3}\b",
    r"\b\d{6}\s+\d{1,3}\b",  # e.g. "102027 10", article IDs welded to page count
]


def is_publisher_line(line):
    line = line.strip()

    if not line:
        return True

    for pattern in PUBLISHER_PATTERNS:
        if re.search(
            pattern,
            line,
            re.IGNORECASE
        ):
            return True

    return False


def remove_publisher_content(text):
    if not text:
        return ""

    # ------------------------------------------------------------
    # First pass: strip known inline artifacts wherever they occur
    # ------------------------------------------------------------
    for pattern in INLINE_ARTIFACT_PATTERNS:
        text = re.sub(
            pattern,
            " ",
            text,
            flags=re.IGNORECASE
        )

    # ------------------------------------------------------------
    # Second pass: whole-line filter for isolated header/footer lines
    # ------------------------------------------------------------
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        if not is_publisher_line(line):
            cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)

    # Collapse doubled-up spacing from inline replacements
    cleaned_text = re.sub(
        r"[ \t]{2,}",
        " ",
        cleaned_text
    )

    return cleaned_text.strip()


def sanitize_answer_text(text):
    """
    Final-pass safety net to ensure no publisher debris, table header fragments,
    or floating OCR junk slips into user-facing answers.
    """
    if not text:
        return ""

    # Strip inline artifacts
    for pattern in INLINE_ARTIFACT_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    # Remove floating source citations like "Source: Authors' Analysis ) 102027 10"
    text = re.sub(r"Source:\s*[A-Za-z0-9 ’'()]+\d+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"EJIM\s+\d+[\s,]+\d+\s+\d+", " ", text, flags=re.IGNORECASE)

    # Clean multiple spaces and blank lines
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
