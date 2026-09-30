import re


# ================================================================
# STATISTICAL NOTATION REPAIR
#
# Some PDFs (particularly journal exports with non-standard math
# fonts) have a broken character map: the glyph for "=" gets
# extracted by pymupdf as a literal digit "5", and the glyph for
# "-" / "−" (minus) gets extracted as "Ø", "\x01", or "‒" (figure dash).
#
# Examples:
#   "F(2, 124) 5 82.96" -> "F(2, 124) = 82.96"
#   "F (2,231) 5 143.16" -> "F(2,231) = 143.16"
#   "t(177) 5 \x014.44" -> "t(177) = -4.44"
#   "t(232) 5 ‒5.61" -> "t(232) = -5.61"
#   "R 2 5 0.553" -> "R² = 0.553"
#
# This is a font-encoding problem in the source PDF. The fix below
# is a targeted heuristic: it only converts " 5 " to " = " when it
# sits directly between a recognized statistics symbol/abbreviation
# and a number (or scale label). Genuine numbers (e.g. "5 participants")
# are strictly preserved.
# ================================================================

_STAT_LABELS = (
    r"M(?:\s?age)?",
    r"SD(?:\s?age)?",
    r"SE",
    r"N",
    r"n",
    r"df\d*",
    r"F(?:\s*\([^)]*\))?",
    r"t(?:\s*\([^)]*\))?",
    r"r",
    r"R\s*[²2]",
    r"p(?:\s*value)?",
    r"β",
    r"beta",
    r"α",
    r"χ²",
    r"chi2",
    r"d",
    r"k",
    r"z",
    r"CI",
    r"ICC",
    r"AVE",
    r"CR",
    r"ω",
    r"ΔR²",
    r"η[pP]²",
    r"factor",
    r"variance",
    r"loading",
    r"correlation",
    r"coefficient",
    r"power",
    r"rate",
    r"effect",
    r"index",
    r"non-luxuryBrand",
    r"luxuryBrand",
)

_STAT_LABEL_GROUP = "|".join(_STAT_LABELS)

# Matches: <stat symbol><spaces>5<spaces>, only when followed by an
# optional minus-artifact or dash and then a digit
_STAT_EQUALS_PATTERN = re.compile(
    rf"\b(?P<label>{_STAT_LABEL_GROUP})\s+5\s+(?=[-–—‒\x01Ø]?\s*\d)",
    flags=re.IGNORECASE,
)

# Scale anchors like "1 5 strongly disagree", "7 5 very high"
_SCALE_EQUALS_PATTERN = re.compile(
    r"\b(?P<val>[1-7])\s+5\s+(?=(?:strongly|very|non-luxury|luxury|agree|disagree|low|high)\b)",
    flags=re.IGNORECASE,
)

# Glyph artifacts representing minus before digits:
# "Ø", "\x01", and Unicode figure dash "‒" (U+2012)
_MINUS_ARTIFACT_PATTERN = re.compile(r"(?:Ø|\x01|‒)(?=\s*\d)")


def _repair_statistical_notation(text):
    text = _STAT_EQUALS_PATTERN.sub(
        lambda m: f"{m.group('label')} = ",
        text,
    )

    text = _SCALE_EQUALS_PATTERN.sub(
        r"\g<val> = ",
        text,
    )

    text = _MINUS_ARTIFACT_PATTERN.sub("-", text)

    # Normalize "R 2" and "R2" to "R²" in statistical contexts
    text = re.sub(r"\bR\s*2\b", "R²", text)

    return text


def clean_text(text):
    """
    Clean extracted PDF text while preserving
    meaningful research content.
    """

    # Replace non-breaking spaces
    text = text.replace("\xa0", " ")

    # --------------------------------------------------------
    # Remove soft hyphens (U+00AD) left over from justified-text
    # line wrapping in the source PDF.
    # --------------------------------------------------------
    text = re.sub(r"\xad\s*", "", text)

    # --------------------------------------------------------
    # Repair font encoding artifacts before whitespace normalization,
    # so statistical label patterns see original spacing.
    # --------------------------------------------------------
    text = _repair_statistical_notation(text)

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces at the beginning/end of lines
    text = "\n".join(line.strip() for line in text.splitlines())

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
