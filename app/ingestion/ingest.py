from pathlib import Path
import re

from app.ingestion.pdf_parser import extract_text_from_pdf
from app.ingestion.metadata_extractor import extract_metadata
from app.ingestion.section_blocks import create_section_blocks
from app.models.schemas import (
    ResearchPaper,
    PaperMetadata,
    Page,
    SectionBlock,
)


def _page_text(page):
    if isinstance(page, dict):
        return str(page.get("text", "") or "")
    return str(getattr(page, "text", "") or "")


def _looks_like_title(line):
    line = re.sub(r"\s+", " ", str(line or "")).strip()
    if not line:
        return False
    if len(line) < 18 or len(line) > 300:
        return False

    low = line.lower()

    reject = (
        "abstract", "keywords", "introduction", "contents",
        "received ", "accepted ", "doi:", "https://", "www.",
        "copyright", "journal", "issn", "volume", "vol.",
        "available online", "article history",
    )
    if any(x in low for x in reject):
        return False

    # Avoid obvious author-only lines and page headers.
    if re.fullmatch(r"[A-Z][A-Za-z .,'-]{2,120}", line):
        words = line.split()
        if len(words) <= 5 and all(
            w[:1].isupper() for w in words if w[:1].isalpha()
        ):
            return False

    return True


def _title_from_first_page(raw_pages):
    if not raw_pages:
        return ""

    text = _page_text(raw_pages[0])
    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in text.splitlines()
    ]
    lines = [x for x in lines if x]

    candidates = []
    for idx, line in enumerate(lines[:35]):
        if not _looks_like_title(line):
            continue

        # Prefer early, long, title-like lines.
        score = 0.0
        score += max(0, 25 - idx)
        if 30 <= len(line) <= 220:
            score += 8
        if ":" in line:
            score += 1
        if line.endswith("."):
            score -= 2

        candidates.append((score, idx, line))

    if not candidates:
        return ""

    candidates.sort(reverse=True)
    best = candidates[0][2]

    # If the title was split over two consecutive lines, combine them
    # only when the second line also looks title-like and is close by.
    best_idx = candidates[0][1]
    if best_idx + 1 < len(lines):
        nxt = lines[best_idx + 1]
        if (
            _looks_like_title(nxt)
            and len(best) < 180
            and len(nxt) < 140
            and "abstract" not in nxt.lower()
        ):
            combined = f"{best} {nxt}".strip()
            if len(combined) <= 300:
                best = combined

    return best.strip()


def ingest_paper(pdf_path):
    pdf_path = str(pdf_path)
    path = Path(pdf_path)

    raw_pages = extract_text_from_pdf(pdf_path)

    metadata = extract_metadata(
        raw_pages,
        pdf_path,
    ) or {}

    paper_id = str(
        metadata.get("paper_id")
        or path.stem
    ).strip()

    filename = str(
        metadata.get("filename")
        or path.name
    ).strip()

    title = str(
        metadata.get("title")
        or ""
    ).strip()

    # PDF metadata can be blank or placeholder. Try first-page text before falling back
    # to the filename.
    if not title or title.lower() in {paper_id.lower(), filename.lower(), path.stem.lower()}:
        first_page_title = _title_from_first_page(raw_pages)
        if first_page_title:
            title = first_page_title

    if not title:
        title = path.stem.replace("_", " ").strip()

    authors = metadata.get("authors") or []

    try:
        total_pages = int(
            metadata.get("total_pages")
            or len(raw_pages)
        )
    except (TypeError, ValueError):
        total_pages = len(raw_pages)

    section_pages = create_section_blocks(raw_pages)

    paper_metadata = PaperMetadata(
        paper_id=paper_id,
        filename=filename,
        title=title,
        authors=authors,
        total_pages=total_pages,
    )

    pages = []
    for page in section_pages:
        blocks = []
        for block in page["blocks"]:
            blocks.append(
                SectionBlock(
                    section=block["section"],
                    text=block["text"],
                    page_number=block["page_number"],
                )
            )

        pages.append(
            Page(
                page_number=page["page_number"],
                blocks=blocks,
            )
        )

    return ResearchPaper(
        metadata=paper_metadata,
        pages=pages,
    )
