import re
from pathlib import Path


def extract_metadata(pages, pdf_path):
    """
    Extract basic metadata from the first page of a research paper.

    Returns:
        Dictionary containing paper metadata.
    """

    filename = Path(pdf_path).name
    paper_id = Path(pdf_path).stem

    if not pages:
        return {
            "paper_id": paper_id,
            "filename": filename,
            "title": "",
            "authors": [],
            "total_pages": 0,
        }

    first_page_text = pages[0]["text"] if isinstance(pages[0], dict) else getattr(pages[0], "text", "")

    lines = [
        line.strip()
        for line in first_page_text.splitlines()
        if line.strip()
    ]

    title = ""
    authors = []

    # --------------------------------------------------------
    # Find the Abstract or Article Info boundary
    # --------------------------------------------------------
    abstract_index = None

    for i, line in enumerate(lines):
        normalized = line.strip().lower()
        collapsed = re.sub(r"\s+", "", normalized)

        if (
            collapsed == "abstract"
            or normalized == "abstract"
            or re.match(r"^abstract\b", normalized)
            or re.match(r"^a\s*b\s*s\s*t\s*r\s*a\s*c\s*t\b", normalized)
        ):
            abstract_index = i
            break

    # If abstract not found, check for "article info"
    if abstract_index is None:
        for i, line in enumerate(lines):
            collapsed = re.sub(r"\s+", "", line.strip().lower())
            if collapsed in ("articleinfo", "keywords:"):
                abstract_index = i
                break

    if abstract_index is not None and abstract_index > 0:
        header_lines = lines[:abstract_index]

        # Filter out obvious journal / publisher lines from header
        candidate_lines = []
        for hline in header_lines:
            low = hline.lower()
            if any(term in low for term in ("received", "accepted", "doi:", "issn", "volume", "vol.", "emerald", "elsevier", "sustainable futures")):
                continue
            candidate_lines.append(hline)

        # Look for article info boundary within candidate lines
        article_info_idx = None
        for idx, cl in enumerate(candidate_lines):
            collapsed = re.sub(r"\s+", "", cl.lower())
            if collapsed in ("articleinfo", "keywords:"):
                article_info_idx = idx
                break
        if article_info_idx is not None:
            candidate_lines = candidate_lines[:article_info_idx]

        # Case A: Standard two-line / multi-line header before Abstract
        if len(candidate_lines) >= 2:
            # Check if lines have affiliations (marked with 'department of', 'university', etc.)
            affiliation_start = None
            for idx, cl in enumerate(candidate_lines):
                cl_low = cl.lower()
                if any(k in cl_low for k in ("department of", "university", "faculty of", "school of", "centre for", "center for")):
                    affiliation_start = idx
                    break

            if affiliation_start is not None and affiliation_start >= 2:
                # Lines before affiliation are title and authors
                pre_affil = candidate_lines[:affiliation_start]
                # Author line is typically immediately preceding affiliation
                author_candidate = pre_affil[-1]
                if pre_affil[-1].startswith(",") and len(pre_affil) >= 3:
                    author_candidate = f"{pre_affil[-2]} {pre_affil[-1]}"
                    title_lines = pre_affil[:-2]
                else:
                    title_lines = pre_affil[:-1]

                title = " ".join(title_lines).strip()
                cleaned_author = re.sub(r"\b[a-z]\b|\*|\d", "", author_candidate)
                authors = [
                    a.strip()
                    for a in re.split(r",|\band\b", cleaned_author)
                    if len(a.strip()) > 2 and not any(c.isdigit() for c in a)
                ]
            else:
                title = " ".join(candidate_lines[:-2]).strip()
                author_line = candidate_lines[-2]
                authors = [
                    author.strip()
                    for author in re.split(r",|\band\b", author_line)
                    if author.strip()
                ]

    title = title.strip()

    metadata = {
        "paper_id": paper_id,
        "filename": filename,
        "title": title,
        "authors": authors,
        "total_pages": len(pages),
    }

    return metadata
