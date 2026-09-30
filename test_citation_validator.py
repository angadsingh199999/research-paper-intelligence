from app.analysis.citation_validator import (
    CitationValidator
)


print()
print("=" * 80)
print("CITATION VALIDATOR TEST")
print("=" * 80)


# ============================================================
# REAL EVIDENCE
# ============================================================

evidence = [

    {
        "chunk_id": "paper1_chunk_0021",
        "paper_id": "paper1",
        "paper_title": (
            "Innovation, signaling, and virtual try-on: "
            "aligning immersive tools with a luxury brand"
        ),
        "section": (
            "5.3 Limitations and future research"
        ),
        "page_start": 16,
        "page_end": 17,
        "text": (
            "While the present study offers important "
            "theoretical and practical contributions, "
            "several limitations must be acknowledged."
        )
    },

    {
        "chunk_id": "paper1_chunk_0009",
        "paper_id": "paper1",
        "paper_title": (
            "Innovation, signaling, and virtual try-on: "
            "aligning immersive tools with a luxury brand"
        ),
        "section": (
            "4.2 Experimental design approach"
        ),
        "page_start": 7,
        "page_end": 8,
        "text": (
            "Two between-subjects experimental studies "
            "were conducted."
        )
    }
]


# ============================================================
# TEST 1 — VALID CITATION
# ============================================================

valid_citation = {

    "paper_id": "paper1",

    "paper_title": (
        "Innovation, signaling, and virtual try-on: "
        "aligning immersive tools with a luxury brand"
    ),

    "page_start": 16,

    "page_end": 17,

    "section": (
        "5.3 Limitations and future research"
    ),

    "chunk_id": "paper1_chunk_0021"
}


# ============================================================
# TEST 2 — HALLUCINATED CITATION
# ============================================================

invalid_citation = {

    "paper_id": "paper5",

    "paper_title": (
        "Completely Invented Research Paper"
    ),

    "page_start": 123,

    "page_end": 145,

    "section": "Methodology",

    "chunk_id": "paper5_chunk_9999"
}


validator = CitationValidator()


result = validator.validate(
    [
        valid_citation,
        invalid_citation
    ],
    evidence
)


print()
print("VALID CITATIONS")
print("-" * 80)

for citation in result["valid"]:
    print(citation)


print()
print("INVALID CITATIONS")
print("-" * 80)

for citation in result["invalid"]:
    print(citation)


print()
print("ALL VALID:", result["all_valid"])

print()
print("=" * 80)
print("TEST COMPLETE")
print("=" * 80)
