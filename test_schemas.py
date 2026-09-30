from app.models.schemas import (
    Citation,
    MethodologyAnalysis,
    DatasetAnalysis,
    ModelAnalysis,
    LimitationAnalysis
)


print("\n")
print("=" * 80)
print("PYDANTIC SCHEMA TEST")
print("=" * 80)


# ============================================================
# Citation
# ============================================================

citation = Citation(

    paper_id="paper1",

    paper_title=(
        "Innovation, signaling, and "
        "virtual try-on: aligning "
        "immersive tools with a luxury brand"
    ),

    page_start=7,

    page_end=8,

    section=(
        "4.2 Experimental design approach"
    ),

    chunk_id="paper1_chunk_0009"
)


print("\nCitation:")
print(citation)


# ============================================================
# Methodology
# ============================================================

methodology = MethodologyAnalysis(

    methodology_type="Experimental",

    research_design=(
        "Two between-subjects experimental studies"
    ),

    studies="Study 1 and Study 2",

    participants=(
        "Consumers recruited in Italy"
    ),

    sampling_method=(
        "Convenience sampling and snowball sampling"
    ),

    data_collection=(
        "Online survey using Qualtrics"
    ),

    analysis_method=(
        "Statistical analysis of experimental data"
    ),

    key_methodological_details=[

        "Two experimental studies",

        "Clothing and footwear categories",

        "Brand type was manipulated",

        "Participants were randomly assigned"
    ],

    citations=[citation]
)


print("\n")
print("Methodology Analysis:")
print(methodology)


# ============================================================
# JSON output
# ============================================================

print("\n")
print("JSON:")
print(
    methodology.model_dump_json(
        indent=4
    )
)