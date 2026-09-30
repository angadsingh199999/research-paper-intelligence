from app.llm.context_builder import ContextBuilder


results = [

    {
        "chunk_id": "paper1_chunk_0021",
        "paper_id": "paper1",
        "paper_title": (
            "Innovation, signaling, and virtual "
            "try-on: aligning immersive tools "
            "with a luxury brand"
        ),
        "section": (
            "5.3 Limitations and future research"
        ),
        "page_start": 16,
        "page_end": 17,
        "text": (
            "While the present study offers "
            "important theoretical and practical "
            "contributions, several limitations "
            "must be acknowledged."
        )
    },

    {
        "chunk_id": "paper1_chunk_0009",
        "paper_id": "paper1",
        "paper_title": (
            "Innovation, signaling, and virtual "
            "try-on: aligning immersive tools "
            "with a luxury brand"
        ),
        "section": (
            "4.2 Experimental design approach"
        ),
        "page_start": 7,
        "page_end": 8,
        "text": (
            "Two between-subjects experimental "
            "studies were conducted."
        )
    }
]


builder = ContextBuilder()

context = builder.build(results)


print("=" * 80)
print("CONTEXT BUILDER TEST")
print("=" * 80)

print(context)
