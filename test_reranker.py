from app.retrieval.reranker import Reranker


query = "What are the limitations of the study?"


results = [

    {
        "chunk_id": "chunk_1",

        "text": """
        While the present study offers important
        theoretical and practical contributions,
        several limitations must be acknowledged.
        The sample was composed of Italian consumers,
        which limits cross-cultural generalizability.
        """,

        "section":
        "5.3 Limitations and future research",

        "page_start": 16,

        "page_end": 17
    },


    {
        "chunk_id": "chunk_2",

        "text": """
        Data were collected through an online survey
        administered via Qualtrics. Participants were
        randomly assigned to experimental conditions.
        """,

        "section":
        "4.3.1 Participants and design",

        "page_start": 8,

        "page_end": 9
    },


    {
        "chunk_id": "chunk_3",

        "text": """
        Virtual Try-On technologies can enhance
        consumer interaction with products by allowing
        consumers to visualize products digitally.
        """,

        "section":
        "1. Introduction",

        "page_start": 1,

        "page_end": 2
    }

]


print("\n")
print("=" * 80)
print("RERANKER TEST")
print("=" * 80)

print(
    f"\nQuery: {query}"
)


reranker = Reranker()


results = reranker.rerank(
    query,
    results,
    top_k=3
)


for rank, result in enumerate(
    results,
    start=1
):

    print("\n" + "-" * 80)

    print(
        f"Rank: {rank}"
    )

    print(
        f"Chunk: "
        f"{result['chunk_id']}"
    )

    print(
        f"Section: "
        f"{result['section']}"
    )

    print(
        f"Pages: "
        f"{result['page_start']}-"
        f"{result['page_end']}"
    )

    print(
        f"Reranker Score: "
        f"{result['reranker_score']:.4f}"
    )