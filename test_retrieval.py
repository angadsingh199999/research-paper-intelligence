from app.retrieval.vector_retriever import (
    VectorRetriever
)


retriever = VectorRetriever(
    top_k=5
)


queries = [

    "What is Virtual Try-On?",

    "What methodology did the researchers use?",

    "What are the limitations of the study?"

]


for query in queries:

    print("\n")
    print("=" * 80)

    print(
        f"QUERY: {query}"
    )

    print("=" * 80)

    results = retriever.search(
        query
    )

    for rank, result in enumerate(
        results,
        start=1
    ):

        metadata = (
            result["metadata"]
        )

        print("\n" + "-" * 80)

        print(
            f"RANK: {rank}"
        )

        print(
            f"Chunk ID: "
            f"{result['chunk_id']}"
        )

        print(
            f"Section: "
            f"{metadata['section']}"
        )

        print(
            f"Pages: "
            f"{metadata['page_start']}-"
            f"{metadata['page_end']}"
        )

        print(
            f"Distance: "
            f"{result['distance']:.4f}"
        )

        print("\nTEXT:")

        print(
            result["text"][:1000]
        )