from app.analysis.query_decomposer import (
    QueryDecomposer
)


decomposer = QueryDecomposer()


queries = [

    "Compare the methodology and limitations "
    "of these papers.",

    "Compare datasets, models and future "
    "research across the papers.",

    "Give me a complete comparison of "
    "methodology, datasets, models, "
    "limitations and future work.",

    "What methodology did this paper use?"

]


print("\n")
print("=" * 80)

print("QUERY DECOMPOSITION TEST")

print("=" * 80)


for query in queries:

    print("\n")
    print("-" * 80)

    print(
        f"ORIGINAL QUERY:\n{query}"
    )

    print("-" * 80)


    results = decomposer.decompose(
        query
    )


    for i, result in enumerate(
        results,
        start=1
    ):

        print("\n")

        print(
            f"SUBTASK {i}"
        )

        print(
            f"Task: "
            f"{result['task']}"
        )

        print(
            f"Query: "
            f"{result['query']}"
        )