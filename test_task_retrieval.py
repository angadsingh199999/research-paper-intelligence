from app.ingestion.ingest import ingest_paper
from app.indexing.chunker import create_chunks

from app.retrieval.vector_retriever import (
    VectorRetriever
)

from app.retrieval.bm25_retriever import (
    BM25Retriever
)

from app.retrieval.hybrid_retriever import (
    HybridRetriever
)

from app.retrieval.task_retriever import (
    TaskSpecificRetriever
)


PDF_PATH = "data/raw_papers/paper1.pdf"


# ============================================================
# INGEST
# ============================================================

print("\nIngesting paper...")

paper = ingest_paper(
    PDF_PATH
)


# ============================================================
# CHUNKS
# ============================================================

print("\nCreating chunks...")

chunks = create_chunks(
    paper
)

print(
    f"Total chunks: {len(chunks)}"
)


# ============================================================
# BM25
# ============================================================

print("\nBuilding BM25...")

bm25 = BM25Retriever(
    chunks
)


# ============================================================
# VECTOR
# ============================================================

print("\nBuilding Vector Retriever...")

vector = VectorRetriever(
    top_k=15
)


# ============================================================
# HYBRID
# ============================================================

print("\nBuilding Hybrid Retriever...")

hybrid = HybridRetriever(
    vector_retriever=vector,
    bm25_retriever=bm25,
    rrf_k=60
)


# ============================================================
# TASK RETRIEVER
# ============================================================

print("\nBuilding Task-Specific Retriever...")

task_retriever = TaskSpecificRetriever(
    hybrid
)


# ============================================================
# TEST CASES
# ============================================================

tests = [

    (
        "What are the limitations of the study?",
        "limitations"
    ),

    (
        "What methodology did the researchers use?",
        "methodology"
    ),

    (
        "What future research did the authors suggest?",
        "future_work"
    ),

    (
        "What was the experimental design?",
        "methodology"
    )

]


# ============================================================
# RUN TESTS
# ============================================================

for query, task in tests:

    print("\n")
    print("=" * 80)

    print(
        f"QUERY: {query}"
    )

    print(
        f"TASK: {task}"
    )

    print("=" * 80)


    results = task_retriever.search(
        query,
        task,
        top_k=5
    )


    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            "\n" + "-" * 80
        )

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
            f"Task Score: "
            f"{result['task_score']}"
        )

        print(
            f"RRF Score: "
            f"{result['rrf_score']:.6f}"
        )