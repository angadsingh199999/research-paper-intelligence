from app.ingestion.ingest import ingest_paper

from app.indexing.chunker import create_chunks

from app.retrieval.bm25_retriever import BM25Retriever

from app.retrieval.vector_retriever import VectorRetriever

from app.retrieval.hybrid_retriever import HybridRetriever

from app.retrieval.task_retriever import (
    TaskSpecificRetriever
)

from app.analysis.limitation_analyzer import (
    LimitationAnalyzer
)


# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATH = "data/raw_papers/paper1.pdf"


# ============================================================
# 1. INGEST PAPER
# ============================================================

print("\n" + "=" * 80)

print("STEP 1: INGESTION")

print("=" * 80)

paper = ingest_paper(
    PDF_PATH
)

print(
    f"Paper: {paper.metadata.title}"
)

print(
    f"Pages: {paper.metadata.total_pages}"
)


# ============================================================
# 2. CREATE CHUNKS
# ============================================================

print("\n" + "=" * 80)

print("STEP 2: CHUNKING")

print("=" * 80)

chunks = create_chunks(
    paper
)

print(
    f"Total chunks: {len(chunks)}"
)


# ============================================================
# 3. BUILD BM25
# ============================================================

print("\n" + "=" * 80)

print("STEP 3: BM25")

print("=" * 80)

bm25_retriever = BM25Retriever(
    chunks
)

print(
    "BM25 ready."
)


# ============================================================
# 4. BUILD VECTOR RETRIEVER
# ============================================================

print("\n" + "=" * 80)

print("STEP 4: VECTOR RETRIEVAL")

print("=" * 80)

vector_retriever = VectorRetriever(
    top_k=5
)

print(
    "Vector retriever ready."
)


# ============================================================
# 5. BUILD HYBRID RETRIEVER
# ============================================================

print("\n" + "=" * 80)

print("STEP 5: HYBRID RETRIEVAL")

print("=" * 80)

hybrid_retriever = HybridRetriever(
    bm25_retriever,
    vector_retriever
)

print(
    "Hybrid retriever ready."
)


# ============================================================
# 6. BUILD TASK-SPECIFIC RETRIEVER
# ============================================================

print("\n" + "=" * 80)

print("STEP 6: TASK-SPECIFIC RETRIEVAL")

print("=" * 80)

task_retriever = TaskSpecificRetriever(
    hybrid_retriever
)

print(
    "Task-specific retriever ready."
)


# ============================================================
# 7. RETRIEVE LIMITATION EVIDENCE
# ============================================================

print("\n" + "=" * 80)

print("STEP 7: RETRIEVING LIMITATION EVIDENCE")

print("=" * 80)

query = """
What limitations did the authors identify
and what future research directions did
they propose?
"""

results = task_retriever.search(
    query=query,
    task="limitations",
    top_k=5
)

print(
    f"Evidence chunks retrieved: {len(results)}"
)


for i, result in enumerate(
    results,
    start=1
):

    print("\n" + "-" * 80)

    print(
        f"EVIDENCE {i}"
    )

    print(
        f"Paper ID: "
        f"{result.get('paper_id')}"
    )

    print(
        f"Paper Title: "
        f"{result.get('paper_title')}"
    )

    print(
        f"Section: "
        f"{result.get('section')}"
    )

    print(
        f"Pages: "
        f"{result.get('page_start')} - "
        f"{result.get('page_end')}"
    )

    print(
        f"Chunk ID: "
        f"{result.get('chunk_id')}"
    )

    print(
        f"Task Score: "
        f"{result.get('task_score')}"
    )

    print(
        f"RRF Score: "
        f"{result.get('rrf_score')}"
    )

    print(
        f"Reranker Score: "
        f"{result.get('reranker_score')}"
    )

    print(
        "\nTEXT:\n"
        + result.get(
            "text",
            ""
        )
    )


# ============================================================
# 8. LIMITATION ANALYSIS
# ============================================================

print("\n" + "=" * 80)

print("STEP 8: QWEN LIMITATION ANALYSIS")

print("=" * 80)

analyzer = LimitationAnalyzer()

analysis = analyzer.analyze(
    results
)


# ============================================================
# 9. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 80)

print("FINAL LIMITATION ANALYSIS")

print("=" * 80)

print(
    analysis
)
