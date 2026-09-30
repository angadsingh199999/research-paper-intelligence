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

from app.retrieval.reranker import (
    Reranker
)


PDF_PATH = "data/raw_papers/paper1.pdf"


# ============================================================
# 1. INGEST
# ============================================================

print("\nIngesting paper...")

paper = ingest_paper(
    PDF_PATH
)


# ============================================================
# 2. CHUNK
# ============================================================

print("\nCreating chunks...")

chunks = create_chunks(
    paper
)

print(
    f"Total chunks: {len(chunks)}"
)


# ============================================================
# 3. BM25
# ============================================================

print("\nBuilding BM25 index...")

bm25_retriever = BM25Retriever(
    chunks
)

print(
    "BM25 index ready."
)


# ============================================================
# 4. VECTOR
# ============================================================

print("\nCreating vector retriever...")

vector_retriever = VectorRetriever(
    top_k=10
)

print(
    "Vector retriever ready."
)


# ============================================================
# 5. HYBRID
# ============================================================

print("\nCreating hybrid retriever...")

hybrid_retriever = HybridRetriever(
    vector_retriever=vector_retriever,
    bm25_retriever=bm25_retriever,
    rrf_k=60
)

print(
    "Hybrid retriever ready."
)


# ============================================================
# 6. RERANKER
# ============================================================

print("\nCreating reranker...")

reranker = Reranker()

print(
    "Reranker ready."
)


# ============================================================
# 7. QUERY
# ============================================================

query = (
    "What are the limitations of the study?"
)


print("\n")
print("=" * 80)

print(
    f"QUERY: {query}"
)

print("=" * 80)


# ============================================================
# 8. HYBRID RETRIEVAL
# ============================================================

print("\nRunning hybrid retrieval...")

hybrid_results = hybrid_retriever.search(
    query,
    top_k=10
)


print(
    f"Hybrid candidates: "
    f"{len(hybrid_results)}"
)


# ============================================================
# 9. DISPLAY HYBRID RESULTS
# ============================================================

print("\n")
print("=" * 80)
print("HYBRID RESULTS BEFORE RERANKING")
print("=" * 80)


for rank, result in enumerate(
    hybrid_results,
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
        f"RRF Score: "
        f"{result['rrf_score']:.6f}"
    )


# ============================================================
# 10. RERANK
# ============================================================

print("\n")
print(
    "Running Cross-Encoder reranking..."
)

final_results = reranker.rerank(
    query,
    hybrid_results,
    top_k=5
)


# ============================================================
# 11. FINAL RESULTS
# ============================================================

print("\n")
print("=" * 80)
print("FINAL RESULTS AFTER RERANKING")
print("=" * 80)


for rank, result in enumerate(
    final_results,
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
        f"RRF Score: "
        f"{result['rrf_score']:.6f}"
    )

    print(
        f"Reranker Score: "
        f"{result['reranker_score']:.4f}"
    )

    print("\nTEXT:")

    print(
        result["text"][:500]
    )