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


PDF_PATH = "data/raw_papers/paper1.pdf"


# ============================================================
# 1. INGEST PAPER
# ============================================================

print("\nIngesting paper...")

paper = ingest_paper(
    PDF_PATH
)


# ============================================================
# 2. CREATE CHUNKS
# ============================================================

print("\nCreating chunks...")

chunks = create_chunks(
    paper
)

print(
    f"Total chunks: {len(chunks)}"
)


# ============================================================
# 3. BUILD BM25 INDEX
# ============================================================

print("\nBuilding BM25 index...")

bm25_retriever = BM25Retriever(
    chunks
)

print(
    "BM25 index ready."
)


# ============================================================
# 4. CONNECT TO VECTOR RETRIEVER
# ============================================================

print("\nConnecting to vector retriever...")

vector_retriever = VectorRetriever(
    top_k=5
)

print(
    "Vector retriever ready."
)


# ============================================================
# 5. CREATE HYBRID RETRIEVER
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
# 6. TEST QUERIES
# ============================================================

queries = [

    "What is Virtual Try-On?",

    "What methodology did the researchers use?",

    "What are the limitations of the study?",

    "What was Cohen's d?"

]


# ============================================================
# 7. RUN HYBRID SEARCH
# ============================================================

for query in queries:

    print("\n")
    print("=" * 80)

    print(
        f"QUERY: {query}"
    )

    print("=" * 80)

    results = hybrid_retriever.search(
        query,
        top_k=5
    )

    for rank, result in enumerate(
        results,
        start=1
    ):

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

        print("\nTEXT:")

        print(
            result["text"][:1000]
        )