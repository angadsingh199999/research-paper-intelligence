from app.ingestion.ingest import ingest_paper
from app.indexing.chunker import create_chunks

from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.task_retriever import TaskSpecificRetriever

from app.analysis.dataset_analyzer import DatasetAnalyzer


# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATH = "data/raw_papers/paper1.pdf"


# ============================================================
# 1. INGESTION
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
# 2. CHUNKING
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
# 3. BM25
# ============================================================

print("\n" + "=" * 80)
print("STEP 3: BM25")
print("=" * 80)

bm25_retriever = BM25Retriever(
    chunks
)

print("BM25 ready.")


# ============================================================
# 4. VECTOR RETRIEVAL
# ============================================================

print("\n" + "=" * 80)
print("STEP 4: VECTOR RETRIEVAL")
print("=" * 80)

vector_retriever = VectorRetriever()

print("Vector retriever ready.")


# ============================================================
# 5. HYBRID RETRIEVAL
# ============================================================

print("\n" + "=" * 80)
print("STEP 5: HYBRID RETRIEVAL")
print("=" * 80)

hybrid_retriever = HybridRetriever(
    vector_retriever,
    bm25_retriever
)

print("Hybrid retriever ready.")


# ============================================================
# 6. TASK RETRIEVER
# ============================================================

print("\n" + "=" * 80)
print("STEP 6: TASK-SPECIFIC RETRIEVAL")
print("=" * 80)

task_retriever = TaskSpecificRetriever(
    hybrid_retriever
)

print("Task-specific retriever ready.")


# ============================================================
# 7. RETRIEVE DATASET EVIDENCE
# ============================================================

query = (
    "What datasets, data sources, "
    "and samples were used?"
)

print("\n" + "=" * 80)
print("STEP 7: RETRIEVING DATASET EVIDENCE")
print("=" * 80)

results = task_retriever.search(
    query=query,
    task="datasets",
    top_k=5
)

print(
    f"Evidence chunks retrieved: "
    f"{len(results)}"
)

for index, result in enumerate(
    results,
    start=1
):

    print("\n" + "-" * 80)

    print(
        f"EVIDENCE {index}"
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
        f"{result.get('page_start')}-"
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
# 8. DATASET ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("STEP 8: QWEN3 DATASET ANALYSIS")
print("=" * 80)

analyzer = DatasetAnalyzer()

response = analyzer.analyze(
    results
)


# ============================================================
# 9. FINAL RESULT
# ============================================================

print("\n" + "=" * 80)
print("FINAL DATASET ANALYSIS")
print("=" * 80)

print(response)
