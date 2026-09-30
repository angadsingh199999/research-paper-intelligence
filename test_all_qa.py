from app.ingestion.ingest import ingest_paper

from app.indexing.chunker import create_chunks

from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.task_retriever import TaskSpecificRetriever

from app.analysis.qa_analyzer import QAAnalyzer


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
    f"Chunks: {len(chunks)}"
)


# ============================================================
# 3. RETRIEVERS
# ============================================================

print("\n" + "=" * 80)
print("STEP 3: RETRIEVERS")
print("=" * 80)

bm25_retriever = BM25Retriever(
    chunks
)

vector_retriever = VectorRetriever(
    chunks
)

hybrid_retriever = HybridRetriever(
    bm25_retriever,
    vector_retriever
)

task_retriever = TaskSpecificRetriever(
    hybrid_retriever
)

print(
    "Retrieval pipeline ready."
)


# ============================================================
# 4. QA ANALYZER
# ============================================================

qa = QAAnalyzer(
    task_retriever
)


# ============================================================
# 5. QUESTIONS
# ============================================================

questions = [

    (
        "METHODOLOGY",
        "What methodology did the researchers use?"
    ),

    (
        "DATASETS",
        "What data or datasets did the researchers use?"
    ),

    (
        "MODELS",
        "What models or theoretical models were used?"
    ),

    (
        "RESULTS",
        "What were the main findings of the study?"
    ),

    (
        "LIMITATIONS",
        "What limitations did the authors identify?"
    ),

    (
        "FUTURE WORK",
        "What future research did the authors suggest?"
    ),

    (
        "LITERATURE REVIEW",
        "What does the paper say about previous research?"
    ),

]


# ============================================================
# 6. RUN QUESTIONS
# ============================================================

for label, question in questions:

    print("\n\n")

    print(
        "=" * 80
    )

    print(
        label
    )

    print(
        "=" * 80
    )

    print(
        "\nQUESTION:"
    )

    print(
        question
    )

    result = qa.answer(
        question
    )

    print(
        "\nFINAL ANSWER:"
    )

    print(
        "=" * 80
    )

    print(
        result.get(
            "answer",
            ""
        )
    )

    print(
        "\nTASK:"
    )

    print(
        result.get(
            "task"
        )
    )

    print(
        "\nEVIDENCE COUNT:"
    )

    print(
        len(
            result.get(
                "evidence",
                []
            )
        )
    )

    print(
        "\nCLAIM COUNT:"
    )

    print(
        len(
            result.get(
                "claims",
                []
            )
        )
    )
