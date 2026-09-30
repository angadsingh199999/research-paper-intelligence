from app.ingestion.ingest import ingest_paper
from app.indexing.chunker import create_chunks

from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.task_retriever import TaskSpecificRetriever


# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATH = "data/raw_papers/paper1.pdf"

TOP_K = 5


# ============================================================
# RETRIEVAL TEST CASES
# ============================================================

TEST_CASES = [
    {
        "name": "Methodology",
        "query": "What methodology did the researchers use?",
        "expected_terms": [
            "between-subjects",
            "experimental",
            "Virtual Try-On",
            "Qualtrics",
        ],
    },
    {
        "name": "Participants",
        "query": "Who participated in the study?",
        "expected_terms": [
            "participants",
            "sample",
            "responses",
            "Italy",
        ],
    },
    {
        "name": "Data Collection",
        "query": "How was the data collected?",
        "expected_terms": [
            "Qualtrics",
            "online survey",
            "questionnaire",
        ],
    },
    {
        "name": "Results",
        "query": "What were the main findings of the study?",
        "expected_terms": [
            "results",
            "findings",
            "hypothesis",
        ],
    },
    {
        "name": "Limitations",
        "query": "What limitations did the researchers identify?",
        "expected_terms": [
            "limitations",
            "future research",
            "generalizability",
        ],
    },
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_text(item):
    """
    Safely extract chunk text from different
    possible chunk representations.
    """

    if isinstance(item, str):
        return item

    if isinstance(item, dict):

        return str(
            item.get(
                "text",
                item.get(
                    "content",
                    ""
                )
            )
        )

    if hasattr(item, "text"):

        return str(
            getattr(
                item,
                "text",
                ""
            )
        )

    if hasattr(item, "content"):

        return str(
            getattr(
                item,
                "content",
                ""
            )
        )

    return str(item)


def extract_chunk_id(item):
    """
    Safely extract chunk ID.
    """

    if isinstance(item, dict):

        return str(
            item.get(
                "chunk_id",
                item.get(
                    "id",
                    ""
                )
            )
        )

    if hasattr(item, "chunk_id"):

        return str(
            getattr(
                item,
                "chunk_id",
                ""
            )
        )

    if hasattr(item, "id"):

        return str(
            getattr(
                item,
                "id",
                ""
            )
        )

    return ""


def normalize(text):
    return " ".join(
        str(text)
        .lower()
        .split()
    )


def contains_expected_term(
    text,
    expected_terms
):
    """
    Returns True if at least one expected
    retrieval term appears in the chunk.
    """

    normalized_text = normalize(text)

    for term in expected_terms:

        if normalize(term) in normalized_text:

            return True

    return False


def evaluate_results(
    results,
    expected_terms
):

    if results is None:

        return {
            "retrieved": 0,
            "relevant": 0,
            "hit": False,
            "first_relevant_rank": None,
        }

    try:

        results = list(results)

    except TypeError:

        results = []

    relevant_positions = []

    for index, item in enumerate(
        results,
        start=1
    ):

        text = extract_text(item)

        if contains_expected_term(
            text,
            expected_terms
        ):

            relevant_positions.append(
                index
            )

    return {
        "retrieved": len(results),
        "relevant": len(
            relevant_positions
        ),
        "hit": bool(
            relevant_positions
        ),
        "first_relevant_rank":
            min(relevant_positions)
            if relevant_positions
            else None,
    }


# ============================================================
# 1. INGESTION
# ============================================================

print("\n" + "=" * 80)
print("PART 20 - RETRIEVAL EVALUATION")
print("=" * 80)

print("\nSTEP 1: INGESTION")

paper = ingest_paper(
    PDF_PATH
)

print(
    "Paper:",
    paper.metadata.title
)

print(
    "Pages:",
    paper.metadata.total_pages
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
    "Chunks:",
    len(chunks)
)


# ============================================================
# 3. RETRIEVERS
# ============================================================

print("\n" + "=" * 80)
print("STEP 3: INITIALIZING RETRIEVERS")
print("=" * 80)

bm25 = BM25Retriever(
    chunks
)

print(
    "BM25 retriever initialized."
)


vector = VectorRetriever(
    top_k=TOP_K
)

print(
    "Vector retriever initialized."
)


hybrid = HybridRetriever(
    bm25,
    vector
)

print(
    "Hybrid retriever initialized."
)


task_retriever = TaskSpecificRetriever(
    hybrid
)

print(
    "Task-specific retriever initialized."
)


# ============================================================
# 4. RETRIEVAL EVALUATION
# ============================================================

all_results = []


for test_index, test_case in enumerate(
    TEST_CASES,
    start=1
):

    name = test_case["name"]
    query = test_case["query"]
    expected_terms = test_case[
        "expected_terms"
    ]

    print("\n" + "=" * 80)
    print(
        f"RETRIEVAL TEST {test_index}: {name}"
    )
    print("=" * 80)

    print(
        "Query:",
        query
    )

    print(
        "Expected terms:",
        ", ".join(expected_terms)
    )


    # ========================================================
    # BM25
    # ========================================================

    print("\n" + "-" * 80)
    print("BM25 RETRIEVAL")
    print("-" * 80)

    try:

        bm25_results = bm25.retrieve(
            query
        )

    except Exception as exc:

        print(
            "BM25 retrieval error:",
            str(exc)
        )

        bm25_results = []

    bm25_eval = evaluate_results(
        bm25_results,
        expected_terms
    )

    print(
        "Retrieved:",
        bm25_eval["retrieved"]
    )

    print(
        "Relevant:",
        bm25_eval["relevant"]
    )

    print(
        "Hit:",
        bm25_eval["hit"]
    )

    print(
        "First relevant rank:",
        bm25_eval[
            "first_relevant_rank"
        ]
    )


    # ========================================================
    # VECTOR
    # ========================================================

    print("\n" + "-" * 80)
    print("VECTOR RETRIEVAL")
    print("-" * 80)

    try:

        vector_results = vector.retrieve(
            query,
            chunks
        )

    except TypeError:

        try:

            vector_results = vector.retrieve(
                query
            )

        except Exception as exc:

            print(
                "Vector retrieval error:",
                str(exc)
            )

            vector_results = []

    except Exception as exc:

        print(
            "Vector retrieval error:",
            str(exc)
        )

        vector_results = []

    vector_eval = evaluate_results(
        vector_results,
        expected_terms
    )

    print(
        "Retrieved:",
        vector_eval["retrieved"]
    )

    print(
        "Relevant:",
        vector_eval["relevant"]
    )

    print(
        "Hit:",
        vector_eval["hit"]
    )

    print(
        "First relevant rank:",
        vector_eval[
            "first_relevant_rank"
        ]
    )


    # ========================================================
    # HYBRID
    # ========================================================

    print("\n" + "-" * 80)
    print("HYBRID RETRIEVAL")
    print("-" * 80)

    try:

        hybrid_results = hybrid.retrieve(
            query
        )

    except Exception as exc:

        print(
            "Hybrid retrieval error:",
            str(exc)
        )

        hybrid_results = []

    hybrid_eval = evaluate_results(
        hybrid_results,
        expected_terms
    )

    print(
        "Retrieved:",
        hybrid_eval["retrieved"]
    )

    print(
        "Relevant:",
        hybrid_eval["relevant"]
    )

    print(
        "Hit:",
        hybrid_eval["hit"]
    )

    print(
        "First relevant rank:",
        hybrid_eval[
            "first_relevant_rank"
        ]
    )


    # ========================================================
    # TASK-SPECIFIC RETRIEVAL
    # ========================================================

    print("\n" + "-" * 80)
    print("TASK-SPECIFIC RETRIEVAL")
    print("-" * 80)

    try:

        task_results = (
            task_retriever.retrieve(
                query
            )
        )

    except Exception as exc:

        print(
            "Task-specific retrieval error:",
            str(exc)
        )

        task_results = []

    task_eval = evaluate_results(
        task_results,
        expected_terms
    )

    print(
        "Retrieved:",
        task_eval["retrieved"]
    )

    print(
        "Relevant:",
        task_eval["relevant"]
    )

    print(
        "Hit:",
        task_eval["hit"]
    )

    print(
        "First relevant rank:",
        task_eval[
            "first_relevant_rank"
        ]
    )


    # ========================================================
    # STORE RESULTS
    # ========================================================

    all_results.append(
        {
            "name": name,
            "bm25": bm25_eval,
            "vector": vector_eval,
            "hybrid": hybrid_eval,
            "task_specific": task_eval,
        }
    )


# ============================================================
# 5. AGGREGATE METRICS
# ============================================================

print("\n" + "=" * 80)
print("RETRIEVAL EVALUATION SUMMARY")
print("=" * 80)


retriever_names = [
    "bm25",
    "vector",
    "hybrid",
    "task_specific",
]


metrics = {}


for retriever_name in retriever_names:

    total = len(
        all_results
    )

    hits = 0

    ranks = []

    for result in all_results:

        evaluation = result[
            retriever_name
        ]

        if evaluation["hit"]:

            hits += 1

            rank = evaluation[
                "first_relevant_rank"
            ]

            if rank is not None:

                ranks.append(
                    rank
                )


    hit_rate = (
        hits / total
        if total
        else 0.0
    )

    average_rank = (
        sum(ranks) / len(ranks)
        if ranks
        else None
    )

    metrics[
        retriever_name
    ] = {
        "hit_rate": hit_rate,
        "average_rank": average_rank,
    }


# ============================================================
# 6. DISPLAY AGGREGATE RESULTS
# ============================================================

print(
    "\nRetriever Performance:"
)

print(
    "-" * 80
)

for retriever_name in retriever_names:

    metric = metrics[
        retriever_name
    ]

    hit_rate = metric[
        "hit_rate"
    ]

    average_rank = metric[
        "average_rank"
    ]

    print(
        f"\n{retriever_name.upper()}"
    )

    print(
        f"  Hit rate: "
        f"{hit_rate * 100:.2f}%"
    )

    if average_rank is not None:

        print(
            f"  Average first relevant rank: "
            f"{average_rank:.2f}"
        )

    else:

        print(
            "  Average first relevant rank: N/A"
        )


# ============================================================
# 7. OVERALL VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("RETRIEVAL VALIDATION")
print("=" * 80)


validation = {}


for retriever_name in retriever_names:

    validation[
        retriever_name
    ] = (
        metrics[
            retriever_name
        ]["hit_rate"] > 0
    )


all_passed = all(
    validation.values()
)


for retriever_name, passed in validation.items():

    status = (
        "PASS"
        if passed
        else "FAIL"
    )

    print(
        f"[{status}] "
        f"{retriever_name.upper()} "
        f"retrieval produced relevant results"
    )


# ============================================================
# 8. FINAL RESULT
# ============================================================

print("\n" + "=" * 80)

if all_passed:

    print(
        "RETRIEVAL EVALUATION PASSED"
    )

    print(
        "All retrieval components returned "
        "relevant results for the test queries."
    )

else:

    print(
        "RETRIEVAL EVALUATION REQUIRES REVIEW"
    )

    print(
        "One or more retrieval components "
        "did not return relevant results."
    )

print("=" * 80)
