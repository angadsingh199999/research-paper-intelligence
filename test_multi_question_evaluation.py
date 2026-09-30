from app.ingestion.ingest import ingest_paper
from app.indexing.chunker import create_chunks

from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.task_retriever import TaskSpecificRetriever

from app.analysis.claim_grounder import ClaimGrounder
from app.analysis.qa_analyzer import QAAnalyzer

from app.evaluation.multi_question_evaluator import (
    MultiQuestionEvaluator
)


PDF_PATH = "data/raw_papers/paper1.pdf"


# ============================================================
# QUESTIONS
# ============================================================

QUESTIONS = [

    "What methodology did the researchers use?",

    "Who participated in the study and how were participants recruited?",

    "What theoretical framework or models were used in the study?",

    "What were the main findings of the study?",

    "What limitations did the researchers identify?",

    "What directions for future research were suggested?",

    "How did the researchers collect the data?",
]


# ============================================================
# 1. INGESTION
# ============================================================

print("\n" + "=" * 80)
print("STEP 1: PDF INGESTION")
print("=" * 80)

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

print(
    "PDF ingestion successful."
)


# ============================================================
# 2. CHUNKING
# ============================================================

print("\n" + "=" * 80)
print("STEP 2: DOCUMENT CHUNKING")
print("=" * 80)

chunks = create_chunks(
    paper
)

print(
    "Chunks created:",
    len(chunks)
)

print(
    "Document chunking successful."
)


# ============================================================
# 3. RETRIEVAL PIPELINE
# ============================================================

print("\n" + "=" * 80)
print("STEP 3: RETRIEVAL PIPELINE")
print("=" * 80)

bm25 = BM25Retriever(
    chunks
)

vector = VectorRetriever(
    top_k=5
)

hybrid = HybridRetriever(
    bm25,
    vector
)

task_retriever = TaskSpecificRetriever(
    hybrid
)

print(
    "Retrieval pipeline ready."
)


# ============================================================
# 4. CLAIM GROUNDING
# ============================================================

print("\n" + "=" * 80)
print("STEP 4: CLAIM GROUNDING")
print("=" * 80)

claim_grounder = ClaimGrounder()

print(
    "Claim grounding pipeline ready."
)


# ============================================================
# 5. QA ANALYZER
# ============================================================

print("\n" + "=" * 80)
print("STEP 5: QA ANALYZER")
print("=" * 80)

qa = QAAnalyzer(
    task_retriever,
    claim_grounder
)

print(
    "QAAnalyzer initialized."
)


# ============================================================
# 6. MULTI-QUESTION EVALUATOR
# ============================================================

print("\n" + "=" * 80)
print("STEP 6: MULTI-QUESTION EVALUATION")
print("=" * 80)

evaluator = MultiQuestionEvaluator(
    qa
)

print(
    "Multi-question evaluator initialized."
)


# ============================================================
# 7. RUN EVALUATION
# ============================================================

evaluation = evaluator.evaluate(
    QUESTIONS
)


# ============================================================
# 8. RESULTS
# ============================================================

print("\n" + "=" * 80)
print("MULTI-QUESTION EVALUATION RESULTS")
print("=" * 80)

results = evaluation.get(
    "results",
    []
)

for index, item in enumerate(
    results,
    start=1
):

    print("\n" + "-" * 80)

    print(
        f"QUESTION {index}"
    )

    print(
        "Question:",
        item.get(
            "question",
            ""
        )
    )

    print(
        "Task:",
        item.get(
            "task",
            ""
        )
    )

    print(
        "Answer:",
        item.get(
            "answer",
            "No answer generated."
        )
    )

    print(
        "Claims:",
        item.get(
            "claim_count",
            0
        )
    )

    print(
        "Evidence:",
        item.get(
            "evidence_count",
            0
        )
    )

    print(
        "Citations:",
        item.get(
            "citation_count",
            0
        )
    )

    confidence = item.get(
        "confidence",
        {}
    )

    print(
        "Confidence score:",
        confidence.get(
            "score",
            0.0
        )
    )

    print(
        "Confidence level:",
        confidence.get(
            "label",
            "LOW"
        )
    )

    hallucination = item.get(
        "hallucination",
        {}
    )

    if hallucination:

        print(
            "Hallucination detected:",
            hallucination.get(
                "hallucination_detected",
                False
            )
        )

        print(
            "Supported:",
            hallucination.get(
                "supported",
                False
            )
        )

    else:

        print(
            "Hallucination result: Not available"
        )

    print(
        "Answer generated:",
        item.get(
            "answer_generated",
            False
        )
    )

    print(
        "Claims returned:",
        item.get(
            "claims_returned",
            False
        )
    )

    print(
        "Evidence returned:",
        item.get(
            "evidence_returned",
            False
        )
    )

    print(
        "Citations returned:",
        item.get(
            "citations_returned",
            False
        )
    )

    print(
        "Confidence returned:",
        item.get(
            "confidence_returned",
            False
        )
    )

    if "error" in item:

        print(
            "ERROR:",
            item.get(
                "error",
                ""
            )
        )


# ============================================================
# 9. SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("EVALUATION SUMMARY")
print("=" * 80)

print(
    "Total questions:",
    evaluation.get(
        "total_questions",
        0
    )
)

print(
    "Successful answers:",
    evaluation.get(
        "successful_answers",
        0
    )
)

print(
    "Grounded answers:",
    evaluation.get(
        "grounded_answers",
        0
    )
)

print(
    "Cited answers:",
    evaluation.get(
        "cited_answers",
        0
    )
)

print(
    "Confidence available:",
    evaluation.get(
        "confidence_available",
        0
    )
)


# ============================================================
# 10. VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("PIPELINE VALIDATION")
print("=" * 80)

total = evaluation.get(
    "total_questions",
    0
)

successful = evaluation.get(
    "successful_answers",
    0
)

grounded = evaluation.get(
    "grounded_answers",
    0
)

cited = evaluation.get(
    "cited_answers",
    0
)

confidence_available = evaluation.get(
    "confidence_available",
    0
)


if total > 0:

    print(
        "[PASS] Questions evaluated"
    )

else:

    print(
        "[FAIL] No questions evaluated"
    )


if successful == total:

    print(
        "[PASS] All questions produced answers"
    )

else:

    print(
        "[WARNING] Some questions did not produce answers"
    )


if grounded == total:

    print(
        "[PASS] All answers returned grounded evidence"
    )

else:

    print(
        "[WARNING] Not all answers returned grounded evidence"
    )


if cited == total:

    print(
        "[PASS] All answers returned citations"
    )

else:

    print(
        "[WARNING] Not all answers returned citations"
    )


if confidence_available == total:

    print(
        "[PASS] Confidence scores available for all questions"
    )

else:

    print(
        "[WARNING] Confidence scores missing for some questions"
    )


# ============================================================
# 11. COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("PART 19 MULTI-QUESTION EVALUATION COMPLETE")
print("=" * 80)

print(
    "Multiple research questions were evaluated "
    "through the existing QA pipeline."
)
