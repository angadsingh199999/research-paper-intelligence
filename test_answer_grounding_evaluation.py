from app.ingestion.ingest import ingest_paper
from app.indexing.chunker import create_chunks

from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.task_retriever import TaskSpecificRetriever

from app.analysis.claim_grounder import ClaimGrounder
from app.analysis.qa_analyzer import QAAnalyzer


PDF_PATH = "data/raw_papers/paper1.pdf"


QUESTIONS = [
    "What methodology did the researchers use?",
    "Who participated in the study?",
    "How did the researchers collect the data?",
    "What were the main findings?",
    "What are the limitations of the study?",
]


# ============================================================
# HELPERS
# ============================================================

def get_claim_text(claim):
    if isinstance(claim, dict):
        return str(
            claim.get(
                "claim",
                claim.get("text", "")
            )
        ).strip()

    return str(
        getattr(
            claim,
            "claim",
            str(claim)
        )
    ).strip()


def evaluate_grounding(result):
    answer = str(
        result.get(
            "answer",
            ""
        )
    ).strip()

    claims = result.get(
        "claims",
        []
    )

    evidence = result.get(
        "evidence",
        []
    )

    citations = result.get(
        "citations",
        []
    )

    hallucination = result.get(
        "hallucination",
        {}
    )

    confidence = result.get(
        "confidence",
        {}
    )

    checks = {
        "answer_exists": bool(answer),
        "claims_exist": bool(claims),
        "evidence_exists": bool(evidence),
        "citations_exist": bool(citations),
        "hallucination_check_available": bool(
            hallucination
        ),
        "confidence_available": bool(
            confidence
        ),
    }

    supported = hallucination.get(
        "supported",
        None
    )

    hallucination_detected = hallucination.get(
        "hallucination_detected",
        None
    )

    if supported is not None:
        checks["answer_supported"] = bool(
            supported
        )

    if hallucination_detected is not None:
        checks["no_hallucination"] = (
            not bool(hallucination_detected)
        )

    total_checks = len(checks)

    passed_checks = sum(
        1
        for value in checks.values()
        if value is True
    )

    grounding_score = (
        (passed_checks / total_checks) * 100
        if total_checks
        else 0.0
    )

    return {
        "answer": answer,
        "claim_count": len(claims),
        "evidence_count": len(evidence),
        "citation_count": len(citations),
        "checks": checks,
        "passed_checks": passed_checks,
        "total_checks": total_checks,
        "grounding_score": grounding_score,
    }


# ============================================================
# STEP 1: INGESTION
# ============================================================

print("\n" + "=" * 80)
print("PART 21 - ANSWER / GROUNDING EVALUATION")
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
# STEP 2: CHUNKING
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
# STEP 3: RETRIEVAL PIPELINE
# ============================================================

print("\n" + "=" * 80)
print("STEP 3: INITIALIZING RETRIEVERS")
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
# STEP 4: CLAIM GROUNDER
# ============================================================

print("\n" + "=" * 80)
print("STEP 4: CLAIM GROUNDING")
print("=" * 80)

claim_grounder = ClaimGrounder()

print(
    "Claim grounding pipeline ready."
)


# ============================================================
# STEP 5: QA ANALYZER
# ============================================================

print("\n" + "=" * 80)
print("STEP 5: QA ANALYZER")
print("=" * 80)

qa = QAAnalyzer(
    task_retriever,
    claim_grounder
)

print(
    "QAAnalyzer ready."
)


# ============================================================
# STEP 6: EVALUATE QUESTIONS
# ============================================================

evaluation_results = []


for index, question in enumerate(
    QUESTIONS,
    start=1
):

    print("\n" + "=" * 80)
    print(
        f"QUESTION {index}"
    )
    print("=" * 80)

    print(
        "Question:",
        question
    )

    try:

        result = qa.answer(
            question
        )

        evaluation = evaluate_grounding(
            result
        )

        evaluation_results.append(
            evaluation
        )

        print("\nANSWER")
        print("-" * 80)

        print(
            evaluation["answer"]
        )

        print("\nGROUNDING INFORMATION")
        print("-" * 80)

        print(
            "Claims:",
            evaluation["claim_count"]
        )

        print(
            "Evidence:",
            evaluation["evidence_count"]
        )

        print(
            "Citations:",
            evaluation["citation_count"]
        )

        print("\nCHECKS")
        print("-" * 80)

        for name, passed in evaluation[
            "checks"
        ].items():

            status = (
                "[PASS]"
                if passed
                else "[FAIL]"
            )

            print(
                f"{status} {name}"
            )

        print("\nGROUNDING SCORE")
        print("-" * 80)

        print(
            f"{evaluation['grounding_score']:.2f}%"
        )

    except Exception as error:

        print(
            "\n[ERROR] Question evaluation failed:"
        )

        print(
            str(error)
        )


# ============================================================
# STEP 7: SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("ANSWER / GROUNDING EVALUATION SUMMARY")
print("=" * 80)

total_questions = len(
    evaluation_results
)

answers_available = sum(
    1
    for item in evaluation_results
    if item["checks"].get(
        "answer_exists",
        False
    )
)

grounded_answers = sum(
    1
    for item in evaluation_results
    if item["checks"].get(
        "answer_supported",
        False
    )
)

citation_answers = sum(
    1
    for item in evaluation_results
    if item["checks"].get(
        "citations_exist",
        False
    )
)

hallucination_free = sum(
    1
    for item in evaluation_results
    if item["checks"].get(
        "no_hallucination",
        False
    )
)

print(
    "Questions evaluated:",
    total_questions
)

print(
    "Answers generated:",
    answers_available
)

print(
    "Grounded answers:",
    grounded_answers
)

print(
    "Answers with citations:",
    citation_answers
)

print(
    "Hallucination-free answers:",
    hallucination_free
)


# ============================================================
# STEP 8: AVERAGE GROUNDING SCORE
# ============================================================

if evaluation_results:

    average_score = (
        sum(
            item["grounding_score"]
            for item in evaluation_results
        )
        / len(evaluation_results)
    )

else:

    average_score = 0.0


print(
    f"Average grounding score: "
    f"{average_score:.2f}%"
)


# ============================================================
# STEP 9: FINAL VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("PART 21 PIPELINE VALIDATION")
print("=" * 80)

if total_questions > 0:

    print(
        "[PASS] Questions evaluated"
    )

else:

    print(
        "[FAIL] No questions evaluated"
    )


if answers_available == total_questions:

    print(
        "[PASS] All questions produced answers"
    )

else:

    print(
        "[FAIL] Some questions did not produce answers"
    )


if grounded_answers == total_questions:

    print(
        "[PASS] All answers are grounded"
    )

else:

    print(
        "[WARNING] Not all answers were confirmed as grounded"
    )


if citation_answers == total_questions:

    print(
        "[PASS] All answers contain citations"
    )

else:

    print(
        "[WARNING] Some answers do not contain citations"
    )


if hallucination_free == total_questions:

    print(
        "[PASS] No hallucinations detected"
    )

else:

    print(
        "[WARNING] Hallucination detected or unavailable"
    )


print("\n" + "=" * 80)
print("PART 21 - ANSWER / GROUNDING EVALUATION COMPLETE")
print("=" * 80)
