from app.ingestion.ingest import ingest_paper
from app.indexing.chunker import create_chunks

from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.task_retriever import TaskSpecificRetriever

from app.analysis.claim_grounder import ClaimGrounder
from app.analysis.qa_analyzer import QAAnalyzer


PDF_PATH = "data/raw_papers/paper1.pdf"


# ============================================================
# 1. SETUP PIPELINE
# ============================================================

print("\n" + "=" * 80)
print("PART 22 - ADVERSARIAL TESTING")
print("=" * 80)

print("\nLoading paper...")

paper = ingest_paper(PDF_PATH)

print("Paper:", paper.metadata.title)
print("Pages:", paper.metadata.total_pages)


print("\nCreating chunks...")

chunks = create_chunks(paper)

print("Chunks:", len(chunks))


print("\nInitializing retrieval pipeline...")

bm25 = BM25Retriever(chunks)

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

claim_grounder = ClaimGrounder()

qa = QAAnalyzer(
    task_retriever,
    claim_grounder
)

print("QA pipeline ready.")


# ============================================================
# 2. ADVERSARIAL QUESTIONS
# ============================================================

TEST_CASES = [

    {
        "name": "SUPPORTED QUESTION",
        "question": "What methodology did the researchers use?"
    },

    {
        "name": "UNSUPPORTED INFORMATION",
        "question": (
            "What was the exact salary paid to each participant "
            "in the study?"
        )
    },

    {
        "name": "MISLEADING QUESTION",
        "question": (
            "How did the researchers prove that virtual try-on "
            "always increases luxury brand sales?"
        )
    },

    {
        "name": "SPECIFIC CLAIM",
        "question": (
            "What exact software was used to perform all the "
            "statistical analyses?"
        )
    },

    {
        "name": "IRRELEVANT QUESTION",
        "question": (
            "What is the capital city of France?"
        )
    },

    {
        "name": "OUTSIDE DOMAIN",
        "question": (
            "What are the latest stock prices of Apple and Microsoft?"
        )
    },

    {
        "name": "EMPTY QUESTION",
        "question": ""
    },

]


# ============================================================
# 3. RUN ADVERSARIAL TESTS
# ============================================================

results = []


for index, test_case in enumerate(
    TEST_CASES,
    start=1
):

    name = test_case["name"]
    question = test_case["question"]

    print("\n" + "=" * 80)
    print(
        f"TEST {index}: {name}"
    )
    print("=" * 80)

    print(
        "Question:",
        repr(question)
    )

    try:

        result = qa.answer(
            question
        )

        answer = result.get(
            "answer",
            ""
        )

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

        sources = result.get(
            "sources",
            []
        )

        confidence = result.get(
            "confidence",
            {}
        )

        hallucination = result.get(
            "hallucination",
            {}
        )

        print("\nANSWER")
        print("-" * 80)
        print(answer)

        print("\nCLAIM COUNT:")
        print(len(claims))

        print("\nEVIDENCE COUNT:")
        print(len(evidence))

        print("\nCITATION GROUP COUNT:")
        print(len(citations))

        print("\nSOURCE COUNT:")
        print(len(sources))

        print("\nCONFIDENCE:")
        print(
            confidence
        )

        print("\nHALLUCINATION CHECK:")
        print(
            hallucination
        )

        results.append(
            {
                "name": name,
                "success": True,
                "answer": answer,
                "claim_count": len(claims),
                "evidence_count": len(evidence),
                "citation_count": len(citations),
                "source_count": len(sources),
                "confidence": confidence,
                "hallucination": hallucination,
            }
        )

    except Exception as error:

        print("\nERROR")
        print("-" * 80)
        print(
            type(error).__name__,
            ":",
            str(error)
        )

        results.append(
            {
                "name": name,
                "success": False,
                "error": str(error),
            }
        )


# ============================================================
# 4. ADVERSARIAL TEST SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("ADVERSARIAL TEST SUMMARY")
print("=" * 80)


passed = 0
failed = 0


for item in results:

    if item["success"]:

        passed += 1

        print(
            f"[PASS] {item['name']}"
        )

    else:

        failed += 1

        print(
            f"[FAIL] {item['name']}"
        )


print("\nTotal tests:", len(results))
print("Passed:", passed)
print("Failed:", failed)


# ============================================================
# 5. ROBUSTNESS CHECK
# ============================================================

print("\n" + "=" * 80)
print("ROBUSTNESS CHECK")
print("=" * 80)


for item in results:

    if not item["success"]:
        continue

    print(
        f"\n{item['name']}"
    )

    print(
        "  Claims:",
        item["claim_count"]
    )

    print(
        "  Evidence:",
        item["evidence_count"]
    )

    print(
        "  Citations:",
        item["citation_count"]
    )

    print(
        "  Sources:",
        item["source_count"]
    )

    confidence = item.get(
        "confidence",
        {}
    )

    if confidence:

        print(
            "  Confidence score:",
            confidence.get(
                "score",
                0.0
            )
        )

        print(
            "  Confidence level:",
            confidence.get(
                "label",
                "UNKNOWN"
            )
        )


# ============================================================
# 6. FINAL RESULT
# ============================================================

print("\n" + "=" * 80)
print("PART 22 VALIDATION")
print("=" * 80)


if failed == 0:

    print(
        "[PASS] All adversarial test cases executed successfully."
    )

    print(
        "[PASS] Existing QA pipeline handled adversarial inputs."
    )

else:

    print(
        "[WARNING]",
        failed,
        "adversarial test case(s) raised errors."
    )


print("\nPart 22 adversarial testing complete.")
