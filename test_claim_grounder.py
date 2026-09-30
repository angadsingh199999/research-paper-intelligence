from app.ingestion.ingest import ingest_paper
from app.indexing.chunker import create_chunks
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.task_retriever import TaskSpecificRetriever
from app.analysis.claim_grounder import ClaimGrounder


PDF_PATH = "data/raw_papers/paper1.pdf"


print("\n" + "=" * 80)
print("CLAIM-LEVEL CITATION TEST")
print("=" * 80)


# ============================================================
# INGESTION
# ============================================================

paper = ingest_paper(
    PDF_PATH
)

print(
    f"\nPaper: {paper.metadata.title}"
)


# ============================================================
# CHUNKS
# ============================================================

chunks = create_chunks(
    paper
)

print(
    f"Chunks: {len(chunks)}"
)


# ============================================================
# RETRIEVAL
# ============================================================

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

retriever = TaskSpecificRetriever(
    hybrid
)


# ============================================================
# RETRIEVE EVIDENCE
# ============================================================

query = """
What were the main findings of the study,
including the effects of VTO on experiential
technology acceptance, attitude toward VTO,
and willingness to buy?
"""

evidence = retriever.search(
    query=query,
    task="results",
    top_k=5
)

print(
    f"Evidence retrieved: {len(evidence)}"
)


# ============================================================
# GROUND CLAIMS
# ============================================================

grounder = ClaimGrounder()

grounded = grounder.ground(
    evidence=evidence,
    instruction="""
Extract the main factual research findings.
Each claim must be directly supported by
the retrieved evidence.
"""
)


# ============================================================
# OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("GROUNDED CLAIMS")
print("=" * 80)

for claim in grounded.claims:

    print(
        "\nClaim:"
    )

    print(
        claim.claim
    )

    print(
        "Evidence IDs:"
    )

    print(
        claim.evidence_ids
    )

    print(
        "Citations:"
    )

    for citation in claim.citations:

        print(
            citation
        )
