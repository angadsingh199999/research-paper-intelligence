from app.ingestion.ingest import ingest_paper
from app.indexing.chunker import create_chunks
from app.retrieval.bm25_retriever import BM25Retriever


PDF_PATH = "data/raw_papers/paper1.pdf"


# ----------------------------------------
# 1. Ingest paper
# ----------------------------------------

print("\nIngesting paper...")

paper = ingest_paper(
    PDF_PATH
)


# ----------------------------------------
# 2. Create chunks
# ----------------------------------------

print("\nCreating chunks...")

chunks = create_chunks(
    paper
)

print(
    f"Total chunks: {len(chunks)}"
)


# ----------------------------------------
# 3. Build BM25 index
# ----------------------------------------

print("\nBuilding BM25 index...")

retriever = BM25Retriever(
    chunks
)

print("BM25 index ready.")


# ----------------------------------------
# 4. Test queries
# ----------------------------------------

queries = [

    "What is Virtual Try-On?",

    "What methodology did the researchers use?",

    "What are the limitations of the study?",

    "What was Cohen's d?"
]


for query in queries:

    print("\n")
    print("=" * 80)

    print(
        f"QUERY: {query}"
    )

    print("=" * 80)

    results = retriever.search(
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
            f"BM25 Score: "
            f"{result['score']:.4f}"
        )

        print("\nTEXT:")

        print(
            result["text"][:1000]
        )