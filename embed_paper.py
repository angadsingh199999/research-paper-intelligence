from app.ingestion.ingest import ingest_paper
from app.indexing.chunker import create_chunks
from app.embeddings.embedder import EmbeddingModel


PDF_PATH = "data/raw_papers/paper1.pdf"


# --------------------------------------------
# 1. Ingest paper
# --------------------------------------------

print("\nIngesting paper...")

paper = ingest_paper(
    PDF_PATH
)


# --------------------------------------------
# 2. Create chunks
# --------------------------------------------

print("\nCreating chunks...")

chunks = create_chunks(
    paper
)

print(
    f"Total chunks: {len(chunks)}"
)


# --------------------------------------------
# 3. Extract text
# --------------------------------------------

texts = [
    chunk.text
    for chunk in chunks
]


# --------------------------------------------
# 4. Load embedding model
# --------------------------------------------

embedder = EmbeddingModel()


# --------------------------------------------
# 5. Generate embeddings
# --------------------------------------------

print(
    "\nGenerating embeddings..."
)

embeddings = embedder.encode(
    texts
)


# --------------------------------------------
# 6. Inspect results
# --------------------------------------------

print("\n" + "=" * 70)
print("PAPER EMBEDDING RESULTS")
print("=" * 70)

print(
    f"\nPaper: "
    f"{paper.metadata.title}"
)

print(
    f"Chunks: {len(chunks)}"
)

print(
    f"Embedding matrix shape: "
    f"{embeddings.shape}"
)


for i, chunk in enumerate(
    chunks[:5]
):

    print("\n" + "-" * 70)

    print(
        f"Chunk ID: "
        f"{chunk.chunk_id}"
    )

    print(
        f"Section: "
        f"{chunk.section}"
    )

    print(
        f"Pages: "
        f"{chunk.page_start}-"
        f"{chunk.page_end}"
    )

    print(
        f"Words: "
        f"{len(chunk.text.split())}"
    )

    print(
        f"Vector dimensions: "
        f"{len(embeddings[i])}"
    )

    print(
        "First 10 vector values:"
    )

    print(
        embeddings[i][:10]
    )