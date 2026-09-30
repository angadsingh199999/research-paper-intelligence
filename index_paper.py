from app.ingestion.ingest import ingest_paper
from app.indexing.chunker import create_chunks
from app.embeddings.embedder import EmbeddingModel
from app.vectorstore.chroma_store import ChromaVectorStore


PDF_PATH = "data/raw_papers/paper1.pdf"


# ------------------------------------------------
# STEP 1 — Ingest
# ------------------------------------------------

print("\nIngesting paper...")

paper = ingest_paper(
    PDF_PATH
)


# ------------------------------------------------
# STEP 2 — Chunk
# ------------------------------------------------

print("\nCreating chunks...")

chunks = create_chunks(
    paper
)

print(
    f"Total chunks: {len(chunks)}"
)


# ------------------------------------------------
# STEP 3 — Embeddings
# ------------------------------------------------

print("\nLoading embedding model...")

embedder = EmbeddingModel()


texts = [
    chunk.text
    for chunk in chunks
]


print("\nGenerating embeddings...")

embeddings = embedder.encode(
    texts
)


# ------------------------------------------------
# STEP 4 — Vector database
# ------------------------------------------------

print("\nConnecting to ChromaDB...")

vector_store = ChromaVectorStore()


# ------------------------------------------------
# STEP 5 — Store everything
# ------------------------------------------------

print("\nAdding chunks to vector database...")

vector_store.add_chunks(
    chunks,
    embeddings
)


# ------------------------------------------------
# STEP 6 — Verify
# ------------------------------------------------

print("\n" + "=" * 70)
print("VECTOR DATABASE INDEXING COMPLETE")
print("=" * 70)

print(
    f"\nChunks created: "
    f"{len(chunks)}"
)

print(
    f"Vectors generated: "
    f"{len(embeddings)}"
)

print(
    f"Vectors stored in ChromaDB: "
    f"{vector_store.count()}"
)