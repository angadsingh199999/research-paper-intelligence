from app.ingestion.ingest import ingest_paper
from app.indexing.chunker import create_chunks


pdf_path = "data/raw_papers/paper1.pdf"


# -----------------------------------
# Ingest paper
# -----------------------------------

paper = ingest_paper(pdf_path)


# -----------------------------------
# Create chunks
# -----------------------------------

chunks = create_chunks(paper)


print("\n" + "=" * 70)
print("CHUNKING RESULTS")
print("=" * 70)

print(f"Paper: {paper.metadata.title}")

print(f"Total chunks: {len(chunks)}")


# -----------------------------------
# Inspect first 10 chunks
# -----------------------------------

for chunk in chunks[:10]:

    print("\n" + "-" * 70)

    print(f"Chunk ID: {chunk.chunk_id}")

    print(f"Section: {chunk.section}")

    print(
        f"Pages: "
        f"{chunk.page_start}-{chunk.page_end}"
    )

    print(
        f"Word count: "
        f"{len(chunk.text.split())}"
    )

    print("\nTEXT:")

    print(chunk.text[:1000])