import chromadb


class ChromaVectorStore:

    def __init__(
        self,
        persist_directory="data/chroma",
        collection_name="research_papers"
    ):

        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=collection_name
            )
        )

    # ========================================================
    # ADD OR UPSERT CHUNKS
    # ========================================================

    def add_chunks(
        self,
        chunks,
        embeddings
    ):
        if not chunks:
            return

        ids = [
            chunk.chunk_id
            for chunk in chunks
        ]

        documents = [
            chunk.text
            for chunk in chunks
        ]

        metadatas = [
            {
                "paper_id": chunk.paper_id,
                "paper_title": chunk.paper_title,
                "section": chunk.section,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "word_count": chunk.word_count
            }
            for chunk in chunks
        ]

        embedding_list = (
            embeddings.tolist()
            if hasattr(embeddings, "tolist")
            else embeddings
        )

        # Use upsert to safely update or insert chunks idempotently
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embedding_list,
            metadatas=metadatas
        )

    # ========================================================
    # GET EXISTING IDS
    # ========================================================

    def get_existing_ids(self):
        try:
            res = self.collection.get()
            return set(res.get("ids", []))
        except Exception:
            return set()

    # ========================================================
    # COUNT
    # ========================================================

    def count(self):
        return self.collection.count()
