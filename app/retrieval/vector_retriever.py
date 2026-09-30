"""Vector retrieval over the persistent Chroma collection."""

from __future__ import annotations

from app.embeddings.embedder import EmbeddingModel
from app.vectorstore.chroma_store import ChromaVectorStore


class VectorRetriever:

    def __init__(
        self,
        top_k=5,
        embedder=None,
        vector_store=None,
    ):

        self.top_k = int(
            top_k
        )

        self.embedder = (
            embedder
            if embedder is not None
            else EmbeddingModel()
        )

        self.vector_store = (
            vector_store
            if vector_store is not None
            else ChromaVectorStore()
        )

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query,
        top_k=None,
        exclude_references=True,
        where=None,
    ):

        query_text = str(
            query or ""
        ).strip()

        if not query_text:
            return []

        # ----------------------------------------------------
        # Query embedding
        # ----------------------------------------------------

        query_embedding = (
            self.embedder.encode_query(
                query_text
            )
        )

        # ----------------------------------------------------
        # Metadata filter
        # ----------------------------------------------------

        where_filter = where

        if (
            where_filter is None
            and exclude_references
        ):

            where_filter = {
                "section": {
                    "$ne": "References"
                }
            }

        # ----------------------------------------------------
        # Result count
        # ----------------------------------------------------

        number_of_results = int(
            top_k
            if top_k is not None
            else self.top_k
        )

        if number_of_results <= 0:
            return []

        # ----------------------------------------------------
        # Query Chroma
        # ----------------------------------------------------

        results = (
            self.vector_store
            .collection
            .query(
                query_embeddings=[
                    query_embedding.tolist()
                ],
                n_results=number_of_results,
                where=where_filter,
            )
        )

        # ----------------------------------------------------
        # Extract result arrays
        # ----------------------------------------------------

        ids = (
            results.get("ids")
            or [[]]
        )[0]

        documents = (
            results.get("documents")
            or [[]]
        )[0]

        metadatas = (
            results.get("metadatas")
            or [[]]
        )[0]

        distances = (
            results.get("distances")
            or [[]]
        )[0]

        # ----------------------------------------------------
        # Normalize result structure
        # ----------------------------------------------------

        retrieved_chunks = []

        for index, chunk_id in enumerate(
            ids
        ):

            metadata = (
                metadatas[index]
                or {}
            )

            document = (
                documents[index]
                if index < len(documents)
                else ""
            )

            distance = (
                distances[index]
                if index < len(distances)
                else 0.0
            )

            retrieved_chunks.append(
                {
                    "chunk_id":
                        chunk_id,

                    "paper_id":
                        metadata.get(
                            "paper_id",
                            "",
                        ),

                    "paper_title":
                        metadata.get(
                            "paper_title",
                            "",
                        ),

                    "text":
                        document,

                    "section":
                        metadata.get(
                            "section",
                            "",
                        ),

                    "page_start":
                        metadata.get(
                            "page_start",
                            0,
                        ),

                    "page_end":
                        metadata.get(
                            "page_end",
                            0,
                        ),

                    "word_count":
                        metadata.get(
                            "word_count",
                            0,
                        ),

                    "distance":
                        distance,

                    "metadata":
                        metadata,
                }
            )

        return retrieved_chunks

    # ========================================================
    # COMPATIBILITY ALIASES
    # ========================================================

    def retrieve(
        self,
        query,
        *args,
        **kwargs,
    ):

        top_k = kwargs.get(
            "top_k"
        )

        where = kwargs.get(
            "where"
        )

        exclude_references = kwargs.get(
            "exclude_references",
            True,
        )

        if args:

            if isinstance(
                args[0],
                int,
            ):

                top_k = args[0]

            elif (
                len(args) > 1
                and isinstance(
                    args[1],
                    int,
                )
            ):

                top_k = args[1]

        return self.search(
            query=query,
            top_k=top_k,
            exclude_references=exclude_references,
            where=where,
        )

    def run(
        self,
        query,
        top_k=None,
        where=None,
    ):

        return self.search(
            query=query,
            top_k=top_k,
            where=where,
        )