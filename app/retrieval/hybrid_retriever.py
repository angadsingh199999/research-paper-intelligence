class HybridRetriever:

    def __init__(
        self,
        vector_retriever=None,
        bm25_retriever=None,
        rrf_k=60
    ):
        # Auto-detect swapped arguments (e.g., if BM25Retriever was passed as first positional arg)
        is_first_bm25 = (
            vector_retriever is not None
            and hasattr(vector_retriever, "bm25")
            and not hasattr(vector_retriever, "collection")
            and not hasattr(vector_retriever, "chroma")
        )
        is_second_vector = (
            bm25_retriever is not None
            and (
                hasattr(bm25_retriever, "collection")
                or hasattr(bm25_retriever, "chroma")
                or hasattr(bm25_retriever, "embedder")
            )
        )

        if is_first_bm25 and is_second_vector:
            vector_retriever, bm25_retriever = bm25_retriever, vector_retriever
        elif is_first_bm25 and bm25_retriever is None:
            bm25_retriever = vector_retriever
            vector_retriever = None

        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.bm25 = bm25_retriever
        self.vector = vector_retriever
        self.rrf_k = rrf_k

    @property
    def chunks(self):
        """Expose chunks if available on either underlying retriever."""
        if self.bm25_retriever is not None and hasattr(self.bm25_retriever, "chunks"):
            return getattr(self.bm25_retriever, "chunks", [])
        if self.vector_retriever is not None and hasattr(self.vector_retriever, "chunks"):
            return getattr(self.vector_retriever, "chunks", [])
        return []


    # ============================================================
    # SEARCH
    # ============================================================

    def search(
        self,
        query,
        top_k=10
    ):

        vector_results = []
        bm25_results = []

        # --------------------------------------------------------
        # Vector retrieval
        # --------------------------------------------------------

        if self.vector_retriever is not None:

            if hasattr(
                self.vector_retriever,
                "search"
            ):
                vector_results = (
                    self.vector_retriever.search(
                        query,
                        top_k=top_k
                    )
                )

            elif hasattr(
                self.vector_retriever,
                "retrieve"
            ):
                vector_results = (
                    self.vector_retriever.retrieve(
                        query,
                        top_k=top_k
                    )
                )

        # --------------------------------------------------------
        # BM25 retrieval
        # --------------------------------------------------------

        if self.bm25_retriever is not None:

            if hasattr(
                self.bm25_retriever,
                "search"
            ):
                bm25_results = (
                    self.bm25_retriever.search(
                        query,
                        top_k=top_k
                    )
                )

            elif hasattr(
                self.bm25_retriever,
                "retrieve"
            ):
                bm25_results = (
                    self.bm25_retriever.retrieve(
                        query,
                        top_k=top_k
                    )
                )

        # --------------------------------------------------------
        # Normalize results
        # --------------------------------------------------------

        vector_results = (
            vector_results
            if vector_results
            else []
        )

        bm25_results = (
            bm25_results
            if bm25_results
            else []
        )

        # --------------------------------------------------------
        # Reciprocal Rank Fusion
        # --------------------------------------------------------

        fused = {}

        # Vector ranking
        for rank, result in enumerate(
            vector_results,
            start=1
        ):

            chunk_id = result.get(
                "chunk_id"
            )

            if not chunk_id:
                continue

            if chunk_id not in fused:

                fused[chunk_id] = {
                    "result": result.copy(),
                    "rrf_score": 0.0
                }

            fused[chunk_id][
                "rrf_score"
            ] += 1.0 / (
                self.rrf_k + rank
            )

        # BM25 ranking
        for rank, result in enumerate(
            bm25_results,
            start=1
        ):

            chunk_id = result.get(
                "chunk_id"
            )

            if not chunk_id:
                continue

            if chunk_id not in fused:

                fused[chunk_id] = {
                    "result": result.copy(),
                    "rrf_score": 0.0
                }

            fused[chunk_id][
                "rrf_score"
            ] += 1.0 / (
                self.rrf_k + rank
            )

        # --------------------------------------------------------
        # Build final results
        # --------------------------------------------------------

        results = []

        for item in fused.values():

            result = item["result"].copy()

            result["rrf_score"] = float(
                item["rrf_score"]
            )

            results.append(result)

        # --------------------------------------------------------
        # Sort
        # --------------------------------------------------------

        results.sort(
            key=lambda x: x.get(
                "rrf_score",
                0.0
            ),
            reverse=True
        )

        return results[:top_k]

    # ============================================================
    # RETRIEVE ALIAS
    # ============================================================

    def retrieve(
        self,
        query,
        top_k=10
    ):
        return self.search(
            query,
            top_k=top_k
        )

    # ============================================================
    # RUN ALIAS
    # ============================================================

    def run(
        self,
        query,
        top_k=10
    ):
        return self.search(
            query,
            top_k=top_k
        )