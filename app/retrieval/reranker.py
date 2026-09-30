import app.config  # ensures offline env flags are active before huggingface model resolution
from sentence_transformers import CrossEncoder


_CROSS_ENCODER_CACHE = {}


def get_shared_cross_encoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    if model_name not in _CROSS_ENCODER_CACHE:
        print(f"Loading reranker model: {model_name}")
        _CROSS_ENCODER_CACHE[model_name] = CrossEncoder(model_name)
        print("Reranker model loaded.")
    return _CROSS_ENCODER_CACHE[model_name]


class Reranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.model = get_shared_cross_encoder(model_name)

    def rerank(
        self,
        query,
        results,
        top_k=5
    ):

        if not results:
            return []

        # ----------------------------------------
        # Create query-document pairs
        # ----------------------------------------

        pairs = [
            [
                query,
                result.get("text", "")
            ]
            for result in results
        ]

        # ----------------------------------------
        # Calculate semantic relevance
        # ----------------------------------------

        scores = self.model.predict(
            pairs
        )

        # ----------------------------------------
        # Attach reranker score
        # ----------------------------------------

        reranked_results = []

        for result, score in zip(
            results,
            scores
        ):

            result_copy = result.copy()

            result_copy[
                "reranker_score"
            ] = float(score)

            # ------------------------------------
            # Combined task-aware score
            # ------------------------------------
            #
            # Task score is intentionally kept
            # dominant so that task-specific
            # retrieval is not destroyed by the
            # generic CrossEncoder.
            #
            # Example:
            #
            # task_score = 10
            # reranker = -4
            #
            # combined =
            # 10 * 2.0 + (-4 * 0.1)
            # = 19.6
            #
            # ------------------------------------

            task_score = result_copy.get(
                "task_score",
                0
            )

            rrf_score = result_copy.get(
                "rrf_score",
                0.0
            )

            combined_score = (
                task_score * 2.0
                + float(score) * 0.1
                + rrf_score
            )

            result_copy[
                "combined_score"
            ] = combined_score

            reranked_results.append(
                result_copy
            )

        # ----------------------------------------
        # Sort using combined task-aware score
        # ----------------------------------------

        reranked_results.sort(
            key=lambda x: x[
                "combined_score"
            ],
            reverse=True
        )

        # ----------------------------------------
        # Return best evidence
        # ----------------------------------------

        return reranked_results[:top_k]
