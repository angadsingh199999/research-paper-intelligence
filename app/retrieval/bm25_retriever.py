import re

from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, chunks):

        self.chunks = chunks

        self.tokenized_chunks = [
            self._tokenize(chunk.text)
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_chunks
        )

    # ========================================================
    # TOKENIZATION
    # ========================================================

    def _tokenize(self, text):

        return re.findall(
            r"\b\w+\b",
            text.lower()
        )

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query,
        top_k=5
    ):

        query_tokens = self._tokenize(
            query
        )

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []

        for index in ranked_indices:

            chunk = self.chunks[index]

            section = (
                chunk.section or ""
            ).strip().lower()

            # ------------------------------------------------
            # Ignore non-research material
            # ------------------------------------------------

            if section == "references":
                continue

            if section == "appendix":
                continue

            if section.startswith(
                "stimulus in"
            ):
                continue

            # ------------------------------------------------
            # Return normalized result structure
            # ------------------------------------------------

            results.append({

                "chunk_id": chunk.chunk_id,

                "paper_id": chunk.paper_id,

                "paper_title": chunk.paper_title,

                "text": chunk.text,

                "section": chunk.section,

                "page_start": chunk.page_start,

                "page_end": chunk.page_end,

                "word_count": chunk.word_count,

                "score": float(
                    scores[index]
                )

            })

            if len(results) >= top_k:
                break

        return results

    def retrieve(self, query, top_k=5):
        return self.search(query, top_k=top_k)

    def run(self, query, top_k=5):
        return self.search(query, top_k=top_k)

