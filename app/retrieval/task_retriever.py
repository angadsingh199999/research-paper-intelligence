"""Task-aware retrieval with a hard semantic gate."""

from __future__ import annotations

from app.analysis.task_semantics import (
    is_task_valid_sentence,
    normalize_section,
    query_expansion,
    split_sentences,
)
from app.retrieval.reranker import Reranker


class TaskSpecificRetriever:
    """Retrieve evidence and reject chunks that cannot answer the task."""

    def __init__(
        self,
        hybrid_retriever,
        reranker=None,
        candidate_k=36,
        top_k=8,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker if reranker is not None else Reranker()
        self.candidate_k = int(candidate_k)
        self.top_k = int(top_k)
        self.chunks = self._discover_chunks()

    def _discover_chunks(self):
        candidates = (
            self.hybrid_retriever,
            getattr(self.hybrid_retriever, "bm25_retriever", None),
            getattr(self.hybrid_retriever, "bm25", None),
            getattr(self.hybrid_retriever, "vector_retriever", None),
            getattr(self.hybrid_retriever, "vector", None),
        )
        for obj in candidates:
            chunks = getattr(obj, "chunks", None)
            if chunks:
                return list(chunks)
        return []

    def search(self, query, task=None, top_k=None):
        query = str(query or "").strip()
        if not query:
            return []

        final_k = int(top_k) if top_k is not None else self.top_k
        candidate_k = max(self.candidate_k, final_k * 4)

        retrieval_query = query
        if task:
            expansion = query_expansion(str(task).lower().strip())
            if expansion:
                retrieval_query = f"{query} {expansion}"

        candidates = self._call_hybrid(
            retrieval_query,
            candidate_k,
        )

        if not candidates and self.chunks:
            # Reliable lexical fallback over the entire local corpus.
            candidates = self._local_bm25_search(
                retrieval_query,
                self.chunks,
                candidate_k,
            )

        if not candidates:
            return []

        normalized = [self._normalize_result(x) for x in candidates]

        if task:
            normalized = self._hard_task_filter(normalized, task)

        if not normalized:
            # One more full-corpus lexical pass. We never return unvalidated
            # evidence for a task-specific request.
            if self.chunks:
                fallback = self._local_bm25_search(
                    retrieval_query,
                    self.chunks,
                    max(candidate_k * 2, 60),
                )
                fallback = [self._normalize_result(x) for x in fallback]
                normalized = self._hard_task_filter(
                    fallback,
                    task,
                )

        if not normalized:
            return []

        try:
            reranked = self.reranker.rerank(
                query=query,
                results=normalized,
                top_k=min(len(normalized), max(final_k * 2, final_k)),
            )
        except Exception:
            reranked = normalized

        # Task validity is authoritative; reranker only orders already-valid
        # evidence.
        reranked = [self._normalize_result(x) for x in reranked]
        reranked.sort(
            key=lambda x: (
                float(x.get("task_score", 0.0)),
                float(x.get("valid_sentence_count", 0)),
                float(x.get("reranker_score", 0.0)),
                float(x.get("rrf_score", 0.0)),
            ),
            reverse=True,
        )
        return reranked[:final_k]

    def retrieve(self, query, task=None, top_k=None):
        return self.search(query=query, task=task, top_k=top_k)

    def run(self, query, task=None, top_k=None):
        return self.search(query=query, task=task, top_k=top_k)

    def _call_hybrid(self, query, top_k):
        for name in ("search", "retrieve", "run"):
            method = getattr(self.hybrid_retriever, name, None)
            if method is None:
                continue
            try:
                return method(query=query, top_k=top_k)
            except TypeError:
                try:
                    return method(query, top_k)
                except Exception:
                    continue
            except Exception:
                continue
        return []

    @staticmethod
    def _local_bm25_search(query, chunks, top_k):
        try:
            from app.retrieval.bm25_retriever import BM25Retriever
            return BM25Retriever(chunks).search(query, top_k=top_k)
        except Exception:
            return []

    @staticmethod
    def _normalize_result(result):
        item = dict(result)
        metadata = item.get("metadata") or {}
        if isinstance(metadata, dict):
            for key in (
                "paper_id", "paper_title", "section", "page_start",
                "page_end", "word_count", "chunk_id",
            ):
                if not item.get(key) and metadata.get(key) is not None:
                    item[key] = metadata.get(key)

        item.setdefault("paper_id", "")
        item.setdefault("paper_title", "")
        item.setdefault("section", "")
        item.setdefault("page_start", 0)
        item.setdefault("page_end", 0)
        item.setdefault("text", "")
        item.setdefault("chunk_id", "")
        item.setdefault("rrf_score", 0.0)
        item.setdefault("score", 0.0)
        return item

    @staticmethod
    def _hard_task_filter(results, task):
        task = str(task or "question_answering").lower().strip()
        output = []

        for item in results:
            section = normalize_section(item.get("section", ""))
            text = str(item.get("text", "") or "").strip()
            if not text:
                continue

            sentences = split_sentences(text)
            if not sentences:
                sentences = [text]

            valid = [
                sentence
                for sentence in sentences
                if is_task_valid_sentence(
                    sentence,
                    task,
                    section,
                )
            ]

            if not valid:
                continue

            item = dict(item)
            item["valid_sentence_count"] = len(valid)
            item["task_score"] = float(
                len(valid) * 10
                + (10 if any(
                    marker in section
                    for marker in (
                        "limitation", "discussion", "conclusion",
                        "method", "design", "participant", "sample",
                        "data", "result", "finding", "future",
                        "contribution", "implication", "model",
                        "hypothesis", "measure",
                    )
                ) else 0)
            )
            # Keep the whole chunk for grounding/citation; ClaimGrounder will
            # select only the valid sentences from it.
            output.append(item)

        return output
