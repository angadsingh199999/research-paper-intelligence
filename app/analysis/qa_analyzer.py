"""Core QA orchestration for Research Paper Intelligence.

The important rule in this module is:
    evidence support is not the same thing as task correctness.

Task validity is therefore checked before a claim can reach the final answer.
No fallback re-introduces task-invalid claims.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.analysis.answer_synthesizer import GroundedAnswerSynthesizer
from app.analysis.claim_grounder import ClaimGrounder
from app.analysis.claim_relevance_validator import ClaimRelevanceValidator
from app.analysis.citation_generator import CitationGenerator
from app.analysis.confidence_scorer import ConfidenceScorer
from app.analysis.hallucination_detector import HallucinationDetector
from app.analysis.task_semantics import (
    detect_task,
    is_comparison_question,
    is_task_valid_sentence,
    normalize_section,
    query_expansion,
    split_sentences,
)
from app.retrieval.bm25_retriever import BM25Retriever


TASK_FILTERED = {
    "methodology",
    "datasets",
    "models",
    "results",
    "limitations",
    "future_work",
    "literature_review",
    "contributions",
    "variables",
}


class QAAnalyzer:
    """Run routing, retrieval, grounding, validation, synthesis and checks."""

    def __init__(
        self,
        task_retriever,
        claim_grounder=None,
        claim_relevance_validator=None,
        task_router=None,
        answer_synthesizer=None,
        citation_generator=None,
        hallucination_detector=None,
        confidence_scorer=None,
    ):
        self.task_retriever = task_retriever
        self.claim_grounder = claim_grounder or ClaimGrounder()
        self.claim_relevance_validator = (
            claim_relevance_validator or ClaimRelevanceValidator()
        )
        self.task_router = task_router
        self.answer_synthesizer = answer_synthesizer or GroundedAnswerSynthesizer()
        self.citation_generator = citation_generator or CitationGenerator()
        self.hallucination_detector = hallucination_detector or HallucinationDetector()
        self.confidence_scorer = confidence_scorer or ConfidenceScorer()

    def answer(self, question, task=None):
        question = str(question or "").strip()
        if not question:
            return self._empty_response("Please provide a question.")

        if task is None:
            task, task_scores = self._detect_task(question)
        else:
            task = str(task).lower().strip()
            task_scores = {task: 1}

        print(f"Detected task: {task}")
        comparison = is_comparison_question(question)

        evidence = self._retrieve(question, task, comparison)
        if not evidence:
            return self._empty_response(
                "The papers do not contain enough task-specific evidence to answer this question reliably.",
                task=task,
                task_scores=task_scores,
            )

        grounded = self.claim_grounder.ground(
            evidence=evidence,
            task=task,
        )
        grounded_claims = list(getattr(grounded, "claims", []) or [])

        # Hard task validation comes before semantic ranking.
        task_valid_claims = self._hard_filter_claims(
            grounded_claims,
            task,
        )

        print(
            f"Task-valid grounded claims: {len(task_valid_claims)} / {len(grounded_claims)}"
        )

        # Rank only claims that have already passed the task gate.
        ranked_claims, relevance_scores = self._rank_valid_claims(
            question,
            task_valid_claims,
            task,
        )

        # Comparison answers require one or more valid claims per paper when
        # such claims are available; never add an invalid claim merely for coverage.
        relevant_claims = self._select_claims(
            ranked_claims,
            comparison=comparison,
        )

        if not relevant_claims:
            return self._empty_response(
                "The retrieved papers do not contain enough task-specific evidence to answer this question reliably.",
                task=task,
                task_scores=task_scores,
                evidence=evidence,
            )

        available_papers = self._available_papers()
        represented_papers = self._claim_paper_ids(
            relevant_claims,
            evidence,
        )

        print(
            "Papers represented in final claims:",
            sorted(represented_papers),
        )

        synthesis = self.answer_synthesizer.synthesize(
            question=question,
            task=task,
            grounded_claims=relevant_claims,
            evidence=evidence,
            available_papers=available_papers,
        )
        answer_text = str(synthesis.get("answer") or "").strip()

        if not answer_text:
            answer_text = (
                "The available evidence does not contain enough grounded information "
                "to answer this question reliably."
            )

        # Verify the actual final answer against the evidence, not merely the claims.
        try:
            hallucination = self.hallucination_detector.verify(
                answer=answer_text,
                evidence=evidence,
            )
        except Exception as exc:
            print(f"Hallucination detection failed: {exc}")
            hallucination = {
                "supported": False,
                "hallucination_detected": False,
                "support_score": 0.0,
                "claims": [],
            }

        try:
            citation_result = self.citation_generator.generate(
                claims=relevant_claims,
                evidence=evidence,
            )
        except Exception as exc:
            print(f"Citation generation failed: {exc}")
            citation_result = {
                "claim_citations": [],
                "sources": [],
            }

        citations = citation_result.get("claim_citations", []) or []
        sources = citation_result.get("sources", []) or []

        try:
            confidence = self.confidence_scorer.score(
                claims=relevant_claims,
                evidence=evidence,
                citations=citations,
                hallucination=hallucination,
                relevance_scores=relevance_scores,
            )
        except Exception as exc:
            print(f"Confidence scoring failed: {exc}")
            confidence = {
                "score": 0.0,
                "label": "LOW",
                "reason": "Confidence scoring failed.",
            }

        confidence = self._apply_confidence_guards(
            confidence=confidence,
            task=task,
            claims=relevant_claims,
            evidence=evidence,
            hallucination=hallucination,
            comparison=comparison,
            represented_papers=represented_papers,
            available_papers=set(available_papers),
        )

        return {
            "answer": answer_text,
            "task": task,
            "claims": relevant_claims,
            "evidence": self._build_evidence_output(evidence),
            "task_scores": task_scores,
            "hallucination": hallucination,
            "citations": citations,
            "sources": sources,
            "confidence": confidence,
        }

    # ============================================================
    # TASK DETECTION
    # ============================================================

    def _detect_task(self, question):
        # Explicit deterministic routing is safer than allowing generic words
        # such as "both", "data", or "study" to decide the task.
        return detect_task(question)

    # ============================================================
    # RETRIEVAL
    # ============================================================

    def _retrieve(self, question, task, comparison):
        if comparison:
            per_paper = self._retrieve_per_paper(question, task)
            if per_paper:
                return per_paper

        retriever = self.task_retriever
        for name in ("search", "retrieve", "run"):
            method = getattr(retriever, name, None)
            if method is None:
                continue
            try:
                results = method(
                    query=question,
                    task=task,
                    top_k=8,
                )
            except TypeError:
                try:
                    results = method(question, task, 8)
                except Exception:
                    continue
            except Exception:
                continue
            if results:
                # For task-specific tasks, TaskSpecificRetriever already applies
                # a hard gate; this second gate protects against older implementations.
                if task in TASK_FILTERED:
                    results = self._filter_evidence_items(results, task)
                if results:
                    return results
        return []

    def _retrieve_per_paper(self, question, task):
        chunks = self._get_available_chunks()
        if not chunks:
            return []

        grouped = defaultdict(list)
        for chunk in chunks:
            paper_id = self._field(chunk, "paper_id")
            if paper_id:
                grouped[str(paper_id)].append(chunk)

        if not grouped:
            return []

        merged = {}
        expansion = query_expansion(task)
        local_query = f"{question} {expansion}".strip()
        reranker = getattr(self.task_retriever, "reranker", None)

        for paper_id, paper_chunks in grouped.items():
            try:
                local_bm25 = BM25Retriever(paper_chunks)
                local_results = local_bm25.search(
                    local_query,
                    top_k=min(12, max(8, len(paper_chunks))),
                )
            except Exception:
                local_results = []

            local_results = [self._normalize_item(x) for x in local_results]
            if task in TASK_FILTERED:
                local_results = self._filter_evidence_items(local_results, task)

            if reranker is not None and local_results:
                try:
                    local_results = reranker.rerank(
                        query=question,
                        results=local_results,
                        top_k=len(local_results),
                    )
                except Exception:
                    pass

            local_results = [self._normalize_item(x) for x in local_results]
            local_results.sort(
                key=lambda x: (
                    float(x.get("task_score", 0.0)),
                    float(x.get("reranker_score", 0.0)),
                    float(x.get("score", x.get("rrf_score", 0.0))),
                ),
                reverse=True,
            )

            # Keep at most 3 strong task-valid chunks per paper.
            for item in local_results[:3]:
                chunk_id = item.get("chunk_id")
                if chunk_id:
                    merged[str(chunk_id)] = item

        return list(merged.values())

    def _filter_evidence_items(self, results, task):
        filtered = []
        for raw in results:
            item = self._normalize_item(raw)
            text = str(item.get("text", "") or "")
            section = normalize_section(item.get("section", ""))
            sentences = split_sentences(text) or [text]
            valid_sentences = [
                sentence
                for sentence in sentences
                if is_task_valid_sentence(sentence, task, section)
            ]
            if not valid_sentences:
                continue
            item["task_valid_sentences"] = valid_sentences
            item["task_score"] = float(
                len(valid_sentences) * 10
                + (10 if section else 0)
            )
            filtered.append(item)
        return filtered

    # ============================================================
    # CLAIM VALIDATION / SELECTION
    # ============================================================

    def _hard_filter_claims(self, claims, task):
        if task not in TASK_FILTERED:
            return list(claims)

        output = []
        seen = set()
        for claim in claims:
            text = self._claim_text(claim)
            section = self._claim_section(claim)
            sentences = split_sentences(text) or [text]

            valid = any(
                is_task_valid_sentence(sentence, task, section)
                for sentence in sentences
            )
            if not valid:
                continue

            key = " ".join(text.lower().split())
            if key in seen:
                continue
            seen.add(key)
            output.append(claim)
        return output

    def _rank_valid_claims(self, question, claims, task):
        if not claims:
            return [], {}

        try:
            scored = self.claim_relevance_validator.score_claims(
                question=question,
                claims=claims,
                task=task,
            )
        except TypeError:
            try:
                scored = self.claim_relevance_validator.score_claims(
                    question=question,
                    claims=claims,
                )
            except Exception:
                scored = []
        except Exception:
            scored = []

        if not scored:
            # All claims already passed hard task filtering. Lexical ordering is
            # a safe fallback if the neural ranker is unavailable.
            scored = []
            q_words = self._tokens(question)
            for claim in claims:
                text = self._claim_text(claim)
                overlap = len(q_words & self._tokens(text)) / max(1, len(q_words))
                scored.append({"claim": claim, "relevance_score": overlap})
            scored.sort(key=lambda x: x["relevance_score"], reverse=True)

        ranked = [item["claim"] for item in scored if item.get("claim") is not None]
        scores = {
            id(item["claim"]): float(item.get("relevance_score", 0.0))
            for item in scored
            if item.get("claim") is not None
        }
        return ranked, scores

    def _select_claims(self, ranked_claims, comparison):
        if not ranked_claims:
            return []

        deduped = []
        seen = set()
        for claim in ranked_claims:
            key = " ".join(self._claim_text(claim).lower().split())
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(claim)

        if not comparison:
            return deduped[:8]

        grouped = defaultdict(list)
        for claim in deduped:
            grouped[self._claim_paper_id_from_claim(claim) or "unknown"].append(claim)

        selected = []
        # First one valid claim per paper.
        for paper_id in sorted(grouped):
            selected.append(grouped[paper_id][0])

        # Then add more only after every represented paper has a valid claim.
        for claim in deduped:
            if len(selected) >= max(8, len(grouped) * 2):
                break
            if claim in selected:
                continue
            selected.append(claim)

        return selected

    # ============================================================
    # CONFIDENCE / SAFETY
    # ============================================================

    def _apply_confidence_guards(
        self,
        confidence,
        task,
        claims,
        evidence,
        hallucination,
        comparison,
        represented_papers,
        available_papers,
    ):
        confidence = dict(confidence or {})
        score = float(confidence.get("score", 0.0) or 0.0)
        reason = str(confidence.get("reason", "") or "")

        if not claims:
            score = 0.0
            label = "LOW"
        else:
            label = "HIGH" if score >= 85 else "MEDIUM" if score >= 60 else "LOW"

        if hallucination.get("hallucination_detected"):
            score = min(score, 49.99)
            label = "LOW"
            reason = "The final answer contains claims that were not sufficiently supported by retrieved evidence."

        if task in TASK_FILTERED:
            # Every final claim is hard task-valid. This is a prerequisite for
            # a high score, not merely another weighted feature.
            if not claims:
                score = 0.0
                label = "LOW"
            else:
                # Never let a weighted heuristic hide weak task coverage.
                if comparison and available_papers and represented_papers != available_papers:
                    coverage = len(represented_papers & available_papers) / len(available_papers)
                    score = min(score, 50 + 29.99 * coverage)
                    label = "MEDIUM" if score >= 60 else "LOW"
                    reason = "The answer is task-valid, but not every available paper contributed task-specific evidence."

        confidence["score"] = round(max(0.0, min(100.0, score)), 2)
        confidence["label"] = label
        confidence["task_validated"] = bool(claims) if task in TASK_FILTERED else True
        if reason:
            confidence["reason"] = reason
        return confidence

    # ============================================================
    # HELPERS
    # ============================================================

    def _get_available_chunks(self):
        candidates = [
            self.task_retriever,
            getattr(self.task_retriever, "hybrid_retriever", None),
        ]
        hybrid = getattr(self.task_retriever, "hybrid_retriever", None)
        if hybrid is not None:
            candidates.extend(
                [
                    getattr(hybrid, "bm25_retriever", None),
                    getattr(hybrid, "bm25", None),
                    getattr(hybrid, "vector_retriever", None),
                    getattr(hybrid, "vector", None),
                ]
            )
        for candidate in candidates:
            chunks = getattr(candidate, "chunks", None)
            if chunks:
                return list(chunks)
        return []

    def _available_papers(self):
        result = set()
        for chunk in self._get_available_chunks():
            paper_id = self._field(chunk, "paper_id")
            if paper_id:
                result.add(str(paper_id))
        return sorted(result)

    @staticmethod
    def _field(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _normalize_item(self, item):
        if isinstance(item, dict):
            return dict(item)
        return {
            "chunk_id": self._field(item, "chunk_id", ""),
            "paper_id": self._field(item, "paper_id", ""),
            "paper_title": self._field(item, "paper_title", ""),
            "section": self._field(item, "section", ""),
            "text": self._field(item, "text", ""),
            "page_start": self._field(item, "page_start", 0),
            "page_end": self._field(item, "page_end", 0),
            "score": self._field(item, "score", 0.0),
        }

    def _claim_paper_id_from_claim(self, claim):
        if isinstance(claim, dict):
            direct = claim.get("paper_id")
            citations = claim.get("citations") or []
            evidence_ids = claim.get("evidence_ids") or []
        else:
            direct = getattr(claim, "paper_id", None)
            citations = getattr(claim, "citations", []) or []
            evidence_ids = getattr(claim, "evidence_ids", []) or []

        if direct:
            return str(direct)
        for citation in citations:
            value = (
                citation.get("paper_id")
                if isinstance(citation, dict)
                else getattr(citation, "paper_id", None)
            )
            if value:
                return str(value)
        for evidence_id in evidence_ids:
            value = str(evidence_id)
            if "_chunk_" in value:
                return value.split("_chunk_", 1)[0]
        return None

    def _claim_paper_ids(self, claims, evidence):
        evidence_map = {
            str(item.get("chunk_id")): str(item.get("paper_id"))
            for item in evidence
            if item.get("chunk_id") and item.get("paper_id")
        }
        ids = set()
        for claim in claims:
            direct = self._claim_paper_id_from_claim(claim)
            if direct:
                ids.add(str(direct))
                continue
            for evidence_id in (
                claim.get("evidence_ids", [])
                if isinstance(claim, dict)
                else getattr(claim, "evidence_ids", [])
            ):
                if str(evidence_id) in evidence_map:
                    ids.add(evidence_map[str(evidence_id)])
                    break
        return ids

    @staticmethod
    def _claim_text(claim):
        if isinstance(claim, dict):
            return str(claim.get("claim") or claim.get("text") or "").strip()
        return str(
            getattr(claim, "claim", None)
            or getattr(claim, "text", None)
            or ""
        ).strip()

    @staticmethod
    def _claim_section(claim):
        if isinstance(claim, dict) and claim.get("section"):
            return str(claim["section"])
        citations = (
            claim.get("citations", [])
            if isinstance(claim, dict)
            else getattr(claim, "citations", [])
        ) or []
        if citations:
            first = citations[0]
            return str(
                first.get("section", "")
                if isinstance(first, dict)
                else getattr(first, "section", "")
            )
        return ""

    @staticmethod
    def _tokens(text):
        stop = {
            "what", "which", "were", "was", "are", "the", "and", "for", "how",
            "does", "did", "from", "with", "about", "both", "each", "paper", "papers",
            "study", "studies", "according", "using", "used", "their", "this", "that",
        }
        return {
            x for x in __import__("re").findall(r"[a-z0-9]{3,}", str(text).lower())
            if x not in stop
        }

    def _build_evidence_output(self, evidence):
        output = []
        for item in evidence:
            output.append(
                {
                    "paper_id": item.get("paper_id", ""),
                    "paper_title": item.get("paper_title", ""),
                    "page_start": item.get("page_start", 0),
                    "page_end": item.get("page_end", 0),
                    "section": item.get("section", ""),
                    "chunk_id": item.get("chunk_id", ""),
                    "text": item.get("text", ""),
                    "task_score": item.get("task_score", 0.0),
                    "rrf_score": item.get("rrf_score", 0.0),
                    "reranker_score": item.get("reranker_score", 0.0),
                    "combined_score": item.get("combined_score", 0.0),
                }
            )
        return output

    @staticmethod
    def _empty_response(message, task="question_answering", task_scores=None, evidence=None):
        return {
            "answer": message,
            "task": task,
            "claims": [],
            "evidence": [],
            "task_scores": task_scores or {},
            "hallucination": {
                "supported": False,
                "hallucination_detected": False,
                "support_score": 0.0,
                "claims": [],
            },
            "citations": [],
            "sources": [],
            "confidence": {
                "score": 0.0,
                "label": "LOW",
                "task_validated": False,
            },
        }
