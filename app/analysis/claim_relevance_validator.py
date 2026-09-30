"""Question relevance + strict research-task validation."""

from __future__ import annotations

import re

from sentence_transformers import CrossEncoder

from app.analysis.task_semantics import is_noise, is_task_valid, normalize_task


class ClaimRelevanceValidator:
    """Rank claims while never allowing an invalid task type through.

    The cross-encoder is used for ranking, not as a type classifier. The
    deterministic task-semantic gate therefore runs *before* ranking and is
    repeated during filtering as defense in depth.
    """

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        print("Loading claim relevance model:")
        print(model_name)
        self.model = CrossEncoder(model_name)
        print("Claim relevance model loaded.")

    def score_claims(self, question, claims, task=None):
        if not claims:
            return []

        task = normalize_task(task)
        valid = []
        for claim in claims:
            text = self._claim_text(claim)
            section = self._claim_section(claim)
            if is_noise(text):
                continue
            if task != "question_answering" and not is_task_valid(task, text, section):
                continue
            valid.append(claim)

        if not valid:
            return []

        pairs = [[str(question), self._claim_text(claim)] for claim in valid]
        semantic_scores = self.model.predict(pairs)

        q_tokens = set(re.findall(r"[a-z0-9]{4,}", str(question).lower()))
        scored = []
        for claim, semantic in zip(valid, semantic_scores):
            text = self._claim_text(claim)
            c_tokens = set(re.findall(r"[a-z0-9]{4,}", text.lower()))
            lexical = len(q_tokens & c_tokens) / max(1, len(q_tokens))
            task_score = 0.0
            if task != "question_answering":
                decision = self._task_decision(task, text, self._claim_section(claim))
                task_score = float(decision.get("score", 0.0))

            score = float(semantic) + 0.60 * lexical + 0.90 * task_score
            scored.append({"claim": claim, "relevance_score": score})

        scored.sort(key=lambda item: item["relevance_score"], reverse=True)
        return scored

    def filter_claims(self, question, claims, task=None, max_claims=5):
        task = normalize_task(task)
        scored = self.score_claims(question, claims, task=task)
        if not scored:
            return []

        filtered = []
        for item in scored:
            claim = item["claim"]
            text = self._claim_text(claim)
            section = self._claim_section(claim)
            if is_noise(text):
                continue
            if task != "question_answering" and not is_task_valid(task, text, section):
                continue
            filtered.append(item)

        if not filtered:
            return []

        return [item["claim"] for item in filtered[:max(1, int(max_claims))]]

    @staticmethod
    def _task_decision(task, text, section=""):
        from app.analysis.task_semantics import task_decision
        return task_decision(task, text, section)

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
        citations = claim.get("citations", []) if isinstance(claim, dict) else getattr(claim, "citations", [])
        for citation in citations or []:
            if isinstance(citation, dict):
                value = citation.get("section", "")
            else:
                value = getattr(citation, "section", "")
            if value:
                return str(value)
        return ""