import math


class ConfidenceScorer:
    """Transparent reliability score, not a probability of correctness."""
    def __init__(
        self,
        hallucination_weight=0.35,
        citation_weight=0.20,
        grounding_weight=0.15,
        retrieval_weight=0.15,
        relevance_weight=0.15,
    ):
        self.weights = {
            "hallucination_support": float(hallucination_weight),
            "citation_coverage": float(citation_weight),
            "grounding_coverage": float(grounding_weight),
            "retrieval_quality": float(retrieval_weight),
            "question_relevance": float(relevance_weight),
        }
        total = sum(self.weights.values()) or 1.0
        self.weights = {k: v / total for k, v in self.weights.items()}

    def score(self, claims=None, evidence=None, citations=None, hallucination=None, relevance_scores=None):
        claims = list(claims or [])
        evidence = list(evidence or [])
        citations = list(citations or [])
        hallucination = hallucination or {}

        support = float(hallucination.get("support_score", 0.0))
        if hallucination.get("supported") and claims:
            support = max(support, 85.0)
        elif hallucination.get("hallucination_detected"):
            support = min(support, 35.0)

        cited = 0
        for c in claims:
            if isinstance(c, dict):
                if c.get("citations") or c.get("evidence_ids"):
                    cited += 1
            elif getattr(c, "citations", None) or getattr(c, "evidence_ids", None):
                cited += 1
        citation_score = 100.0 * cited / max(1, len(claims))
        grounding_score = 100.0 if claims and all(self._has_grounding(c) for c in claims) else (70.0 if claims else 0.0)

        retrieval_score = self._retrieval_quality(evidence)
        relevance_score = self._relevance(relevance_scores, claims)

        components = {
            "hallucination_support": round(support, 2),
            "citation_coverage": round(citation_score, 2),
            "grounding_coverage": round(grounding_score, 2),
            "retrieval_quality": round(retrieval_score, 2),
            "question_relevance": round(relevance_score, 2),
        }
        score = sum(components[k] * self.weights[k] for k in self.weights)
        if not claims or not evidence:
            score = min(score, 25.0)
        if hallucination.get("hallucination_detected"):
            score = min(score, 45.0)

        label = "HIGH" if score >= 80 else "MEDIUM" if score >= 60 else "LOW"
        return {"score": round(score, 2), "label": label, "components": components, "reason": self._reason(label, components)}

    @staticmethod
    def _has_grounding(claim):
        if isinstance(claim, dict):
            return bool(claim.get("evidence_ids") or claim.get("citations") or claim.get("paper_id"))
        return bool(getattr(claim, "evidence_ids", None) or getattr(claim, "citations", None) or getattr(claim, "paper_id", None))

    @staticmethod
    def _retrieval_quality(evidence):
        if not evidence:
            return 0.0
        vals = []
        for x in evidence:
            if not isinstance(x, dict):
                continue
            if "combined_score" in x:
                try: vals.append(float(x["combined_score"]))
                except Exception: pass
        if not vals:
            return 75.0
        avg_val = sum(vals) / len(vals)
        # Bounded scaling: combined scores around 0.5-0.8 map to 75-95
        return max(30.0, min(100.0, 40.0 + avg_val * 70.0))

    @staticmethod
    def _relevance(scores, claims):
        if scores:
            vals = []
            for x in scores:
                try: vals.append(float(x))
                except Exception: pass
            if vals:
                # Map cross-encoder logits monotonically via logistic sigmoid
                probs = [1.0 / (1.0 + math.exp(-max(-6.0, min(6.0, v)))) * 100.0 for v in vals]
                return sum(probs) / len(probs)
        return 80.0 if claims else 0.0

    @staticmethod
    def _reason(label, c):
        if label == "HIGH": return "Strong grounding, citation coverage, retrieval evidence, relevance, and support signals."
        if label == "MEDIUM": return "The answer is reasonably supported, but one or more reliability signals are not strong enough for high confidence."
        return "The evidence or support signals are insufficient for a reliable high-confidence answer."
