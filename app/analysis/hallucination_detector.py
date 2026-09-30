import re
from app.retrieval.reranker import get_shared_cross_encoder


class HallucinationDetector:
    """Claim-level evidence support checker.

    Scores are raw CrossEncoder logits; they are converted only into a bounded
    support-strength heuristic. The detector never generates factual content.
    """
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", support_threshold=-3.0, unsupported_ratio_threshold=0.25):
        self.model = get_shared_cross_encoder(model_name)
        self.support_threshold = float(support_threshold)
        self.unsupported_ratio_threshold = float(unsupported_ratio_threshold)
        print("Hallucination detection model loaded.")

    def verify(self, answer, evidence=None):
        evidence = list(evidence or [])
        raw_answer = str(answer or "").strip()

        # Check for explicit refusal or insufficient evidence statements
        insufficient_phrases = (
            "insufficient", "does not provide sufficient evidence",
            "cannot be established", "does not contain enough",
            "no sufficiently grounded claims",
        )
        if any(p in raw_answer.lower() for p in insufficient_phrases):
            return {
                "answer": raw_answer,
                "supported": True,
                "hallucination_detected": False,
                "grounding_status": "INSUFFICIENT_EVIDENCE",
                "support_score": 100.0,
                "unsupported_count": 0,
                "unsupported_ratio": 0.0,
                "claims": [],
            }

        claims = self._extract_claims(raw_answer)
        if not claims:
            return {
                "answer": raw_answer,
                "supported": False,
                "hallucination_detected": False,
                "grounding_status": "INSUFFICIENT_EVIDENCE",
                "support_score": 0.0,
                "unsupported_count": 0,
                "unsupported_ratio": 0.0,
                "claims": [],
            }

        evidence_texts = [self._evidence_text(x) for x in evidence]
        evidence_texts = [x for x in evidence_texts if x]
        if not evidence_texts:
            return {
                "answer": raw_answer,
                "supported": False,
                "hallucination_detected": True,
                "grounding_status": "UNSUPPORTED",
                "support_score": 0.0,
                "unsupported_count": len(claims),
                "unsupported_ratio": 1.0,
                "claims": [{"claim": c, "supported": False, "support_score": 0.0} for c in claims],
            }

        pairs = [[claim, ev] for claim in claims for ev in evidence_texts]
        scores = list(self.model.predict(pairs))
        results = []
        pos = 0
        for claim in claims:
            claim_scores = [float(x) for x in scores[pos:pos + len(evidence_texts)]]
            pos += len(evidence_texts)
            best = max(claim_scores) if claim_scores else -999.0
            # Monotonic bounded heuristic. 0 at/under -7, 100 at/above -1.
            strength = max(0.0, min(100.0, (best + 7.0) / 6.0 * 100.0))
            supported = best >= self.support_threshold
            results.append({"claim": claim, "supported": supported, "support_score": round(strength, 2), "raw_best_score": best})

        unsupported = sum(1 for x in results if not x["supported"])
        ratio = unsupported / max(1, len(results))
        overall = sum(x["support_score"] for x in results) / len(results)

        if unsupported == 0:
            is_supported = True
            grounding_status = "FULLY_SUPPORTED"
            hallucination_detected = False
        elif ratio <= self.unsupported_ratio_threshold and overall >= 55.0:
            is_supported = True
            grounding_status = "PARTIALLY_SUPPORTED"
            hallucination_detected = False
        else:
            is_supported = False
            grounding_status = "UNSUPPORTED"
            hallucination_detected = True

        return {
            "answer": raw_answer,
            "supported": is_supported,
            "grounding_status": grounding_status,
            "hallucination_detected": hallucination_detected,
            "support_score": round(overall, 2),
            "unsupported_count": unsupported,
            "unsupported_ratio": round(ratio, 3),
            "claims": results,
        }

    @staticmethod
    def _evidence_text(item):
        if isinstance(item, dict):
            return str(item.get("text") or "").strip()
        return str(getattr(item, "text", "") or "").strip()

    @staticmethod
    def _extract_claims(text):
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        if not text:
            return []
        return [p.strip(" -•\t") for p in re.split(r"(?<=[.!?])\s+|\n+", text) if len(p.strip()) > 12]
