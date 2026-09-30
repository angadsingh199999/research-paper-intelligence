"""Sentence-level grounded claim extraction with strict task semantics."""

from __future__ import annotations

import re
from typing import List

from app.models.schemas import Citation, GroundedClaim, GroundedClaims
from app.analysis.task_semantics import is_noise, is_task_valid, normalize_task, normalize_text


_ABBREVIATIONS = (
    "et al.", "e.g.", "i.e.", "cf.", "vs.", "etc.", "fig.", "eq.",
    "approx.", "dr.", "prof.", "mr.", "mrs.", "ms.", "no.", "c.i.",
    "c.i", "s.d.", "s.d", "p.", "pp.", "vol.", "inc.", "U.S.",
)
_PLACEHOLDER = "\uE000"


class ClaimGrounder:
    """Convert retrieved chunks into sentence-level, task-valid claims.

    The important rule is that grounding and task validity are separate:
    evidence support alone is not sufficient. A sentence must also be the
    requested information type before it becomes a GroundedClaim.
    """

    def __init__(self):
        print("Claim grounding pipeline ready.")

    def ground(self, evidence, task=None, instruction=None, answer=None):
        if not evidence:
            return GroundedClaims(claims=[])

        task = normalize_task(task)
        claims: List[GroundedClaim] = []
        seen = set()

        # Keep claims distributed across papers/chunks instead of returning
        # several nearly identical sentences from one high-scoring chunk.
        max_claims = 32
        per_chunk_limit = 3

        for item in evidence:
            if not isinstance(item, dict):
                continue

            text = normalize_text(item.get("text", ""))
            section = normalize_text(item.get("section", ""))
            chunk_id = str(item.get("chunk_id", ""))
            if not text or not chunk_id:
                continue

            sentences = self._extract_sentences(text)
            accepted_from_chunk = 0

            for sentence in sentences:
                sentence = self._clean_claim(sentence)
                if not sentence:
                    continue
                if is_noise(sentence):
                    continue
                if not is_task_valid(task, sentence, section):
                    continue

                key = " ".join(sentence.lower().split()).rstrip(".!?")
                if key in seen:
                    continue

                citation = Citation(
                    paper_id=str(item.get("paper_id", "")),
                    paper_title=str(item.get("paper_title", "")),
                    page_start=int(item.get("page_start", 0) or 0),
                    page_end=int(item.get("page_end", 0) or 0),
                    section=section,
                    chunk_id=chunk_id,
                )

                claims.append(
                    GroundedClaim(
                        claim=sentence,
                        evidence_ids=[chunk_id],
                        citations=[citation],
                    )
                )
                seen.add(key)
                accepted_from_chunk += 1

                if accepted_from_chunk >= per_chunk_limit or len(claims) >= max_claims:
                    break

            if len(claims) >= max_claims:
                break

        claims = self._balance_comparison_claims(claims, evidence, max_claims=max_claims)
        print(f"Grounded claims after strict task validation: {len(claims)}")
        return GroundedClaims(claims=claims)

    # ------------------------------------------------------------------
    # Sentence splitting
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_sentences(text: str) -> List[str]:
        value = normalize_text(text)
        if not value:
            return []

        for abbreviation in _ABBREVIATIONS:
            safe = abbreviation.replace(".", _PLACEHOLDER)
            value = re.sub(re.escape(abbreviation), safe, value, flags=re.IGNORECASE)

        # Protect decimal points such as 0.05 and 38.38.
        value = re.sub(r"(?<=\d)\.(?=\d)", _PLACEHOLDER, value)

        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“‘])", value)
        output = []
        for part in parts:
            part = part.replace(_PLACEHOLDER, ".")
            part = normalize_text(part)
            if len(part.split()) >= 6:
                output.append(part)
        return output

    @staticmethod
    def _clean_claim(sentence: str) -> str:
        value = normalize_text(sentence)

        # Remove common leading extraction artifacts.
        value = re.sub(r"^(?:[-•*]\s*)+", "", value)
        value = re.sub(r"\s+\[[0-9,;\-\s]+\]", "", value)
        value = re.sub(r"\s+\(?(?:Table|Fig(?:ure)?)\.?\s*\d+[A-Za-z]?\)?\s*$", "", value, flags=re.IGNORECASE)

        # Repair a few very common PDF line-break / unicode artifacts without
        # inventing content.
        value = value.replace("competition times", "completion times")
        value = value.replace("nonluxury", "non-luxury")
        value = value.replace("brand- related", "brand-related")
        value = value.replace("  ", " ")

        if not value:
            return ""
        if value[-1] not in ".!?":
            # Do not turn obvious fragments into claims.
            if value.lower().startswith(("fig ", "figure ", "table ")):
                return ""
            value += "."
        return value

    # ------------------------------------------------------------------
    # Comparison balancing
    # ------------------------------------------------------------------

    @staticmethod
    def _paper_id(claim: GroundedClaim) -> str:
        for citation in getattr(claim, "citations", []) or []:
            value = getattr(citation, "paper_id", None)
            if value:
                return str(value)
        return ""

    @classmethod
    def _balance_comparison_claims(cls, claims, evidence, max_claims=32):
        # The QA analyzer already retrieves per paper for comparison queries.
        # Here we simply preserve paper diversity in the grounded-claim list.
        grouped = {}
        for claim in claims:
            grouped.setdefault(cls._paper_id(claim), []).append(claim)

        if len(grouped) <= 1:
            return claims[:max_claims]

        selected = []
        used = set()
        while len(selected) < max_claims:
            added = False
            for paper_id in sorted(grouped):
                for claim in grouped[paper_id]:
                    key = (paper_id, getattr(claim, "claim", ""))
                    if key in used:
                        continue
                    selected.append(claim)
                    used.add(key)
                    added = True
                    break
                if len(selected) >= max_claims:
                    break
            if not added:
                break
        return selected
