from typing import Any, Dict


class CitationGenerator:
    """
    Creates citations only from grounded claim/evidence provenance.
    No citation metadata is invented.
    """

    def __init__(self):
        print("Citation generation pipeline ready.")

    def generate(self, claims=None, evidence=None) -> Dict[str, Any]:
        claims = list(claims or [])
        evidence = list(evidence or [])

        evidence_map = {
            str(item.get("chunk_id")): item
            for item in evidence
            if item.get("chunk_id")
        }

        claim_citations = []
        source_keys = set()
        sources = []

        for claim in claims:
            citation_records = []
            evidence_ids = self._get(claim, "evidence_ids", []) or []
            citations = self._get(claim, "citations", []) or []

            # Prefer the citation already attached by ClaimGrounder.
            for citation in citations:
                record = self._citation_to_dict(citation)
                if record.get("paper_id") or record.get("chunk_id"):
                    citation_records.append(record)

            # Fill missing provenance from evidence.
            if not citation_records:
                for eid in evidence_ids:
                    item = evidence_map.get(str(eid))
                    if item:
                        citation_records.append(
                            self._citation_from_evidence(item)
                        )

            # De-duplicate citations for this claim.
            citation_records = self._dedupe_records(citation_records)

            for record in citation_records:
                key = (
                    record.get("paper_id", ""),
                    record.get("chunk_id", ""),
                    record.get("page_start", 0),
                    record.get("page_end", 0),
                )
                if key in source_keys:
                    continue
                source_keys.add(key)
                sources.append(record)

            claim_citations.append(
                {
                    "claim": self._get(claim, "claim", ""),
                    "citations": citation_records,
                }
            )

        return {
            "claim_citations": claim_citations,
            "sources": sources,
            "source_count": len(sources),
        }

    @staticmethod
    def _get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @classmethod
    def _citation_to_dict(cls, citation):
        return cls._normalize(
            {
                "paper_id": cls._get(citation, "paper_id", ""),
                "paper_title": cls._get(citation, "paper_title", ""),
                "page_start": cls._get(citation, "page_start", 0),
                "page_end": cls._get(citation, "page_end", 0),
                "section": cls._get(citation, "section", ""),
                "chunk_id": cls._get(citation, "chunk_id", ""),
            }
        )

    @classmethod
    def _citation_from_evidence(cls, item):
        return cls._normalize(
            {
                "paper_id": item.get("paper_id", ""),
                "paper_title": item.get("paper_title", ""),
                "page_start": item.get("page_start", 0),
                "page_end": item.get("page_end", 0),
                "section": item.get("section", ""),
                "chunk_id": item.get("chunk_id", ""),
            }
        )

    @staticmethod
    def _normalize(record):
        def integer(value):
            try:
                return int(value or 0)
            except (TypeError, ValueError):
                return 0

        return {
            "paper_id": str(record.get("paper_id", "") or ""),
            "paper_title": str(record.get("paper_title", "") or ""),
            "page_start": integer(record.get("page_start")),
            "page_end": integer(record.get("page_end")),
            "section": str(record.get("section", "") or ""),
            "chunk_id": str(record.get("chunk_id", "") or ""),
        }

    @staticmethod
    def _dedupe_records(records):
        out = []
        seen = set()
        for record in records:
            key = (
                record.get("paper_id", ""),
                record.get("chunk_id", ""),
                record.get("page_start", 0),
                record.get("page_end", 0),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(record)
        return out
