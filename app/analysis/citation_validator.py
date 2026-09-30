from app.models.schemas import Citation


class CitationValidator:

    def validate(
        self,
        citations,
        evidence
    ):
        """
        Validate citations against the actual
        retrieved evidence.

        A citation is valid only if its:
        - paper_id
        - chunk_id
        - page range
        - section

        correspond to retrieved evidence.
        """

        valid_citations = []
        invalid_citations = []

        # ----------------------------------------
        # Build lookup from real evidence
        # ----------------------------------------

        evidence_lookup = {}

        for item in evidence:

            chunk_id = item.get(
                "chunk_id",
                ""
            )

            if not chunk_id:
                continue

            evidence_lookup[chunk_id] = item

        # ----------------------------------------
        # Validate each citation
        # ----------------------------------------

        for citation in citations:

            if isinstance(
                citation,
                Citation
            ):
                citation_data = citation

            else:
                try:
                    citation_data = Citation(
                        **citation
                    )
                except Exception:
                    invalid_citations.append(
                        citation
                    )
                    continue

            chunk_id = citation_data.chunk_id

            # ------------------------------------
            # Chunk must exist in evidence
            # ------------------------------------

            if chunk_id not in evidence_lookup:

                invalid_citations.append(
                    citation_data
                )

                continue

            source = evidence_lookup[
                chunk_id
            ]

            # ------------------------------------
            # Validate paper ID
            # ------------------------------------

            if (
                citation_data.paper_id
                != source.get(
                    "paper_id",
                    ""
                )
            ):

                invalid_citations.append(
                    citation_data
                )

                continue

            # ------------------------------------
            # Validate pages
            # ------------------------------------

            source_page_start = source.get(
                "page_start"
            )

            source_page_end = source.get(
                "page_end"
            )

            if (
                citation_data.page_start
                != source_page_start
                or
                citation_data.page_end
                != source_page_end
            ):

                invalid_citations.append(
                    citation_data
                )

                continue

            # ------------------------------------
            # Validate section
            # ------------------------------------

            if (
                citation_data.section
                != source.get(
                    "section",
                    ""
                )
            ):

                invalid_citations.append(
                    citation_data
                )

                continue

            # ------------------------------------
            # Citation is valid
            # ------------------------------------

            valid_citations.append(
                citation_data
            )

        return {
            "valid": valid_citations,
            "invalid": invalid_citations,
            "all_valid": (
                len(invalid_citations) == 0
            )
        }
