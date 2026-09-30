from app.llm.client import LLMClient
from app.llm.context_builder import ContextBuilder
from app.models.schemas import (
    LimitationAnalysis,
    Citation
)


class LimitationAnalyzer:

    def __init__(self):

        self.llm = LLMClient(
            model="qwen3:1.7b"
        )

        self.context_builder = ContextBuilder(
            max_chunks=5
        )

    # ============================================================
    # SELECT LIMITATION EVIDENCE
    # ============================================================

    def _select_limitation_evidence(self, results):

        selected = []

        # --------------------------------------------------------
        # First priority:
        # exact limitations/future research section
        # --------------------------------------------------------

        for result in results:

            section = (
                result.get(
                    "section",
                    ""
                )
                or ""
            ).strip().lower()

            if (
                "limitations and future research"
                in section
            ):

                selected.append(
                    result
                )

        # --------------------------------------------------------
        # Second priority:
        # general discussion
        #
        # Only use this if we don't have the dedicated
        # limitations section.
        # --------------------------------------------------------

        if not selected:

            for result in results:

                section = (
                    result.get(
                        "section",
                        ""
                    )
                    or ""
                ).strip().lower()

                if (
                    "general discussion"
                    in section
                ):

                    selected.append(
                        result
                    )

        # --------------------------------------------------------
        # Final fallback:
        # use retrieved evidence.
        # --------------------------------------------------------

        if not selected:

            selected = results

        return selected

    # ============================================================
    # BUILD CONTEXT
    # ============================================================

    def _build_context(self, results):

        selected = (
            self._select_limitation_evidence(
                results
            )
        )

        return (
            self.context_builder.build(
                selected
            )
        )

    # ============================================================
    # BUILD CITATIONS
    # ============================================================

    def _build_citations(self, results):

        citations = []

        selected = (
            self._select_limitation_evidence(
                results
            )
        )

        for result in selected:

            citations.append(
                Citation(
                    paper_id=result.get(
                        "paper_id",
                        ""
                    ),

                    paper_title=result.get(
                        "paper_title",
                        ""
                    ),

                    page_start=result.get(
                        "page_start",
                        0
                    ),

                    page_end=result.get(
                        "page_end",
                        0
                    ),

                    section=result.get(
                        "section",
                        ""
                    ),

                    chunk_id=result.get(
                        "chunk_id",
                        ""
                    )
                )
            )

        return citations

    # ============================================================
    # ANALYZE
    # ============================================================

    def analyze(self, results):

        # --------------------------------------------------------
        # Use only limitation-focused evidence.
        # --------------------------------------------------------

        selected_results = (
            self._select_limitation_evidence(
                results
            )
        )

        context = (
            self.context_builder.build(
                selected_results
            )
        )

        # --------------------------------------------------------
        # Strict extraction instructions
        # --------------------------------------------------------

        instructions = """

You are extracting limitations and future research
from an academic research paper.

IMPORTANT:

Use ONLY the supplied evidence.

The supplied evidence comes from the paper's
limitations/future-research discussion.

Your job is extraction, NOT reasoning.

DO NOT:

- invent information
- infer information
- calculate anything
- use outside knowledge
- use information from other parts of the paper
- create generic limitations
- assume a sample is small
- assume a methodology is weak
- assume self-report is a limitation unless the
  authors explicitly state it
- assume a technology limitation
- create future research suggestions yourself

Every item must be directly supported by the
supplied evidence.

==================================================
LIMITATIONS
==================================================

List the limitations explicitly acknowledged
by the authors.

==================================================
GENERALIZABILITY ISSUES
==================================================

List only explicit concerns about:

- culture
- country
- population
- sampling
- sample diversity
- generalizability
- product/category scope

==================================================
METHODOLOGICAL LIMITATIONS
==================================================

List only methodological limitations explicitly
acknowledged by the authors.

Examples include:

- self-reported measures
- scenario-based experiments
- lack of actual behavioral data
- missing psychological variables

ONLY include these if the authors explicitly
identify them.

==================================================
FUTURE DIRECTIONS
==================================================

List only research directions explicitly proposed
by the authors.

Do NOT invent suggestions.

==================================================
STYLE
==================================================

Keep each item concise.

Do not include sample sizes unless the authors
explicitly discuss sample size as a limitation.

Do not include citations.

Python will generate citations separately.

Return ONLY the requested structured output.

"""

        input_text = f"""

Extract the limitations and future research
directions from this evidence.

================ EVIDENCE ================

{context}

============== END EVIDENCE ==============

Again:

Use ONLY this evidence.

Do not infer.

Do not invent.

"""

        # --------------------------------------------------------
        # Structured Qwen output
        # --------------------------------------------------------

        raw_response = (
            self.llm.generate_structured(
                instructions=instructions,
                input_text=input_text,
                schema=LimitationAnalysis
            )
        )

        # --------------------------------------------------------
        # Validate structured response
        # --------------------------------------------------------

        analysis = (
            LimitationAnalysis.model_validate_json(
                raw_response
            )
        )

        # --------------------------------------------------------
        # Remove any citations generated by Qwen.
        # --------------------------------------------------------

        analysis.citations = []

        # --------------------------------------------------------
        # Add citations from actual evidence.
        # --------------------------------------------------------

        analysis.citations = (
            self._build_citations(
                selected_results
            )
        )

        return analysis
