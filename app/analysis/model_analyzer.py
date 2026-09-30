from app.llm.client import LLMClient
from app.llm.context_builder import ContextBuilder
from app.models.schemas import (
    ModelAnalysis,
    Citation
)


class ModelAnalyzer:

    def __init__(self):

        self.llm = LLMClient(
            model="qwen3:1.7b"
        )

        self.context_builder = ContextBuilder(
            max_chunks=5
        )

    # ============================================================
    # BUILD CITATIONS
    # ============================================================

    def _build_citations(self, results):

        citations = []

        for result in results:

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
    # SELECT MODEL EVIDENCE
    # ============================================================

    def _select_model_evidence(self, results):

        selected = []

        preferred_terms = [
            "conceptual model",
            "theoretical background",
            "methodology",
            "model",
            "results and discussion"
        ]

        for result in results:

            section = (
                result.get(
                    "section",
                    ""
                )
                or ""
            ).lower()

            if any(
                term in section
                for term in preferred_terms
            ):

                selected.append(
                    result
                )

        if not selected:

            selected = results

        return selected

    # ============================================================
    # ANALYZE
    # ============================================================

    def analyze(self, results):

        selected_results = (
            self._select_model_evidence(
                results
            )
        )

        context = (
            self.context_builder.build(
                selected_results
            )
        )

        instructions = """

You are extracting models, algorithms, and
architectures from an academic research paper.

Use ONLY the supplied evidence.

Do NOT use outside knowledge.

Do NOT invent anything.

Do NOT assume that every paper contains
machine-learning models.

==================================================
MODELS
==================================================

Extract explicitly named research, theoretical,
statistical, computational, or conceptual models.

Examples:

- Technology Acceptance Model
- Experiential Technology Acceptance Model
- conceptual model
- regression model
- mediation model
- moderated mediation model

Only include a model if it is explicitly
supported by the evidence.

==================================================
ALGORITHMS
==================================================

Extract explicitly named analytical algorithms,
procedures, or computational methods.

Examples:

- PROCESS Macro Model 4
- PROCESS Macro Model 7
- random forest
- support vector machine
- k-means

Do not turn ordinary statistical tests into
algorithms unless the paper explicitly treats
them as an analytical method.

==================================================
ARCHITECTURES
==================================================

Extract explicitly named technical architectures.

Examples:

- CNN
- ResNet
- Transformer
- LSTM
- U-Net

If the paper does not contain a technical
architecture, return an empty list.

==================================================
IMPORTANT
==================================================

This may be a behavioral, social-science,
medical, engineering, or machine-learning paper.

Do not force ML terminology onto a paper.

Every item must be supported by the evidence.

Do not include explanations.

Do not include citations.

Python will generate citations.

Return ONLY valid structured output.

"""

        input_text = f"""

Extract models, algorithms, and architectures
from the following evidence.

================ EVIDENCE ================

{context}

============== END EVIDENCE ==============

Remember:

Use only the supplied evidence.

Do not invent technical models or architectures.

"""

        raw_response = (
            self.llm.generate_structured(
                instructions=instructions,
                input_text=input_text,
                schema=ModelAnalysis
            )
        )

        analysis = (
            ModelAnalysis.model_validate_json(
                raw_response
            )
        )

        # Never trust LLM-generated citations.

        analysis.citations = (
            self._build_citations(
                selected_results
            )
        )

        return analysis
