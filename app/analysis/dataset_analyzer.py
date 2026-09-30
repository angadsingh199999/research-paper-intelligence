from app.llm.client import LLMClient
from app.llm.context_builder import ContextBuilder
from app.models.schemas import DatasetAnalysis, Citation


class DatasetAnalyzer:

    def __init__(self):

        self.llm = LLMClient(
            model="qwen3:1.7b"
        )

        self.context_builder = ContextBuilder(
            max_chunks=5
        )

    # ============================================================
    # BUILD CITATIONS FROM REAL RETRIEVED EVIDENCE
    # ============================================================

    def _build_citations(self, results):

        citations = []

        for result in results:

            citation = Citation(
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

            citations.append(citation)

        return citations

    # ============================================================
    # ANALYZE DATASET / DATA SOURCES
    # ============================================================

    def analyze(self, results):

        # --------------------------------------------------------
        # 1. Build evidence
        # --------------------------------------------------------

        context = self.context_builder.build(
            results
        )

        # --------------------------------------------------------
        # 2. Instructions
        # --------------------------------------------------------

        instructions = """

You are a research paper data extraction system.

Your ONLY source is the EVIDENCE supplied below.

Extract ONLY information explicitly written in the evidence.

Do NOT use outside knowledge.

Do NOT guess.

Do NOT infer a dataset name.

Do NOT invent a dataset.

Do NOT invent participant numbers.

Do NOT invent sample sizes.

Do NOT invent data sources.

Do NOT invent citations.

IMPORTANT:

A "dataset" means a named dataset or database explicitly
used by the researchers.

Examples:
- ImageNet
- COCO
- MNIST
- Kaggle dataset

If the paper does NOT explicitly mention a named dataset,
return:

"datasets": []

A "data source" means where the research data came from.

Examples:
- online survey
- Qualtrics
- interviews
- experiments
- observations
- secondary database
- email recruitment
- social media recruitment

Extract these when explicitly stated.

"sample_description" should contain explicit sample
information from the evidence.

For example:

"250 responses were collected and 234 were retained."

If multiple studies have different samples, include them
separately.

Do not combine different sample sizes into one number.

Return ONLY valid JSON matching the supplied schema.

The citations field MUST ALWAYS be [].

Python will construct citations separately from the real
retrieved evidence.

"""

        # --------------------------------------------------------
        # 3. Input
        # --------------------------------------------------------

        input_text = f"""

Extract the dataset and data-source information.

Use ONLY the evidence below.

================ EVIDENCE ================

{context}

============== END EVIDENCE ==============

Return:

1. datasets
   - Named datasets/databases explicitly mentioned.
   - If none are explicitly mentioned, return [].

2. data_sources
   - Explicit sources or collection channels for the research data.

3. sample_description
   - Explicit sample/response information.
   - Preserve different study sample sizes separately.

4. citations
   - MUST be [].

Do not explain your answer.

Return JSON only.

"""

        # --------------------------------------------------------
        # 4. Generate structured response
        # --------------------------------------------------------

        raw_response = self.llm.generate_structured(
            instructions=instructions,
            input_text=input_text,
            schema=DatasetAnalysis
        )

        # --------------------------------------------------------
        # 5. Parse response
        # --------------------------------------------------------

        analysis = DatasetAnalysis.model_validate_json(
            raw_response
        )

        # --------------------------------------------------------
        # 6. Never trust LLM-generated citations
        # --------------------------------------------------------

        analysis.citations = []

        # --------------------------------------------------------
        # 7. Attach citations generated from real evidence
        # --------------------------------------------------------

        analysis.citations = self._build_citations(
            results
        )

        return analysis
