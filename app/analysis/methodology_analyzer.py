from app.llm.client import LLMClient

from app.models.schemas import (
    MethodologyAnalysis,
    Citation,
    GeneralMethodologyExtraction,
    PretestExtraction,
    StudyExtraction
)


class MethodologyAnalyzer:

    def __init__(self):

        self.llm = LLMClient(
            model="qwen3:1.7b"
        )

    # ========================================================
    # BUILD EVIDENCE TEXT
    # ========================================================

    def _build_evidence(self, results):

        if not results:
            return "No evidence available."

        parts = []

        for index, result in enumerate(
            results,
            start=1
        ):

            parts.append(
                f"""
EVIDENCE {index}

Section: {result.get("section", "")}

Pages: {result.get("page_start", "")}-{result.get("page_end", "")}

Text:
{result.get("text", "")}
""".strip()
            )

        return (
            "\n\n"
            + "\n\n------------------------------\n\n".join(parts)
        )

    # ========================================================
    # GENERIC EXTRACTION
    # ========================================================

    def _extract(
        self,
        evidence,
        schema,
        fields
    ):

        field_list = "\n".join(
            f"- {field}"
            for field in fields
        )

        instructions = """
You are a factual research-paper extraction system.

Use ONLY the supplied evidence.

Do NOT use outside knowledge.

Do NOT guess.

Do NOT infer missing information.

Do NOT combine information from different studies.

Copy participant numbers exactly.

Copy brand names exactly.

Copy dates exactly.

Copy statistical methods exactly when explicitly stated.

If information is not explicitly present,
return null.

Return ONLY valid JSON.
"""

        input_text = f"""
Extract ONLY these fields:

{field_list}

================ EVIDENCE ================

{evidence}

============== END EVIDENCE ==============

Every value must be directly supported
by the supplied evidence.
"""

        raw = self.llm.generate_structured(
            instructions=instructions,
            input_text=input_text,
            schema=schema
        )

        return schema.model_validate_json(
            raw
        )

    # ========================================================
    # BUILD CITATIONS
    # ========================================================

    def _build_citations(self, results):

        citations = []

        for result in results:

            chunk_id = result.get(
                "chunk_id",
                ""
            )

            if not chunk_id:
                continue

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
                    chunk_id=chunk_id
                )
            )

        return citations

    # ========================================================
    # MAIN ANALYSIS
    # ========================================================

    def analyze(self, results):

        # ----------------------------------------------------
        # 1. Separate evidence by section
        # ----------------------------------------------------

        pretest_results = []
        study1_results = []
        study2_results = []
        general_results = []

        for result in results:

            section = (
                result.get("section", "")
                or ""
            ).lower().strip()

            if section.startswith("4.1"):
                pretest_results.append(result)

            elif section.startswith("4.3"):
                study1_results.append(result)

            elif section.startswith("4.4"):
                study2_results.append(result)

            elif section.startswith("4.2"):
                general_results.append(result)

        # ----------------------------------------------------
        # 2. GENERAL METHODOLOGY
        # ----------------------------------------------------

        general_evidence = self._build_evidence(
            general_results
        )

        general = self._extract(
            general_evidence,
            GeneralMethodologyExtraction,
            [
                "methodology_type",
                "research_design",
                "studies",
                "sampling_method",
                "data_collection"
            ]
        )

        # ----------------------------------------------------
        # 3. PRE-TEST
        # ----------------------------------------------------

        pretest = self._extract(
            self._build_evidence(
                pretest_results
            ),
            PretestExtraction,
            [
                "sample_size",
                "experimental_conditions",
                "data_collection",
                "analysis_method",
                "purpose"
            ]
        )

        # ----------------------------------------------------
        # 4. STUDY 1
        # ----------------------------------------------------

        study1 = self._extract(
            self._build_evidence(
                study1_results
            ),
            StudyExtraction,
            [
                "sample_size",
                "experimental_conditions",
                "product_category",
                "data_collection",
                "sampling_method",
                "analysis_method"
            ]
        )

        # ----------------------------------------------------
        # 5. STUDY 2
        # ----------------------------------------------------

        study2 = self._extract(
            self._build_evidence(
                study2_results
            ),
            StudyExtraction,
            [
                "sample_size",
                "experimental_conditions",
                "product_category",
                "data_collection",
                "sampling_method",
                "analysis_method"
            ]
        )

        # ----------------------------------------------------
        # 6. COMBINE STUDY INFORMATION
        # ----------------------------------------------------

        analysis_data = {

            "methodology_type":
                general.methodology_type,

            "research_design":
                general.research_design,

            "studies":
                general.studies,

            "pretest":
                pretest.model_dump(),

            "study_1":
                study1.model_dump(),

            "study_2":
                study2.model_dump(),

            "sampling_method":
                general.sampling_method,

            "data_collection":
                general.data_collection,

            "analysis_method":
                (
                    study1.analysis_method
                    or study2.analysis_method
                ),

            "key_methodological_details": [],

            "citations": []
        }

        # ----------------------------------------------------
        # 7. FINAL PYDANTIC VALIDATION
        # ----------------------------------------------------

        analysis = MethodologyAnalysis(
            **analysis_data
        )

        # ----------------------------------------------------
        # 8. CITATIONS ARE ALWAYS PYTHON-GENERATED
        # ----------------------------------------------------

        analysis.citations = (
            self._build_citations(results)
        )

        return analysis
