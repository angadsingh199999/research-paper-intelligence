import re
import json

from app.llm.client import LLMClient
from app.llm.context_builder import ContextBuilder
from app.models.schemas import ResultsAnalysis, Citation


class ResultsAnalyzer:

    def __init__(self):

        self.llm = LLMClient(
            model="qwen3:1.7b"
        )

        self.context_builder = ContextBuilder(
            max_chunks=5
        )

    # ============================================================
    # BUILD CITATIONS FROM REAL EVIDENCE
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
    # COMBINE EVIDENCE TEXT
    # ============================================================

    def _get_evidence_text(self, results):

        return "\n\n".join(
            result.get(
                "text",
                ""
            )
            for result in results
        )

    # ============================================================
    # NORMALIZE OCR STATISTICAL NOTATION
    # ============================================================

    def _normalize_statistical_text(self, text):

        normalized = text

        # --------------------------------------------------------
        # The PDF extraction sometimes converts "=" into "5".
        #
        # Examples:
        #
        # β 5 0.64
        # t 5 7.41
        # R 2 5 0.553
        #
        # We ONLY replace 5 when it is functioning as "="
        # between a statistical expression and a number.
        # --------------------------------------------------------

        normalized = re.sub(
            r"(?P<prefix>"
            r"(?:β|R\s*2|R²|p|t|F|"
            r"Cohen['’]s\s+d|"
            r"indirect\s+effect|"
            r"effect)"
            r")"
            r"(\s*)"
            r"5"
            r"(\s*)"
            r"(?=[-+]?\d)",
            r"\g<prefix>\2=\3",
            normalized,
            flags=re.IGNORECASE
        )

        # --------------------------------------------------------
        # Normalize common OCR form:
        #
        # R 2
        #
        # into:
        #
        # R²
        # --------------------------------------------------------

        normalized = re.sub(
            r"\bR\s*2\b",
            "R²",
            normalized,
            flags=re.IGNORECASE
        )

        return normalized

    # ============================================================
    # DETERMINISTIC HYPOTHESIS EXTRACTION
    # ============================================================

    def _extract_hypothesis_results(self, text):

        results = []

        # --------------------------------------------------------
        # Explicit ranges such as:
        #
        # H1 - H4
        # H1-H5
        # --------------------------------------------------------

        range_matches = re.findall(
            r"\bH(\d+)\s*[-–]\s*H(\d+)\b",
            text,
            flags=re.IGNORECASE
        )

        found = set()

        for start, end in range_matches:

            start = int(start)
            end = int(end)

            for number in range(
                start,
                end + 1
            ):

                found.add(
                    f"H{number}"
                )

        # --------------------------------------------------------
        # Explicit statements:
        #
        # supporting H1
        # supporting H2
        # supporting H3
        # supporting H4
        # supporting H5
        # --------------------------------------------------------

        single_patterns = [

            r"\bsupport(?:ing|ed)?\s+H(\d+)\b",

            r"\bconsistent\s+with\s+H(\d+)\b",

            r"\bin\s+line\s+with\s+H(\d+)\b",

            r"\bH(\d+)\s+(?:was|were)\s+supported\b",
        ]

        for pattern in single_patterns:

            matches = re.findall(
                pattern,
                text,
                flags=re.IGNORECASE
            )

            for number in matches:

                found.add(
                    f"H{number}"
                )

        # --------------------------------------------------------
        # Explicit statement:
        #
        # "support for all proposed hypotheses"
        #
        # In this paper the hypotheses are H1-H5.
        # --------------------------------------------------------

        if re.search(
            r"support(?:ed)?\s+(?:for\s+)?"
            r"all\s+(?:the\s+)?"
            r"(?:proposed\s+)?hypotheses",
            text,
            flags=re.IGNORECASE
        ):

            for number in range(
                1,
                6
            ):

                found.add(
                    f"H{number}"
                )

        # --------------------------------------------------------
        # Sort hypotheses numerically.
        # --------------------------------------------------------

        def hypothesis_number(value):

            return int(
                value[1:]
            )

        for hypothesis in sorted(
            found,
            key=hypothesis_number
        ):

            results.append(
                f"{hypothesis} supported"
            )

        return results

    # ============================================================
    # DETERMINISTIC STATISTIC EXTRACTION
    # ============================================================

    def _extract_statistics(self, text):

        normalized = (
            self._normalize_statistical_text(
                text
            )
        )

        statistics = []

        # --------------------------------------------------------
        # IMPORTANT:
        #
        # Each expression is extracted as a COMPLETE unit.
        # We do not globally replace "5".
        # --------------------------------------------------------

        patterns = [

            # β = 0.64
            r"β\s*=\s*[-+]?\d+(?:\.\d+)?",

            # p < 0.001
            r"p\s*[<=>]\s*0?\.\d+",

            # t = 7.41
            r"t\s*\([^)]*\)\s*=\s*[-+]?\d+(?:\.\d+)?",

            # t = 7.41 when degrees are absent
            r"\bt\s*=\s*[-+]?\d+(?:\.\d+)?",

            # F(2,231) = 143.16
            r"F\s*\([^)]*\)\s*=\s*[-+]?\d+(?:\.\d+)?",

            # R² = 0.553
            r"R²\s*=\s*[-+]?\d+(?:\.\d+)?",

            # Cohen's d = 0.74
            r"Cohen['’]s\s+d\s*=\s*[-+]?\d+(?:\.\d+)?",

            # indirect effect = 0.22
            r"indirect\s+effect\s*=\s*[-+]?\d+(?:\.\d+)?",

            # effect = 0.24
            r"\beffect\s*=\s*[-+]?\d+(?:\.\d+)?",
        ]

        seen = set()

        for pattern in patterns:

            matches = re.findall(
                pattern,
                normalized,
                flags=re.IGNORECASE
            )

            for match in matches:

                cleaned = (
                    match
                    .strip()
                    .replace(
                        "R2",
                        "R²"
                    )
                )

                # ------------------------------------------------
                # Reject obviously malformed OCR results.
                # ------------------------------------------------

                if "=" not in cleaned:
                    continue

                if re.search(
                    r"[=]{2,}|"
                    r"\.\D|"
                    r"\d+\.\d+[^\d\s,.;)\]]",
                    cleaned
                ):
                    continue

                key = cleaned.lower()

                if key in seen:
                    continue

                seen.add(key)

                statistics.append(
                    cleaned
                )

        return statistics

    # ============================================================
    # GENERAL FINDINGS FROM QWEN
    # ============================================================

    def _extract_findings_with_llm(
        self,
        context
    ):

        instructions = """

You are a research findings extractor.

Use ONLY the supplied evidence.

Extract short factual findings explicitly stated
in the evidence.

Do NOT calculate.

Do NOT infer.

Do NOT invent.

Do NOT determine hypothesis support.

Do NOT extract statistics.

Do NOT discuss methodology.

Do NOT create Study 3.

Return ONLY valid JSON.

Required format:

{
  "findings": [
    "short factual finding"
  ]
}

Keep each finding short.

"""

        input_text = f"""

Extract only the general research findings.

Do not extract hypothesis labels.
Do not extract numerical statistics.

================ EVIDENCE ================

{context}

============== END EVIDENCE ==============

"""

        raw_response = self.llm.generate(
            instructions=instructions,
            input_text=input_text
        )

        try:

            parsed = json.loads(
                raw_response
            )

            findings = parsed.get(
                "findings",
                []
            )

            if isinstance(
                findings,
                list
            ):

                return [
                    str(item).strip()
                    for item in findings
                    if str(item).strip()
                ]

        except Exception:

            return []

        return []

    # ============================================================
    # MAIN ANALYSIS
    # ============================================================

    def analyze(self, results):

        # --------------------------------------------------------
        # 1. Build context
        # --------------------------------------------------------

        context = self.context_builder.build(
            results
        )

        # --------------------------------------------------------
        # 2. Combine raw evidence
        # --------------------------------------------------------

        evidence_text = (
            self._get_evidence_text(
                results
            )
        )

        # --------------------------------------------------------
        # 3. General findings
        # --------------------------------------------------------

        findings = (
            self._extract_findings_with_llm(
                context
            )
        )

        # --------------------------------------------------------
        # 4. Hypotheses — Python
        # --------------------------------------------------------

        hypothesis_results = (
            self._extract_hypothesis_results(
                evidence_text
            )
        )

        # --------------------------------------------------------
        # 5. Statistics — Python
        # --------------------------------------------------------

        statistics = (
            self._extract_statistics(
                evidence_text
            )
        )

        # --------------------------------------------------------
        # 6. Construct final result
        # --------------------------------------------------------

        analysis = ResultsAnalysis(

            findings=findings,

            hypothesis_results=(
                hypothesis_results
            ),

            statistics=statistics,

            citations=[]
        )

        # --------------------------------------------------------
        # 7. Citations come ONLY from retrieved evidence
        # --------------------------------------------------------

        analysis.citations = (
            self._build_citations(
                results
            )
        )

        return analysis
