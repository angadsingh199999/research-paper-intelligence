from typing import Any, Dict, List


class MultiQuestionEvaluator:
    """
    Stage 19:
    Multi-question evaluation.

    Runs multiple research questions through the existing
    QAAnalyzer and collects structured evaluation results.

    This component does not change the QA pipeline.
    It only evaluates its outputs across different questions.
    """

    def __init__(self, qa_analyzer):
        self.qa_analyzer = qa_analyzer

    # ============================================================
    # PUBLIC METHOD
    # ============================================================

    def evaluate(
        self,
        questions: List[str],
    ) -> Dict[str, Any]:

        questions = [
            str(question).strip()
            for question in questions
            if str(question).strip()
        ]

        results = []

        for index, question in enumerate(
            questions,
            start=1
        ):

            print("\n" + "=" * 80)
            print(
                f"MULTI-QUESTION EVALUATION {index}"
            )
            print("=" * 80)

            print(
                "Question:",
                question
            )

            try:

                result = self.qa_analyzer.answer(
                    question
                )

                confidence = result.get(
                    "confidence",
                    {}
                )

                hallucination = result.get(
                    "hallucination",
                    {}
                )

                claims = result.get(
                    "claims",
                    []
                )

                evidence = result.get(
                    "evidence",
                    []
                )

                citations = result.get(
                    "citations",
                    []
                )

                evaluation = {

                    "question": question,

                    "answer": result.get(
                        "answer",
                        ""
                    ),

                    "task": result.get(
                        "task",
                        ""
                    ),

                    "claim_count": len(
                        claims
                    ),

                    "evidence_count": len(
                        evidence
                    ),

                    "citation_count": len(
                        citations
                    ),

                    "hallucination": hallucination,

                    "confidence": confidence,

                    "answer_generated": bool(
                        result.get(
                            "answer",
                            ""
                        ).strip()
                    ),

                    "claims_returned": bool(
                        claims
                    ),

                    "evidence_returned": bool(
                        evidence
                    ),

                    "citations_returned": bool(
                        citations
                    ),

                    "confidence_returned": bool(
                        confidence
                    ),
                }

                results.append(
                    evaluation
                )

                print(
                    "Evaluation completed."
                )

            except Exception as exc:

                evaluation = {

                    "question": question,

                    "answer": "",

                    "task": "",

                    "claim_count": 0,

                    "evidence_count": 0,

                    "citation_count": 0,

                    "hallucination": {},

                    "confidence": {},

                    "answer_generated": False,

                    "claims_returned": False,

                    "evidence_returned": False,

                    "citations_returned": False,

                    "confidence_returned": False,

                    "error": str(exc),
                }

                results.append(
                    evaluation
                )

                print(
                    "Evaluation failed:",
                    exc
                )

        # ========================================================
        # SUMMARY
        # ========================================================

        total = len(
            results
        )

        successful = sum(
            1
            for item in results
            if item.get(
                "answer_generated",
                False
            )
        )

        grounded = sum(
            1
            for item in results
            if item.get(
                "claims_returned",
                False
            )
            and item.get(
                "evidence_returned",
                False
            )
        )

        cited = sum(
            1
            for item in results
            if item.get(
                "citations_returned",
                False
            )
        )

        confidence_available = sum(
            1
            for item in results
            if item.get(
                "confidence_returned",
                False
            )
        )

        return {

            "total_questions": total,

            "successful_answers": successful,

            "grounded_answers": grounded,

            "cited_answers": cited,

            "confidence_available": (
                confidence_available
            ),

            "results": results,
        }
