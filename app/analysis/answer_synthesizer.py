"""Deterministic synthesis of already validated grounded claims."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from app.analysis.task_semantics import is_noise, is_task_valid, normalize_task


class GroundedAnswerSynthesizer:
    """Format grounded claims without generating new factual content."""

    def __init__(self, max_claims=12, max_evidence=8):
        self.max_claims = int(max_claims)
        self.max_evidence = int(max_evidence)

    def synthesize(self, question: str, task: str, grounded_claims=None, evidence=None, available_papers=None) -> Dict[str, Any]:
        question = str(question or "").strip()
        task = normalize_task(task)
        grounded_claims = grounded_claims or []
        evidence = evidence or []

        claims = self._prepare_claims(grounded_claims, task)
        if not claims:
            return {
                "answer": "The available grounded evidence is insufficient to answer this question reliably.",
                "task": task,
                "claim_count": 0,
                "evidence_count": min(len(evidence), self.max_evidence),
            }

        comparison = self._is_comparison_question(question)
        if comparison:
            answer = self._comparison_answer(task, claims)
        else:
            answer = self._single_answer(task, claims)

        return {
            "answer": answer.strip(),
            "task": task,
            "claim_count": len(claims),
            "evidence_count": min(len(evidence), self.max_evidence),
        }

    def _prepare_claims(self, grounded_claims, task):
        output = []
        seen = set()
        for item in grounded_claims:
            text = self._claim_text(item)
            section = self._claim_section(item)
            if not text or is_noise(text):
                continue
            if task != "question_answering" and not is_task_valid(task, text, section):
                continue
            key = re.sub(r"\s+", " ", text.lower()).strip().rstrip(".!?")
            if key in seen:
                continue
            seen.add(key)
            output.append(item)
            if len(output) >= self.max_claims:
                break
        return output

    @staticmethod
    def _claim_text(item):
        if isinstance(item, dict):
            return " ".join(str(item.get("claim") or item.get("text") or "").split()).strip()
        return " ".join(
            str(getattr(item, "claim", None) or getattr(item, "text", None) or "").split()
        ).strip()

    @staticmethod
    def _claim_section(item):
        citations = item.get("citations", []) if isinstance(item, dict) else getattr(item, "citations", [])
        for citation in citations or []:
            if isinstance(citation, dict):
                value = citation.get("section", "")
            else:
                value = getattr(citation, "section", "")
            if value:
                return str(value)
        return ""

    @staticmethod
    def _claim_paper(item):
        citations = item.get("citations", []) if isinstance(item, dict) else getattr(item, "citations", [])
        for citation in citations or []:
            if isinstance(citation, dict):
                pid = citation.get("paper_id", "")
            else:
                pid = getattr(citation, "paper_id", "")
            if pid:
                return str(pid)
        return ""

    @staticmethod
    def _claim_title(item):
        citations = item.get("citations", []) if isinstance(item, dict) else getattr(item, "citations", [])
        for citation in citations or []:
            if isinstance(citation, dict):
                title = citation.get("paper_title", "")
            else:
                title = getattr(citation, "paper_title", "")
            if title and title.lower() not in {"paper1", "paper2"}:
                return str(title)
        return ""

    @staticmethod
    def _is_comparison_question(question):
        q = str(question or "").lower()
        return any(
            phrase in q
            for phrase in (
                "compare", "comparison", "differences between", "difference between",
                "similarities between", "similarities and differences", "across papers",
                "between papers", "between studies", "each paper", "both papers",
                "all papers", "all the papers", "the papers", "in the papers",
                "of the papers", "for each paper", "in each paper",
            )
        )

    def _single_answer(self, task, claims):
        opening = self._opening(task)
        bullets = []
        for item in claims:
            text = self._claim_text(item)
            if not text.endswith((".", "!", "?")):
                text += "."
            bullets.append(f"- {text}")
        return opening + "\n" + "\n".join(bullets)

    def _comparison_answer(self, task, claims):
        opening = self._opening(task, comparison=True)
        grouped = {}
        titles = {}
        for item in claims:
            pid = self._claim_paper(item) or "unknown"
            grouped.setdefault(pid, []).append(item)
            title = self._claim_title(item)
            if title:
                titles[pid] = title

        sections = [opening]
        for pid in sorted(grouped):
            label = titles.get(pid) or pid
            sections.append(f"### {label}")
            for item in grouped[pid][:6]:
                text = self._claim_text(item)
                if not text.endswith((".", "!", "?")):
                    text += "."
                sections.append(f"- {text}")
        return "\n".join(sections)

    @staticmethod
    def _opening(task, comparison=False):
        if task == "methodology":
            return "The reported methodology is:"
        if task == "datasets":
            return "The reported sample and participant information is:"
        if task == "results":
            return "The reported findings are:"
        if task == "limitations":
            return "The reported limitations are:"
        if task == "future_work":
            return "The reported future research directions are:"
        if task == "contributions":
            return "The reported research contributions are:"
        if task == "variables":
            return "The reported variables/constructs are:"
        if task == "statistical_analysis":
            return "The reported statistical analyses are:"
        if task == "theoretical_framework":
            return "The reported theoretical framework is:"
        if task == "literature_review":
            return "The reported literature/background points are:"
        return "The following grounded information was retrieved:"
