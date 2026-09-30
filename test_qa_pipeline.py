"""Strict regression test for the two-paper Research Paper Intelligence QA system.

Unlike the older test, this file does not treat "claim exists" as a pass.
It validates whether returned claims have the semantic type requested by the
question and rejects known cross-task contamination.
"""

from pathlib import Path

from app.ingestion.ingest import ingest_paper
from app.indexing.chunker import create_chunks
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.task_retriever import TaskSpecificRetriever
from app.analysis.claim_grounder import ClaimGrounder
from app.analysis.claim_relevance_validator import ClaimRelevanceValidator
from app.analysis.qa_analyzer import QAAnalyzer
from app.analysis.task_semantics import normalize_task, is_task_valid


RAW_PAPERS_DIR = Path("data/raw_papers")

TEST_CASES = [
    {
        "question": "What methodology did the researchers use?",
        "task": "methodology",
        "comparison": False,
    },
    {
        "question": "What are the main findings of the papers?",
        "task": "results",
        "comparison": True,
    },
    {
        "question": "What limitations are reported in the papers?",
        "task": "limitations",
        "comparison": True,
    },
    {
        "question": "Compare the methodologies used in all the papers.",
        "task": "methodology",
        "comparison": True,
        "all_papers": True,
    },
    {
        "question": "Compare the limitations of all the papers.",
        "task": "limitations",
        "comparison": True,
        "all_papers": True,
    },
    {
        "question": "What samples or participants were used in each paper?",
        "task": "datasets",
        "comparison": True,
        "all_papers": True,
    },
    {
        "question": "What future research directions do the authors suggest?",
        "task": "future_work",
        "comparison": False,
    },
    {
        "question": "What are the main contributions of each paper?",
        "task": "contributions",
        "comparison": True,
        "all_papers": True,
    },
    {
        "question": "What factors or variables are examined in the papers?",
        "task": "variables",
        "comparison": True,
        "all_papers": True,
    },
    {
        "question": "What quantum computing algorithms are proposed in the papers?",
        "task": "models",
        "comparison": True,
        "unsupported": True,
    },
]


def value(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def claim_text(claim):
    return str(value(claim, "claim", value(claim, "text", "")) or "").strip()


def claim_section(claim):
    citations = value(claim, "citations", []) or []
    for c in citations:
        section = value(c, "section", "")
        if section:
            return str(section)
    return ""


def claim_paper_ids(claims, evidence):
    evidence_map = {
        str(e.get("chunk_id")): str(e.get("paper_id"))
        for e in evidence
        if isinstance(e, dict) and e.get("chunk_id") and e.get("paper_id")
    }
    ids = set()
    for claim in claims:
        for citation in value(claim, "citations", []) or []:
            pid = value(citation, "paper_id", "")
            if pid:
                ids.add(str(pid))
        for eid in value(claim, "evidence_ids", []) or []:
            pid = evidence_map.get(str(eid))
            if pid:
                ids.add(pid)
    return ids


def forbidden(task, text):
    low = text.lower()
    checks = {
        "limitations": [
            "future research", "future work", "future studies", "promising avenue",
            "can reposition", "major reason", "credit authorship", "pca was",
            "questionnaire items",
        ],
        "results": [
            "process macro model 4 was used", "pca was used", "data were collected",
            "future research", "fig. 3.", "vto technology has a fig",
        ],
        "datasets": [
            "likert scale", "measurement items", "questionnaire items", "adapted items",
            "factor loadings", "composite reliability", "average variance extracted",
            "pca was",
        ],
        "future_work": [
            "limitation of this study", "limitations of our study", "this study did not",
            "sample was limited",
        ],
        "contributions": [
            "credit authorship", "credit authorship statement", "pca was",
            "principal component analysis", "factor loadings", "composite reliability",
            "measurement model",
        ],
        "variables": [
            "measurement model", "reliability", "validity", "factor loadings",
            "composite reliability", "average variance extracted", "pca",
            "harman", "questionnaire items",
        ],
        "methodology": [
            "future research", "limitations", "results show", "findings show",
        ],
    }
    return [term for term in checks.get(task, []) if term in low]


def validate_result(case, result, expected_papers):
    task = normalize_task(case["task"])
    answer = str(result.get("answer", ""))
    claims = result.get("claims", []) or []
    evidence = result.get("evidence", []) or []

    failures = []

    if not answer.strip():
        failures.append("empty answer")

    if case.get("unsupported"):
        # This query should not be answered using unrelated VTO evidence.
        if claims:
            failures.append("unsupported-topic query returned grounded claims")
        return failures

    if not claims:
        failures.append("no grounded claims")
        return failures

    for index, claim in enumerate(claims, start=1):
        text = claim_text(claim)
        section = claim_section(claim)
        if not is_task_valid(task, text, section):
            failures.append(f"claim {index} failed task semantic validation")
        bad = forbidden(task, text)
        if bad:
            failures.append(f"claim {index} contains forbidden cross-task signal: {bad}")

    grounded_papers = claim_paper_ids(claims, evidence)

    if case.get("all_papers"):
        missing = expected_papers - grounded_papers
        if missing:
            failures.append(f"question missing grounded claims from papers: {sorted(missing)}")

    if case.get("comparison"):
        # The comparison answer should visibly partition claims by paper.
        for pid in sorted(expected_papers):
            title_or_id = pid.lower()
            if title_or_id not in answer.lower():
                # A paper title may be used instead of the ID. Evidence-derived
                # titles are checked here.
                candidate_titles = {
                    str(e.get("paper_title", "")).lower()
                    for e in evidence
                    if isinstance(e, dict) and str(e.get("paper_id", "")) == pid
                }
                if not any(t and t in answer.lower() for t in candidate_titles):
                    failures.append(f"comparison answer does not visibly identify {pid}")

    return failures


print("=" * 80)
print("STRICT END-TO-END RESEARCH PAPER QA REGRESSION")
print("=" * 80)

pdf_files = sorted(RAW_PAPERS_DIR.glob("*.pdf"))
if len(pdf_files) < 2:
    raise RuntimeError(f"Expected at least 2 PDFs in {RAW_PAPERS_DIR}, found {len(pdf_files)}")

print("PDFs:", [p.name for p in pdf_files])

all_chunks = []
for index, pdf_path in enumerate(pdf_files, start=1):
    paper_id = f"paper{index}"
    print(f"\nIngesting {paper_id}: {pdf_path.name}")
    paper = ingest_paper(pdf_path)
    chunks = create_chunks(paper)
    all_chunks.extend(chunks)
    print(f"  chunks: {len(chunks)}")

expected_papers = set()
for c in all_chunks:
    pid = c.paper_id if hasattr(c, "paper_id") else (c.get("paper_id") if isinstance(c, dict) else None)
    if pid:
        expected_papers.add(str(pid))

print("\nTotal chunks:", len(all_chunks))
print("Expected papers:", sorted(expected_papers))

bm25 = BM25Retriever(all_chunks)
vector = VectorRetriever(top_k=10)
hybrid = HybridRetriever(vector_retriever=vector, bm25_retriever=bm25)
task_retriever = TaskSpecificRetriever(
    hybrid_retriever=hybrid,
    candidate_k=20,
    top_k=8,
)

grounder = ClaimGrounder()
validator = ClaimRelevanceValidator()
qa = QAAnalyzer(
    task_retriever=task_retriever,
    claim_grounder=grounder,
    claim_relevance_validator=validator,
)

results = []

for number, case in enumerate(TEST_CASES, start=1):
    question = case["question"]
    print("\n" + "=" * 80)
    print(f"Q{number}: {question}")
    print("=" * 80)

    try:
        result = qa.answer(question)
        failures = validate_result(case, result, expected_papers)
        results.append({"question": question, "result": result, "failures": failures})

        print("Task:", result.get("task"))
        print("Claims:", len(result.get("claims", []) or []))
        print("Grounded papers:", sorted(claim_paper_ids(result.get("claims", []) or [], result.get("evidence", []) or [])))
        print("Confidence:", result.get("confidence", {}))
        print("Answer:\n", result.get("answer", ""))

        if failures:
            print("\n[FAIL]")
            for failure in failures:
                print(" -", failure)
        else:
            print("\n[PASS] Strict semantic validation")

    except Exception as exc:
        results.append({"question": question, "result": {}, "failures": [f"execution error: {exc}"]})
        print("\n[FAIL] Execution error:", repr(exc))

print("\n" + "=" * 80)
print("FINAL STRICT TEST SUMMARY")
print("=" * 80)

failed = 0
for index, item in enumerate(results, start=1):
    if item["failures"]:
        failed += 1
        print(f"[FAIL] Q{index}: {item['question']}")
        for failure in item["failures"]:
            print("       -", failure)
    else:
        print(f"[PASS] Q{index}: {item['question']}")

print("\nQuestions tested:", len(results))
print("Questions passed:", len(results) - failed)
print("Questions failed:", failed)

if failed:
    raise SystemExit(1)

print("\nSTRICT END-TO-END QA TEST PASSED")
