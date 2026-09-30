"""
Quick real-world QA smoke test.

Run from the project root:
    python test_realtime_questions.py
"""

from pathlib import Path

from app.ingestion.ingest import ingest_paper
from app.indexing.chunker import create_chunks
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.task_retriever import TaskSpecificRetriever
from app.analysis.qa_analyzer import QAAnalyzer


RAW_DIR = Path("data/raw_papers")
pdf_paths = sorted(RAW_DIR.glob("*.pdf"))

if not pdf_paths:
    raise RuntimeError("No PDFs found in data/raw_papers")

all_chunks = []
for path in pdf_paths:
    paper = ingest_paper(str(path))
    all_chunks.extend(create_chunks(paper))

bm25 = BM25Retriever(all_chunks)
vector = VectorRetriever(top_k=12)
hybrid = HybridRetriever(bm25, vector)
task_retriever = TaskSpecificRetriever(
    hybrid,
    candidate_k=30,
    top_k=12,
)
qa = QAAnalyzer(task_retriever)

questions = [
    "What limitations are reported in both papers?",
    "Compare the methodologies used in all papers.",
    "What are the main findings of each paper?",
    "What sample or participants were used in each paper?",
    "What future research directions do the authors suggest?",
    "How does virtual try-on technology affect consumer behaviour according to the papers?",
]

for question in questions:
    print("\n" + "#" * 100)
    print("QUESTION:", question)
    print("#" * 100)

    result = qa.answer(question)

    print("\nANSWER:\n")
    print(result.get("answer", ""))

    print("\nTASK:", result.get("task"))
    print("CLAIMS:", len(result.get("claims", [])))
    print("EVIDENCE:", len(result.get("evidence", [])))
    print("SOURCES:", len(result.get("sources", [])))
    print("CONFIDENCE:", result.get("confidence", {}))

