"""Interactive terminal interface for Research Paper Intelligence."""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

import app.config as config

from app.ingestion.ingest import ingest_paper
from app.indexing.chunker import create_chunks
from app.embeddings.embedder import EmbeddingModel
from app.vectorstore.chroma_store import ChromaVectorStore
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.task_retriever import TaskSpecificRetriever
from app.analysis.claim_grounder import ClaimGrounder
from app.analysis.claim_relevance_validator import ClaimRelevanceValidator
from app.analysis.qa_analyzer import QAAnalyzer


RAW_PAPERS_DIR = config.RAW_PAPERS_DIR


def load_corpus():
    RAW_PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    pdf_paths = sorted(RAW_PAPERS_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise RuntimeError(f"No PDF files found in {RAW_PAPERS_DIR}")

    print("\n" + "=" * 80)
    print("RESEARCH PAPER INTELLIGENCE")
    print("=" * 80)
    print(f"\nPDF directory: {RAW_PAPERS_DIR}")
    print(f"PDF files detected: {len(pdf_paths)}")
    for path in pdf_paths:
        print(f"  - {path.name}")

    papers = []
    chunks = []
    for path in pdf_paths:
        print(f"\nProcessing: {path.name}")
        paper = ingest_paper(str(path))
        paper_chunks = create_chunks(paper)
        papers.append(paper)
        chunks.extend(paper_chunks)
        meta = getattr(paper, "metadata", None)
        paper_id = getattr(meta, "paper_id", path.stem)
        title = str(getattr(meta, "title", "") or path.stem.replace("_", " ")).strip()
        print(f"Paper ID : {paper_id}")
        print(f"Title    : {title}")
        print(f"Chunks   : {len(paper_chunks)}")

    if not chunks:
        raise RuntimeError("No document chunks were created.")

    print(f"\nTotal papers : {len(papers)}")
    print(f"Total chunks : {len(chunks)}")
    return papers, chunks


def sync_chroma(chunks):
    """Add any missing chunks to ChromaDB."""
    try:
        chroma = ChromaVectorStore()
        existing = set(chroma.get_existing_ids())
        missing = [x for x in chunks if x.chunk_id not in existing]
        if not missing:
            print("Vector index already contains all chunks.")
            return
        print(f"Synchronizing {len(missing)} new chunks into ChromaDB...")
        embedder = EmbeddingModel()
        embeddings = embedder.encode([x.text for x in missing])
        chroma.add_chunks(missing, embeddings)
        print("Vector synchronization complete.")
    except Exception as exc:
        print(f"Vector index synchronization warning: {exc}")


def build_pipeline(chunks):
    print("\nBuilding retrieval and QA pipeline...")
    bm25 = BM25Retriever(chunks)
    vector = VectorRetriever(top_k=12)
    hybrid = HybridRetriever(
        vector_retriever=vector,
        bm25_retriever=bm25,
    )
    task_retriever = TaskSpecificRetriever(
        hybrid,
        candidate_k=36,
        top_k=12,
    )
    qa = QAAnalyzer(
        task_retriever=task_retriever,
        claim_grounder=ClaimGrounder(),
        claim_relevance_validator=ClaimRelevanceValidator(),
    )
    print("✓ QA pipeline ready.")
    return qa


def print_result(result):
    print("\n" + "=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)
    print(str(result.get("answer", "") or "").strip())

    confidence = result.get("confidence", {}) or {}
    print("\n" + "-" * 80)
    print(f"Detected task : {result.get('task', 'N/A')}")
    print(f"Claims        : {len(result.get('claims', []) or [])}")
    print(f"Evidence      : {len(result.get('evidence', []) or [])}")
    print(f"Sources       : {len(result.get('sources', []) or [])}")
    print(f"Confidence    : {confidence.get('score', 0)} {confidence.get('label', 'LOW')}")
    print("-" * 80)


def ask(qa, question):
    question = str(question or "").strip()
    if not question:
        return
    # QAAnalyzer has useful internal debug prints. Hide them in interactive
    # mode so the final answer remains readable.
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            result = qa.answer(question)
    except Exception as exc:
        print(f"\nERROR: {exc}")
        return
    print_result(result)


def interactive(qa):
    print("\n" + "=" * 80)
    print("INTERACTIVE RESEARCH PAPER Q&A")
    print("=" * 80)
    print("Ask any question about the loaded papers.")
    print("Type 'exit', 'quit', or 'q' to close.")

    while True:
        try:
            question = input("\nAsk your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            return

        if question.lower() in {"exit", "quit", "q"}:
            print("\nExiting.")
            return
        if not question:
            print("Please enter a question.")
            continue
        ask(qa, question)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*", help="Optional one-shot question")
    args = parser.parse_args()

    _, chunks = load_corpus()
    sync_chroma(chunks)
    qa = build_pipeline(chunks)

    if args.question:
        ask(qa, " ".join(args.question))
    else:
        interactive(qa)


if __name__ == "__main__":
    main()
