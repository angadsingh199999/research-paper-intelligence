"""Streamlit UI for Research Paper Intelligence.

The module is import-safe: it defines the UI but does not build the RAG
pipeline during import. This prevents model loading during Streamlit startup.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_PAPERS_DIR = ROOT_DIR / "data" / "raw_papers"


st.set_page_config(
    page_title="Research Paper Intelligence",
    page_icon="📄",
    layout="wide",
)


def init_state():
    defaults = {
        "rpi_qa": None,
        "rpi_papers": [],
        "rpi_chunks": [],
        "rpi_signature": None,
        "rpi_result": None,
        "rpi_question": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def pdf_signature():
    RAW_PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    return tuple(
        (
            str(path),
            path.stat().st_mtime_ns,
            path.stat().st_size,
        )
        for path in sorted(RAW_PAPERS_DIR.glob("*.pdf"))
    )


@st.cache_resource(show_spinner=False, max_entries=1)
def build_pipeline(signature):
    """Build the complete RAG pipeline only when explicitly needed."""
    from app.embeddings.embedder import EmbeddingModel
    from app.vectorstore.chroma_store import ChromaVectorStore
    from app.ingestion.ingest import ingest_paper
    from app.indexing.chunker import create_chunks
    from app.retrieval.bm25_retriever import BM25Retriever
    from app.retrieval.vector_retriever import VectorRetriever
    from app.retrieval.hybrid_retriever import HybridRetriever
    from app.retrieval.task_retriever import TaskSpecificRetriever
    from app.analysis.claim_grounder import ClaimGrounder
    from app.analysis.claim_relevance_validator import ClaimRelevanceValidator
    from app.analysis.qa_analyzer import QAAnalyzer

    pdf_paths = sorted(RAW_PAPERS_DIR.glob("*.pdf"))
    papers = []
    chunks = []
    errors = []

    for path in pdf_paths:
        try:
            paper = ingest_paper(str(path))
            paper_chunks = create_chunks(paper)
            if paper_chunks:
                papers.append(paper)
                chunks.extend(paper_chunks)
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    if not chunks:
        return None, papers, chunks, errors

    try:
        chroma = ChromaVectorStore()
        existing_ids = set(chroma.get_existing_ids())
        missing = [x for x in chunks if x.chunk_id not in existing_ids]
        if missing:
            embedder = EmbeddingModel()
            embeddings = embedder.encode([x.text for x in missing])
            chroma.add_chunks(missing, embeddings)
    except Exception as exc:
        errors.append(f"Vector index warning: {exc}")

    bm25 = BM25Retriever(chunks)
    vector = VectorRetriever(top_k=10)
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
    return qa, papers, chunks, errors


def reset_pipeline():
    st.session_state.rpi_qa = None
    st.session_state.rpi_papers = []
    st.session_state.rpi_chunks = []
    st.session_state.rpi_signature = None
    st.session_state.rpi_result = None
    build_pipeline.clear()


def load_pipeline(show_errors=True):
    signature = pdf_signature()
    if not signature:
        return False

    if (
        st.session_state.rpi_qa is not None
        and st.session_state.rpi_signature == signature
    ):
        return True

    with st.spinner("Loading papers and building the RAG pipeline..."):
        qa, papers, chunks, errors = build_pipeline(signature)

    if qa is None:
        if show_errors:
            st.error("No usable chunks could be created from the PDFs.")
        return False

    st.session_state.rpi_qa = qa
    st.session_state.rpi_papers = papers
    st.session_state.rpi_chunks = chunks
    st.session_state.rpi_signature = signature

    if show_errors:
        for error in errors:
            st.warning(error)

    return True


def paper_title(paper):
    metadata = getattr(paper, "metadata", None)
    return str(
        getattr(metadata, "title", "")
        or getattr(metadata, "paper_id", "Paper")
        or "Paper"
    ).strip()


def render_result(result):
    answer = str(result.get("answer", "") or "").strip()
    st.subheader("Answer")
    st.markdown(answer)

    confidence = result.get("confidence", {}) or {}
    st.caption(
        f"Task: {result.get('task', 'N/A')} · "
        f"Claims: {len(result.get('claims', []) or [])} · "
        f"Evidence: {len(result.get('evidence', []) or [])} · "
        f"Confidence: {confidence.get('score', 0)} {confidence.get('label', 'LOW')}"
    )

    with st.expander("Sources"):
        sources = result.get("sources", []) or []
        if sources:
            st.json(sources)
        else:
            st.info("No source records were returned.")

    with st.expander("Grounded claims"):
        claims = result.get("claims", []) or []
        for index, claim in enumerate(claims, start=1):
            text = (
                claim.get("claim")
                if isinstance(claim, dict)
                else getattr(claim, "claim", "")
            )
            st.write(f"{index}. {text}")


def main():
    init_state()
    RAW_PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    st.title("📄 Research Paper Intelligence")
    st.caption(
        "Grounded multi-paper question answering with task-aware retrieval, "
        "citations and support checks."
    )

    uploaded = st.file_uploader(
        "Upload research papers",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded:
        added = 0
        for file in uploaded:
            destination = RAW_PAPERS_DIR / Path(file.name).name
            data = file.getbuffer()
            if not destination.exists() or destination.read_bytes() != data:
                destination.write_bytes(data)
                added += 1
        if added:
            reset_pipeline()
            st.success(f"Added {added} paper(s).")
            st.rerun()

    pdfs = sorted(RAW_PAPERS_DIR.glob("*.pdf"))

    with st.sidebar:
        st.header("Research Library")
        st.write(f"PDFs: {len(pdfs)}")
        for pdf in pdfs:
            st.write(f"• {pdf.name}")

        if st.button("Initialize / Rebuild RAG", use_container_width=True):
            reset_pipeline()
            load_pipeline()

    if not pdfs:
        st.info("Add PDFs to data/raw_papers/ or upload them above.")
        return

    if not load_pipeline(show_errors=False):
        st.warning("Click **Initialize / Rebuild RAG** to load the paper corpus.")
        return

    papers = st.session_state.rpi_papers
    chunks = st.session_state.rpi_chunks

    c1, c2, c3 = st.columns(3)
    c1.metric("Papers", len(papers))
    c2.metric("Chunks", len(chunks))
    c3.metric("Retrieval", "BM25 + Vector + Reranker")

    st.divider()

    st.subheader("Ask the papers")

    with st.form("research_question_form"):
        question = st.text_area(
            "Research question",
            placeholder=(
                "Example: Compare the limitations of both papers."
            ),
            height=120,
        )
        submitted = st.form_submit_button(
            "🧠 Analyze Papers",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        question = question.strip()
        if not question:
            st.warning("Please enter a question.")
        else:
            # Clear the previous result before fetching a new answer so that
            # stale results never bleed through when the new answer is empty.
            st.session_state.rpi_result = None
            st.session_state.rpi_question = question
            with st.spinner("Retrieving evidence and generating a grounded answer..."):
                st.session_state.rpi_result = st.session_state.rpi_qa.answer(question)

    if st.session_state.rpi_result:
        st.divider()
        render_result(st.session_state.rpi_result)
    elif st.session_state.rpi_question:
        # A question was asked but no grounded result could be generated.
        st.divider()
        st.info(
            "The papers do not contain sufficient evidence to answer that question. "
            "Try rephrasing or ask something directly related to the uploaded papers."
        )
    else:
        st.info(
            "The corpus is loaded. Ask a question above to generate an answer."
        )


if __name__ == "__main__":
    main()
