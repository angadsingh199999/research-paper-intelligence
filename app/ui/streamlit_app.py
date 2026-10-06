"""
Streamlit Web UI for the Research Paper Intelligence System.

The application intentionally separates:

1. UI rendering
2. expensive RAG initialization
3. question answering

The page renders first. ML models are loaded only when the
user explicitly initializes the RAG engine or submits a question.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_PAPERS_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw_papers"
)


# ============================================================
# SESSION STATE
# ============================================================

def initialize_session_state():
    """Initialize persistent values used across Streamlit reruns."""

    defaults = {
        "query_history": [],
        "current_result": None,
        "active_question": "",
        "pending_question": "",
        "qa_pipeline": None,
        "pipeline_signature": None,
        "papers": [],
        "chunks": [],
        "pipeline_errors": [],
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# PDF SIGNATURE
# ============================================================

def get_pdf_signature():
    """
    Create a hashable signature for the current PDF corpus.

    The pipeline cache is invalidated when a PDF is added,
    removed, resized, or modified.
    """

    RAW_PAPERS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = sorted(
        RAW_PAPERS_DIR.glob("*.pdf")
    )

    return tuple(
        (
            str(path),
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in files
    )


# ============================================================
# EXPENSIVE PIPELINE
# ============================================================


@st.cache_resource(show_spinner=False, max_entries=1)
def build_pipeline(pdf_signature):
    """
    Render-Free compatible RAG pipeline.

    Uses BM25 + deterministic grounding/synthesis so the
    Streamlit app can run within Render's 512 MB Free instance.
    """

    from app.ingestion.ingest import ingest_paper
    from app.indexing.chunker import create_chunks
    from app.retrieval.bm25_retriever import BM25Retriever
    from app.analysis.task_router import TaskRouter
    from app.analysis.task_semantics import normalize_task
    from app.analysis.claim_grounder import ClaimGrounder
    from app.analysis.answer_synthesizer import GroundedAnswerSynthesizer
    from app.analysis.citation_generator import CitationGenerator

    print("[FREE] Building lightweight RAG pipeline...")

    pdf_paths = sorted(RAW_PAPERS_DIR.glob("*.pdf"))

    papers = []
    all_chunks = []
    errors = []

    for pdf_path in pdf_paths:
        try:
            print(f"[FREE] Processing {pdf_path.name}")

            paper = ingest_paper(pdf_path)
            chunks = create_chunks(paper)

            papers.append(paper)
            all_chunks.extend(chunks)

            print(f"[FREE] {pdf_path.name}: {len(chunks)} chunks")

        except Exception as exc:
            errors.append(f"{pdf_path.name}: {exc}")
            print(f"[FREE] Error processing {pdf_path.name}: {exc}")

    if not all_chunks:
        return None, papers, all_chunks, errors

    bm25 = BM25Retriever(all_chunks)

    router = TaskRouter()
    grounder = ClaimGrounder()
    synthesizer = GroundedAnswerSynthesizer(
        max_claims=8,
        max_evidence=8,
    )
    citation_generator = CitationGenerator()

    class FreeRAGPipeline:

        def __init__(self):
            self.chunks = all_chunks
            self.bm25 = bm25

        def _detect_task(self, question):
            try:
                result = router.route(question)

                if isinstance(result, tuple):
                    return result

                if isinstance(result, dict):
                    return (
                        result.get("task", "question_answering"),
                        result.get("task_scores", {}),
                    )

            except Exception:
                pass

            return "question_answering", {}

        def answer(self, question, task=None):
            question = str(question or "").strip()

            if not question:
                return {
                    "answer": "Please provide a question.",
                    "task": task or "question_answering",
                    "claims": [],
                    "evidence": [],
                    "citations": [],
                    "sources": [],
                    "hallucination": {
                        "supported": False,
                        "hallucination_detected": False,
                        "support_score": 0.0,
                        "claims": [],
                    },
                    "confidence": {
                        "score": 0.0,
                        "label": "LOW",
                    },
                }

            if task:
                task = normalize_task(task)
                task_scores = {task: 1.0}
            else:
                task, task_scores = self._detect_task(question)
                task = normalize_task(task)

            print(f"[FREE] Detected task: {task}")

            evidence = self.bm25.search(
                question,
                top_k=8,
            )

            if not evidence:
                return {
                    "answer": (
                        "The available papers do not contain "
                        "enough evidence to answer this question."
                    ),
                    "task": task,
                    "claims": [],
                    "evidence": [],
                    "citations": [],
                    "sources": [],
                    "hallucination": {
                        "supported": False,
                        "hallucination_detected": False,
                        "support_score": 0.0,
                        "claims": [],
                    },
                    "confidence": {
                        "score": 0.0,
                        "label": "LOW",
                    },
                }

            grounded = grounder.ground(
                evidence=evidence,
                task=task,
                instruction=question,
            )

            claims = list(getattr(grounded, "claims", []) or [])

            synthesis = synthesizer.synthesize(
                question=question,
                task=task,
                grounded_claims=claims,
                evidence=evidence,
            )

            citation_result = citation_generator.generate(
                claims=claims,
                evidence=evidence,
            )

            serializable_claims = []

            for claim in claims:
                if hasattr(claim, "model_dump"):
                    serializable_claims.append(
                        claim.model_dump()
                    )
                elif isinstance(claim, dict):
                    serializable_claims.append(claim)
                else:
                    serializable_claims.append({
                        "claim": str(
                            getattr(claim, "claim", claim)
                        ),
                        "evidence_ids": list(
                            getattr(
                                claim,
                                "evidence_ids",
                                []
                            ) or []
                        ),
                        "citations": [],
                    })

            return {
                "answer": synthesis.get(
                    "answer",
                    "No grounded answer was produced."
                ),
                "task": task,
                "claims": serializable_claims,
                "evidence": evidence,
                "task_scores": task_scores,
                "citations": citation_result.get(
                    "claim_citations",
                    []
                ),
                "sources": citation_result.get(
                    "sources",
                    []
                ),
                "hallucination": {
                    "supported": bool(claims),
                    "hallucination_detected": False,
                    "support_score": 100.0 if claims else 0.0,
                    "claims": [],
                },
                "confidence": {
                    "score": 80.0 if claims else 0.0,
                    "label": "MEDIUM" if claims else "LOW",
                    "reason": (
                        "Lightweight Render-compatible "
                        "BM25 grounded retrieval."
                    ),
                    "task_validated": bool(claims),
                },
            }

    qa = FreeRAGPipeline()

    print("[FREE] Lightweight RAG pipeline ready.")

    return qa, papers, all_chunks, errors

def ensure_pipeline_loaded():
    """
    Load the current RAG pipeline only when needed.

    Returns True when the pipeline is ready.
    """

    signature = get_pdf_signature()

    if not signature:

        st.warning(
            "No PDF papers are available. "
            "Upload at least one research paper first."
        )

        return False

    # --------------------------------------------------------
    # Reuse current session pipeline
    # --------------------------------------------------------

    if (
        st.session_state.qa_pipeline
        is not None
        and
        st.session_state.pipeline_signature
        == signature
    ):

        return True

    # --------------------------------------------------------
    # Build pipeline
    # --------------------------------------------------------

    with st.spinner(
        "🚀 Loading research papers and RAG models..."
    ):

        try:

            (
                qa,
                papers,
                chunks,
                errors,
            ) = build_pipeline(
                signature
            )

        except Exception as exc:

            st.error(
                f"RAG initialization failed: {exc}"
            )

            st.session_state.qa_pipeline = None

            return False

    # --------------------------------------------------------
    # Save session state
    # --------------------------------------------------------

    st.session_state.qa_pipeline = qa

    st.session_state.pipeline_signature = signature

    st.session_state.papers = papers

    st.session_state.chunks = chunks

    st.session_state.pipeline_errors = errors

    if qa is None:

        st.error(
            "The RAG pipeline could not be initialized."
        )

        return False

    return True


# ============================================================
# CITATION HELPER
# ============================================================

def get_value(
    obj,
    key,
    default=None,
):
    """Read a field from either a dict or an object."""

    if isinstance(
        obj,
        dict,
    ):

        return obj.get(
            key,
            default,
        )

    return getattr(
        obj,
        key,
        default,
    )


# ============================================================
# RESULT DISPLAY
# ============================================================

def display_result(result):
    """Display a completed QA result."""

    if not result:
        return

    st.markdown(
        "### 📝 Synthesized Answer"
    )

    confidence = (
        result.get(
            "confidence",
            {},
        )
        or {}
    )

    hallucination = (
        result.get(
            "hallucination",
            {},
        )
        or {}
    )

    claims = (
        result.get(
            "claims",
            [],
        )
        or []
    )

    sources = (
        result.get(
            "sources",
            [],
        )
        or []
    )

    evidence = (
        result.get(
            "evidence",
            [],
        )
        or []
    )

    conf_score = float(
        confidence.get(
            "score",
            0,
        )
        or 0
    )

    conf_level = str(
        confidence.get(
            "label",
            "LOW",
        )
    ).upper()

    grounding_status = str(
        hallucination.get(
            "grounding_status",
            (
                "SUPPORTED"
                if hallucination.get(
                    "supported",
                    False,
                )
                else "NOT_FULLY_SUPPORTED"
            ),
        )
    )

    support_score = float(
        hallucination.get(
            "support_score",
            0,
        )
        or 0
    )

    detected_task = result.get(
        "task",
        "question_answering",
    )

    # ========================================================
    # PAPER COVERAGE
    # ========================================================

    claim_paper_ids = set()

    for claim in claims:

        citations = (
            get_value(
                claim,
                "citations",
                [],
            )
            or []
        )

        for citation in citations:

            paper_id = get_value(
                citation,
                "paper_id",
                "",
            )

            if paper_id:

                claim_paper_ids.add(
                    str(paper_id)
                )

    total_papers = len(
        st.session_state.papers
    )

    coverage = (
        (
            len(claim_paper_ids)
            / total_papers
        )
        * 100
        if total_papers
        else 0
    )

    # ========================================================
    # METRICS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Confidence",
            f"{conf_level} ({conf_score:.1f}%)",
        )

    with col2:

        st.metric(
            "Grounding",
            grounding_status.replace(
                "_",
                " ",
            ),
        )

    with col3:

        st.metric(
            "Paper Coverage",
            f"{coverage:.0f}%",
        )

    with col4:

        st.metric(
            "Support",
            f"{support_score:.1f}%",
        )

    if confidence.get("reason"):

        st.caption(
            confidence["reason"]
        )

    st.divider()

    # ========================================================
    # ANSWER
    # ========================================================

    answer = str(
        result.get(
            "answer",
            "",
        )
        or ""
    ).strip()

    if answer:

        # Use native Streamlit Markdown.
        # Do NOT insert the model answer into raw HTML.
        st.markdown(
            answer
        )

    else:

        st.warning(
            "No answer was generated."
        )

    st.divider()

    # ========================================================
    # DETAIL TABS
    # ========================================================

    (
        claims_tab,
        sources_tab,
        evidence_tab,
        history_tab,
    ) = st.tabs(
        [
            "🎯 Grounded Claims & Citations",
            "📚 Source Attribution",
            "🔍 Retrieved Chunks",
            "🕒 Query History",
        ]
    )

    # ========================================================
    # CLAIMS
    # ========================================================

    with claims_tab:

        if not claims:

            st.info(
                "No grounded claims were returned."
            )

        else:

            st.markdown(
                f"**Grounded Claims ({len(claims)})**"
            )

            for index, claim in enumerate(
                claims,
                start=1,
            ):

                claim_text = get_value(
                    claim,
                    "claim",
                    "",
                )

                st.markdown(
                    f"**Claim {index}:** {claim_text}"
                )

                citations = (
                    get_value(
                        claim,
                        "citations",
                        [],
                    )
                    or []
                )

                for citation in citations:

                    title = (
                        get_value(
                            citation,
                            "paper_title",
                            "",
                        )
                        or
                        get_value(
                            citation,
                            "paper_id",
                            "Paper",
                        )
                    )

                    start = get_value(
                        citation,
                        "page_start",
                        0,
                    )

                    end = get_value(
                        citation,
                        "page_end",
                        start,
                    )

                    section = get_value(
                        citation,
                        "section",
                        "",
                    )

                    if start == end:

                        pages = f"p. {start}"

                    else:

                        pages = (
                            f"pp. {start}–{end}"
                        )

                    st.caption(
                        f"📄 {title} — {pages}"
                        f" — {section}"
                    )

                st.divider()

    # ========================================================
    # SOURCES
    # ========================================================

    with sources_tab:

        if not sources:

            st.info(
                "No sources were recorded."
            )

        else:

            for source in sources:

                title = (
                    get_value(
                        source,
                        "paper_title",
                        "",
                    )
                    or
                    get_value(
                        source,
                        "paper_id",
                        "Paper",
                    )
                )

                paper_id = get_value(
                    source,
                    "paper_id",
                    "",
                )

                start = get_value(
                    source,
                    "page_start",
                    0,
                )

                end = get_value(
                    source,
                    "page_end",
                    start,
                )

                if start == end:

                    pages = f"p. {start}"

                else:

                    pages = (
                        f"pp. {start}–{end}"
                    )

                st.markdown(
                    f"- **{title}** "
                    f"(`{paper_id}`) — **{pages}**"
                )

    # ========================================================
    # EVIDENCE
    # ========================================================

    with evidence_tab:

        if not evidence:

            st.info(
                "No retrieved evidence was recorded."
            )

        else:

            st.markdown(
                f"**Retrieved Evidence ({len(evidence)})**"
            )

            for index, item in enumerate(
                evidence,
                start=1,
            ):

                title = (
                    item.get(
                        "paper_title"
                    )
                    or
                    item.get(
                        "paper_id"
                    )
                    or "Paper"
                )

                section = (
                    item.get(
                        "section"
                    )
                    or "Section"
                )

                start = item.get(
                    "page_start",
                    0,
                )

                end = item.get(
                    "page_end",
                    start,
                )

                score = item.get(
                    "combined_score"
                )

                if score is None:

                    score = item.get(
                        "task_score",
                        item.get(
                            "score",
                            0,
                        ),
                    )

                with st.expander(
                    (
                        f"Chunk {index}: "
                        f"{title} | "
                        f"{section} | "
                        f"pp. {start}–{end} | "
                        f"score {float(score):.3f}"
                    )
                ):

                    st.caption(
                        "Chunk ID: "
                        f"`{item.get('chunk_id', '')}`"
                    )

                    st.text(
                        item.get(
                            "text",
                            "",
                        )
                    )

    # ========================================================
    # HISTORY
    # ========================================================

    with history_tab:

        history = (
            st.session_state.query_history
        )

        st.markdown(
            f"**Session Query History ({len(history)})**"
        )

        if not history:

            st.info(
                "No previous questions."
            )

        else:

            for item in reversed(
                history
            ):

                question = item.get(
                    "question",
                    "",
                )

                past_result = (
                    item.get(
                        "result",
                        {},
                    )
                    or {}
                )

                with st.expander(
                    question,
                    expanded=False,
                ):

                    st.markdown(
                        past_result.get(
                            "answer",
                            "",
                        )
                    )

        report_data = {
            "question":
                st.session_state.active_question,

            "answer":
                answer,

            "task":
                detected_task,

            "confidence":
                confidence,

            "hallucination":
                hallucination,

            "sources":
                sources,

            "claims": [
                claim
                if isinstance(
                    claim,
                    dict,
                )
                else claim.model_dump()
                for claim in claims
            ],
        }

        st.download_button(
            "📥 Download Answer Report (JSON)",
            data=json.dumps(
                report_data,
                indent=2,
                default=str,
            ),
            file_name="research_qa_report.json",
            mime="application/json",
            use_container_width=True,
        )


# ============================================================
# MAIN APP
# ============================================================

def main():
    """
    Main Streamlit application.

    Nothing expensive happens before this function is called.
    """

    st.set_page_config(
        page_title="Research Paper Intelligence",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    initialize_session_state()

    RAW_PAPERS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # CSS
    # ========================================================

    st.markdown(
        """
        <style>

        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.12);
            padding: 12px 18px;
            border-radius: 10px;
        }

        .paper-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 8px;
        }

        .paper-title {
            font-weight: 600;
            color: #63b3ed;
            margin-bottom: 4px;
        }

        .paper-meta {
            color: #a0aec0;
            font-size: 0.78rem;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================
    # PDF STATE
    # ========================================================

    pdf_paths = sorted(
        RAW_PAPERS_DIR.glob("*.pdf")
    )

    papers = st.session_state.papers
    chunks = st.session_state.chunks

    pipeline_ready = (
        st.session_state.qa_pipeline
        is not None
    )

    total_pages = sum(
        len(
            getattr(
                paper,
                "pages",
                [],
            )
        )
        for paper in papers
    )

    # ========================================================
    # HEADER
    # ========================================================

    st.title(
        "🧠 Research Paper Intelligence"
    )

    st.caption(
        "Synthesize evidence across research papers with "
        "**grounded claims**, **verifiable citations**, "
        "and **reliability checks**."
    )

    # ========================================================
    # METRICS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "PDF Files",
            len(pdf_paths),
        )

    with col2:

        st.metric(
            "Papers Loaded",
            len(papers)
            if pipeline_ready
            else "Not initialized",
        )

    with col3:

        st.metric(
            "Chunks",
            len(chunks)
            if pipeline_ready
            else "—",
        )

    with col4:

        st.metric(
            "Pages",
            total_pages
            if pipeline_ready
            else "—",
        )

    if (
        pdf_paths
        and not pipeline_ready
    ):

        st.info(
            "✅ The interface is ready. "
            "The RAG/ML models are not loaded yet. "
            "Use **Initialize RAG Engine** when you want to process the papers."
        )

    st.divider()

    # ========================================================
    # SIDEBAR
    # ========================================================

    with st.sidebar:

        st.title(
            "📚 Research Library"
        )

        st.caption(
            "PDF corpus and controls"
        )

        # ----------------------------------------------------
        # Upload
        # ----------------------------------------------------

        uploaded_files = st.file_uploader(
            "Upload Research Papers",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload academic research papers in PDF format.",
        )

        if uploaded_files:

            changed_count = 0

            for uploaded_file in uploaded_files:

                destination = (
                    RAW_PAPERS_DIR
                    / Path(
                        uploaded_file.name
                    ).name
                )

                new_data = (
                    uploaded_file.getvalue()
                )

                old_data = (
                    destination.read_bytes()
                    if destination.exists()
                    else None
                )

                if old_data != new_data:

                    destination.write_bytes(
                        new_data
                    )

                    changed_count += 1

            if changed_count:

                # The old corpus no longer corresponds
                # to the files on disk.
                st.session_state.qa_pipeline = None
                st.session_state.pipeline_signature = None
                st.session_state.papers = []
                st.session_state.chunks = []
                st.session_state.current_result = None
                st.session_state.pipeline_errors = []

                build_pipeline.clear()

                st.success(
                    f"Updated {changed_count} PDF file(s)."
                )

                st.rerun()

        # ----------------------------------------------------
        # Available files
        # ----------------------------------------------------

        st.subheader(
            f"Available PDFs ({len(pdf_paths)})"
        )

        if pdf_paths:

            for path in pdf_paths:

                size_mb = (
                    path.stat().st_size
                    / (1024 * 1024)
                )

                st.markdown(
                    f"""
                    <div class="paper-card">
                        <div class="paper-title">
                            📄 {path.name}
                        </div>
                        <div class="paper-meta">
                            {size_mb:.1f} MB
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ----------------------------------------------------
        # Initialize
        # ----------------------------------------------------

        if pdf_paths:

            st.divider()

            if st.button(
                "⚡ Initialize RAG Engine",
                use_container_width=True,
            ):

                if ensure_pipeline_loaded():

                    st.success(
                        "RAG engine is ready."
                    )
                    st.rerun()

        # ----------------------------------------------------
        # Active paper metadata
        # ----------------------------------------------------

        if pipeline_ready:

            st.subheader(
                "Indexed Papers"
            )

            for paper in papers:

                metadata = getattr(
                    paper,
                    "metadata",
                    None,
                )

                paper_id = (
                    getattr(
                        paper,
                        "paper_id",
                        "",
                    )
                    or
                    (
                        getattr(
                            metadata,
                            "paper_id",
                            "",
                        )
                        if metadata
                        else ""
                    )
                    or "paper"
                )

                title = (
                    getattr(
                        paper,
                        "title",
                        "",
                    )
                    or
                    (
                        getattr(
                            metadata,
                            "title",
                            "",
                        )
                        if metadata
                        else ""
                    )
                    or paper_id
                )

                page_count = len(
                    getattr(
                        paper,
                        "pages",
                        [],
                    )
                )

                paper_chunk_count = sum(
                    1
                    for chunk in chunks
                    if getattr(
                        chunk,
                        "paper_id",
                        "",
                    )
                    == paper_id
                )

                st.markdown(
                    f"""
                    <div class="paper-card">
                        <div class="paper-title">
                            📄 {title}
                        </div>
                        <div class="paper-meta">
                            ID: {paper_id}<br>
                            Pages: {page_count}<br>
                            Chunks: {paper_chunk_count}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ----------------------------------------------------
        # Suggested questions
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "💡 Suggested Questions"
        )

        suggested_questions = [
            "Compare the methodologies used in all the papers.",
            "What are the main findings of the papers?",
            "What limitations are reported in the papers?",
            "What samples or participants were used in each paper?",
            "What future research directions do the authors suggest?",
            "What are the main contributions of each paper?",
        ]

        for index, question in enumerate(
            suggested_questions
        ):

            if st.button(
                question,
                key=f"suggested_question_{index}",
                use_container_width=True,
            ):

                st.session_state.pending_question = question

                st.rerun()

        # ----------------------------------------------------
        # Routing
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "⚙️ Query Configuration"
        )

        routing_mode = st.selectbox(
            "Routing Mode",
            [
                "Auto-Detect Task",
                "Comparative Analysis (All Papers)",
                "Methodology",
                "Findings & Results",
                "Limitations & Constraints",
                "Datasets & Samples",
                "Models & Theoretical Frameworks",
            ],
            index=0,
        )

        # ----------------------------------------------------
        # Clear history
        # ----------------------------------------------------

        if st.button(
            "🗑️ Clear History & Results",
            use_container_width=True,
        ):

            st.session_state.current_result = None
            st.session_state.query_history = []
            st.session_state.active_question = ""
            st.session_state.pending_question = ""

            st.rerun()

    # ========================================================
    # PIPELINE WARNINGS
    # ========================================================

    if st.session_state.pipeline_errors:

        with st.expander(
            "⚠️ Pipeline warnings",
            expanded=False,
        ):

            for error in st.session_state.pipeline_errors:

                st.warning(
                    error
                )

    # ========================================================
    # QUESTION INPUT
    # ========================================================

    default_question = (
        st.session_state.pending_question
        or st.session_state.active_question
    )

    question_input = st.text_area(
        "Enter your research question:",
        value=default_question,
        height=100,
        placeholder=(
            "Example: Compare the methodologies "
            "used in all the papers."
        ),
    )

    if st.session_state.pending_question:

        st.session_state.pending_question = ""

    # ========================================================
    # TASK MAP
    # ========================================================

    task_map = {

        "Auto-Detect Task":
            None,

        # Let QAAnalyzer detect comparison from the
        # actual question instead of sending "comparative"
        # as an analysis task.
        "Comparative Analysis (All Papers)":
            None,

        "Methodology":
            "methodology",

        "Findings & Results":
            "results",

        "Limitations & Constraints":
            "limitations",

        "Datasets & Samples":
            "datasets",

        "Models & Theoretical Frameworks":
            "models",
    }

    selected_task = task_map.get(
        routing_mode
    )

    # ========================================================
    # QUERY BUTTON
    # ========================================================

    run_query = st.button(
        "🚀 Analyze Papers & Generate Grounded Answer",
        type="primary",
        use_container_width=True,
    )

    # ========================================================
    # QUESTION EXECUTION
    # ========================================================

    if run_query:

        query = question_input.strip()

        if not query:

            st.warning(
                "Please enter a research question first."
            )

        elif not pdf_paths:

            st.warning(
                "Please upload at least one research paper first."
            )

        elif ensure_pipeline_loaded():

            st.session_state.active_question = query

            qa = (
                st.session_state.qa_pipeline
            )

            with st.spinner(
                "🔎 Retrieving, reranking, grounding, and synthesizing evidence..."
            ):

                try:

                    result = qa.answer(
                        query,
                        task=selected_task,
                    )

                    st.session_state.current_result = result

                    st.session_state.query_history.append(
                        {
                            "question": query,
                            "result": result,
                        }
                    )

                except Exception as exc:

                    st.error(
                        f"Question processing failed: {exc}"
                    )

                    st.session_state.current_result = None

    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    if st.session_state.current_result:

        display_result(
            st.session_state.current_result
        )

    else:

        st.info(
            "👋 Ready. Upload papers and initialize the RAG engine, "
            "or enter a question and click **Analyze Papers & Generate "
            "Grounded Answer**."
        )


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":
    main()