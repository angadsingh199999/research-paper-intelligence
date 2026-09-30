# Research Paper Intelligence

A multi-paper **Retrieval-Augmented Generation (RAG)** system for asking grounded questions across research papers.

The project combines PDF ingestion, document chunking, hybrid retrieval, task-aware retrieval, transformer-based reranking, claim grounding, semantic task validation, citation generation, hallucination/support checking, and confidence scoring.

## Overview

Research papers contain information spread across sections such as methodology, participants, results, limitations, future research, contributions, and variables. A basic semantic search system can retrieve relevant text but may still return a sentence that is supported by the paper while answering the wrong type of question.

This project therefore separates:

1. **Retrieval** – find relevant paper chunks.
2. **Grounding** – convert retrieved evidence into evidence-linked claims.
3. **Task validation** – verify that each claim actually matches the requested research task.
4. **Synthesis** – build a focused answer only from validated claims.
5. **Verification** – check support, citations, and confidence.

## RAG Pipeline

```text
Research Paper PDFs
        ↓
PDF Ingestion
        ↓
Text Cleaning + Section Detection
        ↓
Document Chunking
        ↓
Embeddings
        ↓
Chroma / Vector Store
        ↓
BM25 Retrieval + Vector Retrieval
        ↓
Hybrid Retrieval
        ↓
Task-Aware Retrieval
        ↓
Cross-Encoder Reranking
        ↓
Sentence-Level Claim Grounding
        ↓
Task Semantic Validation
        ↓
Claim Relevance Validation
        ↓
Grounded Answer Synthesis
        ↓
Hallucination / Support Check
        ↓
Citation Generation
        ↓
Confidence / Reliability Score
        ↓
Terminal QA + Streamlit UI
```

## Key Features

- Multi-paper question answering
- Semantic vector retrieval
- BM25 lexical retrieval
- Hybrid retrieval using both lexical and semantic signals
- Task-aware retrieval for research-oriented questions
- Cross-encoder reranking
- Sentence-level grounded claims
- Strict semantic validation for research tasks
- Per-paper comparison and coverage handling
- Citation and source attribution
- Hallucination/support detection
- Confidence/reliability scoring
- Interactive Streamlit interface
- Regression testing with multiple research questions

## Supported Research Tasks

The QA layer is designed to handle questions about:

- Methodology
- Samples / participants / datasets
- Models and theoretical frameworks
- Main results / findings
- Limitations
- Future research
- Contributions
- Statistical analysis
- Variables / constructs
- Theoretical frameworks
- Literature review
- General research-paper questions

## Technologies Used

| Component | Technology |
|---|---|
| Language | Python |
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Vector Database | ChromaDB |
| Lexical Retrieval | BM25 |
| Semantic Retrieval | Sentence Transformers |
| UI | Streamlit |
| Data Source | Research-paper PDFs |

## Project Structure

```text
research_paper_intelligence/
│
├── app/
│   ├── analysis/
│   │   ├── answer_synthesizer.py
│   │   ├── claim_grounder.py
│   │   ├── claim_relevance_validator.py
│   │   ├── claim_relevance_validator.py
│   │   ├── confidence_scorer.py
│   │   ├── citation_generator.py
│   │   ├── hallucination_detector.py
│   │   ├── qa_analyzer.py
│   │   ├── task_router.py
│   │   └── task_semantics.py
│   │
│   ├── embeddings/
│   ├── ingestion/
│   ├── indexing/
│   ├── models/
│   ├── retrieval/
│   ├── ui/
│   └── vectorstore/
│
├── data/
│   └── raw_papers/
│
├── test_qa_pipeline.py
├── streamlit_app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/angadsingh199999/research-paper-intelligence.git
cd research-paper-intelligence
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Adding Research Papers

Place the research-paper PDFs inside:

```text
data/raw_papers/
```

The current project is designed around a small multi-paper research corpus. The papers used during development cover virtual try-on, luxury branding, sustainable fashion retail, consumer decision accuracy, return waste, and privacy concerns.

## Run the QA Pipeline

Run the regression test from the project root:

```bash
python test_qa_pipeline.py
```

The test exercises multiple research tasks and checks the returned answer structure, evidence, citations, paper coverage, and task-specific semantic behavior.

## Run the Streamlit Application

Start the interactive application with:

```bash
streamlit run streamlit_app.py
```

Then open the local Streamlit URL shown in the terminal.

## Example Questions

```text
What methodology did the researchers use?
```

```text
Compare the methodologies used in both papers.
```

```text
What are the main findings of the papers?
```

```text
Compare the limitations of both papers.
```

```text
What samples or participants were used in each paper?
```

```text
What future research directions do the authors suggest?
```

```text
What are the main contributions of each paper?
```

```text
What factors or variables are examined in the papers?
```

The system is also tested with unsupported questions to verify that it does not manufacture answers when the loaded papers do not contain sufficient evidence.

## Semantic Validation

A central part of the project is distinguishing between information that is **about the paper** and information that is **the answer to the requested task**.

For example:

- A PCA procedure is not automatically a research contribution.
- A questionnaire item is not automatically a participant/sample description.
- A future recommendation is not automatically a limitation.
- A statistical test is not automatically a result.
- A measurement-model validation statement is not automatically a variable list.

The semantic validation layer operates at the sentence level so that retrieved chunks can contain mixed information without forcing all of that information into the final answer.

## Grounding and Citations

Each grounded claim is linked to the evidence chunk from which it was extracted. Citation metadata can include:

- Paper ID
- Paper title
- Page range
- Section
- Chunk ID

This allows the final answer to be traced back to the source material used by the pipeline.

## Hallucination / Support Checking

The verification stage checks whether the claims selected for the final answer are supported by retrieved evidence. A low-support or insufficient-evidence case is returned conservatively instead of being filled with unsupported information.

## Confidence Score

The confidence component is a **reliability indicator**, not a probability that an answer is factually correct. It combines signals such as grounding, citation coverage, retrieval quality, task relevance, and support checks.

## Notes on Large / Private Files

Do not commit private data, API keys, virtual environments, local databases, generated outputs, or large model files to GitHub.

Keep secrets in environment variables or a local `.env` file that is excluded by `.gitignore`.

If a paper cannot legally be redistributed, keep the PDF outside the public repository and document the expected local path instead.

## Future Improvements

Possible extensions include:

- stronger document-level metadata extraction
- better answer evaluation with additional regression cases
- more research-task-specific validators
- richer source highlighting in the UI
- support for larger paper collections
- additional retrieval and reranking experiments

## Author

**Angad Singh Chhabra**

GitHub: https://github.com/angadsingh199999

## License

Add the license appropriate for the intended use of this repository.
