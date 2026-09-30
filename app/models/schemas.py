from typing import List, Optional, Union
from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    paper_id: str = ""
    filename: str = ""
    title: str = ""
    authors: List[str] = Field(default_factory=list)
    total_pages: int = 0


class SectionBlock(BaseModel):
    section: str = ""
    text: str = ""
    page_number: int = 0


class Page(BaseModel):
    page_number: int = 0
    blocks: List[SectionBlock] = Field(default_factory=list)


class ResearchPaper(BaseModel):
    metadata: PaperMetadata
    pages: List[Page] = Field(default_factory=list)

    @property
    def paper_id(self) -> str:
        return self.metadata.paper_id if self.metadata else ""

    @property
    def title(self) -> str:
        return self.metadata.title if self.metadata else ""


class Chunk(BaseModel):
    chunk_id: str = ""
    paper_id: str = ""
    paper_title: str = ""
    section: str = ""
    text: str = ""
    page_start: int = 0
    page_end: int = 0
    word_count: int = 0


class Citation(BaseModel):
    paper_id: str = ""
    paper_title: str = ""
    page_start: int = 0
    page_end: int = 0
    section: str = ""
    chunk_id: str = ""


class GroundedClaim(BaseModel):
    claim: str
    evidence_ids: List[str] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)

    # Internal ranking signal. Never display this directly in the UI.
    relevance_score: float = 0.0


class GroundedClaims(BaseModel):
    claims: List[GroundedClaim] = Field(default_factory=list)


class QAResponse(BaseModel):
    answer: str
    task: str
    claims: List[GroundedClaim] = Field(default_factory=list)
    evidence: list = Field(default_factory=list)
    task_scores: dict = Field(default_factory=dict)


# ================================================================
# DOMAIN ANALYSIS SCHEMAS
# ================================================================

class GeneralMethodologyExtraction(BaseModel):
    methodology_type: Optional[str] = ""
    research_design: Optional[str] = ""
    studies: Optional[str] = ""
    sampling_method: Optional[str] = ""
    data_collection: Optional[str] = ""


class PretestExtraction(BaseModel):
    sample_size: Optional[str] = ""
    experimental_conditions: Optional[str] = ""
    data_collection: Optional[str] = ""
    analysis_method: Optional[str] = ""
    purpose: Optional[str] = ""


class StudyExtraction(BaseModel):
    sample_size: Optional[str] = ""
    experimental_conditions: Optional[str] = ""
    product_category: Optional[str] = ""
    data_collection: Optional[str] = ""
    sampling_method: Optional[str] = ""
    analysis_method: Optional[str] = ""


class MethodologyAnalysis(BaseModel):
    methodology_type: Optional[str] = ""
    research_design: Optional[str] = ""
    studies: Optional[str] = ""
    participants: Optional[str] = ""
    sampling_method: Optional[str] = ""
    data_collection: Optional[str] = ""
    analysis_method: Optional[str] = ""
    pretest: Optional[dict] = Field(default_factory=dict)
    study_1: Optional[dict] = Field(default_factory=dict)
    study_2: Optional[dict] = Field(default_factory=dict)
    key_methodological_details: List[str] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)


class DatasetAnalysis(BaseModel):
    datasets: List[str] = Field(default_factory=list)
    data_sources: List[str] = Field(default_factory=list)
    sample_description: Union[str, List[str]] = ""
    citations: List[Citation] = Field(default_factory=list)


class ModelAnalysis(BaseModel):
    models: List[str] = Field(default_factory=list)
    algorithms: List[str] = Field(default_factory=list)
    architectures: List[str] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)


class ResultsAnalysis(BaseModel):
    findings: List[str] = Field(default_factory=list)
    hypothesis_results: List[str] = Field(default_factory=list)
    statistics: List[str] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)


class LimitationAnalysis(BaseModel):
    limitations: List[str] = Field(default_factory=list)
    generalizability: List[str] = Field(default_factory=list)
    methodological_limitations: List[str] = Field(default_factory=list)
    future_directions: List[str] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
