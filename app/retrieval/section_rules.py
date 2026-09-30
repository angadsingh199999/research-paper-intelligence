"""Task-specific retrieval rules.

The semantic gate in app.analysis.task_semantics is authoritative for whether
an evidence sentence can answer a task. These rules only guide retrieval.
"""

from app.analysis.task_semantics import PROFILES


TASK_SECTION_RULES = {
    task: {
        "preferred_sections": profile["preferred_sections"],
        "keywords": profile["positive"],
        "excluded_sections": profile["excluded_sections"],
    }
    for task, profile in PROFILES.items()
}


TASK_QUERY_EXPANSIONS = {
    "methodology": (
        "methodology methods research design study design experimental design "
        "participants sample sampling procedure data collection questionnaire "
        "survey interview measurement analysis experiment qualitative quantitative"
    ),
    "datasets": (
        "dataset data source sample participants respondents observations records "
        "responses data collection"
    ),
    "models": (
        "model models algorithm algorithms architecture framework theoretical model "
        "conceptual model research model"
    ),
    "results": (
        "results findings effects significant hypothesis outcome performance evaluation "
        "statistical analysis mediation moderation regression anova"
    ),
    "limitations": (
        "limitations weaknesses constraints generalizability generalisability "
        "self-reported cross-sectional scenario-based sampling ecological validity "
        "did not measure did not account limited generalizability"
    ),
    "future_work": (
        "future research future work future studies further research recommendations "
        "research directions should examine should explore could examine could explore"
    ),
    "literature_review": (
        "literature review related work prior research previous studies theoretical "
        "background existing studies"
    ),
    "contributions": (
        "contribution contributions theoretical practical implications novel original "
        "extends adds advances"
    ),
    "variables": (
        "variables constructs predictors independent dependent mediator moderator "
        "factors hypotheses measures measurement"
    ),
    "question_answering": "",
}
