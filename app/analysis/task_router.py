from collections import namedtuple


class RouteResult(namedtuple("RouteResult", ["task", "scores"])):
    """A dual tuple-dict object ensuring backward-compatibility across all callers."""

    def __getitem__(self, item):
        if isinstance(item, str):
            if item == "task":
                return self.task
            if item in ("scores", "task_scores"):
                return self.scores
            raise KeyError(item)
        return super().__getitem__(item)

    def get(self, key, default=None):
        if key == "task":
            return self.task
        if key in ("scores", "task_scores"):
            return self.scores
        return default


class TaskRouter:
    """Deterministic research-question intent router.

    Comparison words are treated as a modifier; the content intent is returned
    separately so retrieval can remain focused on the requested information.
    """

    PATTERNS = {
        "limitations": ("limitation", "limitations", "weakness", "weaknesses", "constraint", "constraints", "drawbacks", "shortcomings"),
        "future_work": ("future research", "future work", "future studies", "further research", "research directions", "recommendations for future"),
        "contributions": ("contribution", "contributions", "novel contribution", "theoretical contribution", "practical contribution", "main contribution", "key contribution", "what does the paper add"),
        "methodology": ("methodology", "methodologies", "method", "methods", "research design", "experimental design", "study design", "procedure", "data collection", "sampling approach", "how was the study conducted"),
        "datasets": ("dataset", "datasets", "sample size", "participants", "respondents", "sample", "who participated", "data source"),
        "statistical_analysis": ("statistical analysis", "statistical analyses", "statistical test", "statistical tests", "what analyses", "which tests", "regression", "anova", "pca", "mediation analysis", "moderation analysis", "hypothesis testing"),
        "variables": ("variables", "constructs", "factors", "key variables", "independent variable", "dependent variable", "mediator", "moderator", "what variables", "which variables"),
        "models": ("models", "model", "algorithm", "algorithms", "architecture", "framework", "classifier", "embedding model", "transformer"),
        "results": ("main findings", "key findings", "findings", "results", "main result", "outcomes", "what did the study find", "what did the studies find", "effects", "hypotheses supported"),
        "literature_review": ("literature review", "related work", "previous research", "prior research", "prior studies", "theoretical background"),
    }

    def route(self, question):
        q = " ".join(str(question or "").lower().split())
        scores = {}
        for task, terms in self.PATTERNS.items():
            score = 0
            for term in terms:
                if term in q:
                    score += 3 if " " in term else 2
            if score:
                scores[task] = score
        task = max(scores, key=scores.get) if scores else "question_answering"
        return RouteResult(task, scores)
