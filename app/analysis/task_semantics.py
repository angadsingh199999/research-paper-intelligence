"""High-precision semantic/type validation for research-paper QA claims.

This module answers one important question that ordinary retrieval and
cross-encoder relevance do not answer:

    "Is this supported sentence actually the *type of information* the
     user requested?"

Examples:
- A sentence can be supported by the paper but still not be a limitation.
- A sentence about Likert items can be supported but still not describe the
  participants/sample.
- A PCA sentence can be supported but still not be a research contribution.

The rules below are intentionally conservative. It is better to return fewer
claims than to put a semantically wrong claim into the final answer.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple


TASK_ALIASES = {
    "contribution": "contributions",
    "main_contribution": "contributions",
    "contributions": "contributions",
    "sample": "datasets",
    "samples": "datasets",
    "participants": "datasets",
    "participant": "datasets",
    "variables": "variables",
    "constructs": "variables",
    "construct": "variables",
    "statistics": "statistical_analysis",
    "statistical": "statistical_analysis",
    "analysis": "statistical_analysis",
    "analyses": "statistical_analysis",
    "theory": "theoretical_framework",
    "framework": "theoretical_framework",
    "comparison": "comparison",
}


NOISE_PATTERNS = [
    r"\barticle processing charges?\b",
    r"\bdeclaration of generative ai\b",
    r"\bgenerative ai and ai-assisted\b",
    r"\binstitutional ethics committee\b",
    r"\binformed consent\b",
    r"\bpersonally identifiable information was not gathered\b",
    r"\bcreativecommons\b",
    r"\bopen access\b",
    r"\bdoi\s*:\s*10\.\d+\b",
    r"^references?\b",
    r"^bibliography\b",
    r"^appendix\b",
    r"^source:\s*",
    r"^table\s*[0-9ivx]*\b",
    r"^fig(?:ure)?\.?\s*[0-9ivx]*\b",
    r"\bfig(?:ure)?\.?\s*[0-9]+\s*(?:shows|presents|illustrates)\b",
    # CRediT authorship contribution statements
    r"^[A-Z][a-z]+(?: [A-Z][a-z]+){1,3}:\s+(?:Formal analysis|Conceptualization|Data curation|Funding acquisition|Investigation|Methodology|Project administration|Resources|Software|Supervision|Validation|Visualization|Writing)",
    r"\bcredit authorship\b",
    r"\bauthorship contribution\b",
    r"\bcontribution statement\b",
]


# Strong task-specific "wrong-type" signals.
TASK_NEGATIVE = {
    "limitations": [
        r"\bfuture research\b",
        r"\bfuture work\b",
        r"\bfuture studies\b",
        r"\bfurther research\b",
        r"\bresearch directions?\b",
        r"\bfuture researchers?\b",
        r"\bfuture scholars?\b",
        r"\bpromising avenues?\b",
        r"\bcan reposition\b",
        r"\bcan enhance\b",
        r"\bwe recommend\b",
        r"\bfindings? (?:indicate|show|suggest)\b",
        r"\bresults? (?:indicate|show|suggest)\b",
        r"\bmajor reason\b",
        r"\bconsistent with (?:the )?.*assumptions?\b",
        r"\bprior studies?\b",
        r"\bprevious studies?\b",
        r"\bliterature\b",
        r"\bprivacy-oriented .* studies?\b",
        r"\bpca\b",
        r"\bprincipal component analysis\b",
        r"\bcredit authorship\b",
    ],
    "future_work": [
        r"\ba limitation\b",
        r"\blimitations? of (?:the|our|this) study\b",
        r"\bthe study is limited\b",
        r"\bthis study did not\b",
        r"\bour sample (?:was|is) limited\b",
        r"\bself-reported\b.*\bmay limit\b",
        r"\bmeasurement model\b",
        r"\bpca\b",
        r"\bprincipal component analysis\b",
        r"\bcronbach\b",
        r"\bcredit authorship\b",
        r"\binformed consent\b",
    ],
    "results": [
        r"\bfuture research\b",
        r"\bfuture work\b",
        r"\bfuture studies\b",
        r"\blimitations?\b",
        r"\bquestionnaire item\b",
        r"\bi would recommend\b",
        r"\bcredit authorship\b",
        r"\bpca was (?:used|employed|conducted)\b",
    ],
    "contributions": [
        r"\bcredit authorship\b",
        r"\bcontribution roles?\b",
        r"\bpca was\b",
        r"\bprincipal component analysis\b",
        r"\bfactor loadings?\b",
        r"\bcomposite reliability\b",
        r"\baverage variance extracted\b",
        r"\bcronbach\b",
        r"\beigenvalue\b",
        r"\bvarimax\b",
        r"\bmeasurement model\b",
        r"\bcross-loading\b",
        r"\bquestionnaire\b.*\bitem\b",
        r"\bparticipants? were\b",
        r"\bdata (?:were|was) collected\b",
        r"\bstatistical analysis\b",
    ],
    "variables": [
        r"\bmeasurement model\b",
        r"\breliability\b",
        r"\bvalidity\b",
        r"\bfactor loadings?\b",
        r"\bcross-loadings?\b",
        r"\bcronbach\b",
        r"\baverage variance extracted\b",
        r"\bcomposite reliability\b",
        r"\bpca\b",
        r"\bprincipal component analysis\b",
        r"\bharman(?:'s)? single-factor\b",
        r"\bcommon method bias\b",
    ],
    "datasets": [
        r"\blikert(?:-type)? scale\b",
        r"\blikert\b.*\bpoint\b",
        r"\bmeasurement items?\b",
        r"\bquestionnaire items?\b",
        r"\badapted items?\b",
        r"\bconstructs? (?:were|was) (?:measured|assessed)\b",
        r"\bpca\b",
        r"\bprincipal component analysis\b",
        r"\bfactor loadings?\b",
        r"\bcomposite reliability\b",
        r"\baverage variance extracted\b",
    ],
    "methodology": [
        r"\bfuture research\b",
        r"\bfuture work\b",
        r"\bfuture studies\b",
        r"\blimitations?\b",
        r"\bresults? (?:show|indicate|suggest)\b",
        r"\bfindings? (?:show|indicate|suggest)\b",
        r"\bcontribution\b",
        r"\btheoretical implication\b",
        r"\bmanagerial implication\b",
    ],
    "statistical_analysis": [
        r"\bfuture research\b",
        r"\bfuture work\b",
        r"\blimitations?\b",
        r"\bwe recommend\b",
    ],
    "models": [
        r"\bfuture research\b",
        r"\blimitations?\b",
    ],
    "theoretical_framework": [
        r"\bfuture research\b",
        r"\blimitations?\b",
    ],
}


TASK_PATTERNS = {
    "methodology": [
        r"\bmethodology\b",
        r"\bmethod(?:s|ological)?\b",
        r"\bresearch design\b",
        r"\bstudy design\b",
        r"\bexperimental design\b",
        r"\bbetween-subject(?:s)?\b",
        r"\bmixed-method(?:s)?\b",
        r"\bdata (?:were|was) collected\b",
        r"\bdata collection\b",
        r"\bquestionnaire\b",
        r"\bsurvey\b",
        r"\bsemi-?structured interview(?:s)?\b",
        r"\binterview(?:s)?\b",
        r"\bsampling\b",
        r"\bpurposive sampling\b",
        r"\bparticipants? were randomly assigned\b",
        r"\bqualtrics\b",
        r"\bscenario\b",
        r"\bexperiment(?:al)?\b",
        r"\bprocedure\b",
        r"\bthe quantitative phase\b",
        r"\bthe qualitative phase\b",
    ],
    "datasets": [
        r"\bparticipants?\b",
        r"\brespondents?\b",
        r"\bsample size\b",
        r"\bfinal sample\b",
        r"\bsample consisted\b",
        r"\bsample included\b",
        r"\bresponses?\b",
        r"\brecruit(?:ed|ment)\b",
        r"\bresponse rate\b",
        r"\busable rate\b",
        r"\bstudents?\b.*\bworkers?\b",
        r"\bage\b.*\b(?:female|male|participants?)\b",
        r"\b(?:female|male)\b.*\bparticipants?\b",
        r"\bonline shopping customers?\b",
        r"\bconsumers?\b.*\bsample\b",
    ],
    "models": [
        r"\b(?:model|models)\b",
        r"\b(?:algorithm|algorithms)\b",
        r"\barchitecture\b",
        r"\bclassifier\b",
        r"\bembedding\b",
        r"\btransformer\b",
        r"\bframework\b",
        r"\btechnology acceptance model\b",
        r"\btam\b",
        r"\bprivacy calculus\b",
    ],
    "results": [
        r"\bresults? (?:show|indicate|suggest|reveal|demonstrate)\b",
        r"\bfindings? (?:show|indicate|suggest|reveal|demonstrate)\b",
        r"\bfound that\b",
        r"\bthe study found\b",
        r"\bthe analysis found\b",
        r"\bstatistically significant\b",
        r"\bsignificant effect\b",
        r"\bsignificant relationship\b",
        r"\bsupported (?:H|hypothesis)\b",
        r"\bhypothesis(?:es)? (?:was|were) supported\b",
        r"\bβ\s*[=<>]",
        r"\bbeta\s*[=<>]",
        r"\bp\s*[<=>]\s*0?\.\d+\b",
        r"\bt\s*\(\s*\d+\s*\)\s*[=<>]",
        r"\bf\s*\(\s*\d+\s*,\s*\d+\s*\)\s*[=<>]",
        r"\br²?\s*[=<>]",
        r"\br2\s*[=<>]",
        r"\bconfidence interval\b",
        r"\beffect size\b",
        r"\bmediation effect\b",
        r"\bindirect effect\b",
        r"\bdirect effect\b",
        r"\binteraction effect\b",
        r"\bmoderation effect\b",
        r"\bmore likely to\b",
        r"\bpositively (?:evaluate|evaluated)\b",
        r"\badoption intention\b.*\b(?:increased|higher|greater)\b",
    ],
    "limitations": [
        r"\blimitation(?:s)?\b",
        r"\bweakness(?:es)?\b",
        r"\bconstraint(?:s)?\b",
        r"\b(?:may|might|can)\s+limit\b",
        r"\blimited\s+(?:to|by|in|scope|generalizability|generalisation)\b",
        r"\black of\b",
        r"\bdid not\b",
        r"\bdidn't\b",
        r"\bwas not\b",
        r"\bwere not\b",
        r"\bnot directly measured\b",
        r"\bnot empirically tested\b",
        r"\bself[- ]reported\b",
        r"\bscenario[- ]based\b",
        r"\bimagined scenario\b",
        r"\bhypothetical scenario\b",
        r"\bconvenience sampling\b",
        r"\bsnowball sampling\b",
        r"\bcross[- ]sectional\b",
        r"\brestricted to\b",
        r"\bsingle (?:country|region|context)\b",
        r"\bsmall sample\b",
        r"\bonly\b.*\bparticipants?\b",
        r"\bonly\b.*\bconsumers?\b",
    ],
    "future_work": [
        r"\bfuture research\b",
        r"\bfuture work\b",
        r"\bfuture studies\b",
        r"\bfurther research\b",
        r"\bresearch directions?\b",
        r"\bnext research\b",
        r"\bfuture researchers?\b",
        r"\bfuture scholars?\b",
        r"\bshould (?:examine|explore|investigate|test|evaluate|replicate|extend|consider|assess|address)\b",
        r"\bcould (?:examine|explore|investigate|test|evaluate|replicate|extend|consider|assess|address)\b",
        r"\bmay (?:examine|explore|investigate|test|extend|address) in future\b",
        r"\bwe recommend\b.*\b(?:future|further)\b",
        r"\brecommend(?:s|ed)?\b.*\b(?:future|further)\b",
        r"\bshould be explored in future studies\b",
    ],
    "contributions": [
        r"\bcontribution(?:s)?\b",
        r"\btheoretical contribution\b",
        r"\bpractical contribution\b",
        r"\bmanagerial contribution\b",
        r"\bcontributes? to\b",
        r"\bextend(?:s|ed)?\b.*\bliterature\b",
        r"\badvance(?:s|d)?\b.*\bunderstanding\b",
        r"\badvances?\b.*\bliterature\b",
        r"\bnovel\b.*\b(?:framework|insight|contribution|perspective)\b",
        r"\bintroduces?\b.*\bframework\b",
        r"\bintegrat(?:es|ing|ed)\b.*\b(?:tam|privacy calculus)\b",
        r"\bcombining\b.*\b(?:tam|privacy calculus)\b",
        r"\bidentifies?\b.*\b(?:boundary|condition|mechanism)\b",
        r"\bclarifies?\b.*\b(?:role|relationship|mechanism)\b",
        r"\badds? to\b.*\bliterature\b",
        r"\bimplication(?:s)?\b.*\btheory\b",
        r"\bimplication(?:s)?\b.*\bpractice\b",
    ],
    "statistical_analysis": [
        r"\bpca\b",
        r"\bprincipal component analysis\b",
        r"\bregression\b",
        r"\blinear regression\b",
        r"\banova\b",
        r"\bt-test\b",
        r"\bchi-square\b",
        r"\bcorrelation\b",
        r"\bmediation\b",
        r"\bmoderation\b",
        r"\bmoderated mediation\b",
        r"\bbootstrap(?:ping)?\b",
        r"\bprocess macro\b",
        r"\bprocess macro model\b",
        r"\bstructural equation modeling\b",
        r"\bsem\b",
        r"\bconfidence interval\b",
        r"\bvariance inflation factor\b",
        r"\bvif\b",
        r"\bfactor analysis\b",
        r"\btest(?:ed|ing) the hypotheses\b",
    ],
    "variables": [
        r"\bvariables?\b",
        r"\bconstructs?\b",
        r"\bindependent variable\b",
        r"\bdependent variable\b",
        r"\bpredictor\b",
        r"\bmediator\b",
        r"\bmoderator\b",
        r"\bmediating\b",
        r"\bmoderating\b",
        r"\bperceived usefulness\b",
        r"\bperceived ease of use\b",
        r"\bprivacy concern\b",
        r"\bprivacy concerns\b",
        r"\battitude(?: toward| towards)? vto\b",
        r"\battitude(?: toward| towards)? the (?:brand|product)\b",
        r"\bwillingness to (?:buy|use|adopt)\b",
        r"\bpurchase intention\b",
        r"\breturn waste reduction\b",
        r"\baccuracy of product selection\b",
        r"\bproduct selection accuracy\b",
        r"\bap(?:s|s)\b",
        r"\brwr\b",
        r"\bvto\b.*\b(?:construct|variable)\b",
    ],
    "theoretical_framework": [
        r"\btechnology acceptance model\b",
        r"\btam\b",
        r"\bprivacy calculus\b",
        r"\btheoretical framework\b",
        r"\bconceptual framework\b",
        r"\btheory\b",
        r"\btheoretical model\b",
    ],
    "literature_review": [
        r"\bliterature\b",
        r"\brelated work\b",
        r"\bprior research\b",
        r"\bprevious research\b",
        r"\bprior studies?\b",
        r"\bprevious studies?\b",
        r"\bexisting studies?\b",
        r"\bresearch gap\b",
        r"\bthe literature suggests\b",
        r"\bthe literature shows\b",
    ],
}


def normalize_task(task: str | None) -> str:
    value = str(task or "question_answering").strip().lower()
    return TASK_ALIASES.get(value, value)


def normalize_text(text: str | None) -> str:
    value = str(text or "")
    value = value.replace("\u00a0", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def normalize_section(section: str | None) -> str:
    value = normalize_text(section).lower()
    value = re.sub(r"^[0-9.\-\s]+", "", value)
    return value


def _matches(patterns: Iterable[str], text: str) -> List[str]:
    hits = []
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            hits.append(pattern)
    return hits


def _has_any(patterns: Iterable[str], text: str) -> bool:
    return bool(_matches(patterns, text))


def is_noise(text: str) -> bool:
    value = normalize_text(text).lower()
    if not value:
        return True
    if len(value.split()) < 6:
        return True
    if len(re.findall(r"[A-Za-z]", value)) < 20:
        return True
    if _has_any(NOISE_PATTERNS, value):
        return True
    if re.search(r"\b\d{1,3}\s+\|\s*\d{1,3}\b", value):
        return True
    if re.search(r"\b(?:page|pp?)\.?\s*\d+\b", value) and len(value.split()) < 18:
        return True
    # Common PDF extraction corruption seen in captions/figure fragments.
    if re.search(r"\bhas a fig(?:ure)?\.?\s*\d+\b", value):
        return True
    if re.search(r"\bfig(?:ure)?\.?\s*\d+\.?\s*$", value):
        return True
    if re.search(r"\b(?:95%|90%|99%)\s+c\.i\.?\s*$", value):
        return True
    return False


def _limitations_valid(text: str, section: str, positive_hits: List[str]) -> Tuple[bool, str]:
    low = text.lower()
    section_low = normalize_section(section)

    # Literature/background statements about limitations of other studies are
    # not limitations of the present paper.
    if _has_any(
        [
            r"\bprior studies?\b",
            r"\bprevious studies?\b",
            r"\bexisting studies?\b",
            r"\bprivacy-oriented(?: \w+){0,3} studies?\b",
            r"\bother studies?\b",
            r"\bthe literature\b",
            r"\bresearch has shown\b",
            r"\bstudies?\s+lack of\b",
        ],
        low,
    ) and not _has_any(
        [r"\bthis study\b", r"\bthe present study\b", r"\bour study\b", r"\bwe\b"],
        low,
    ):
        return False, "literature_or_other_studies"

    # These phrases are forward-looking recommendations, never study limitations,
    # even when the word "limitation" appears elsewhere in the sentence.
    if _has_any(
        [
            r"\bpromising avenues?\b",
            r"\bcan reposition\b",
            r"\bcan enhance\b",
            r"\bwe recommend\b",
        ],
        low,
    ):
        return False, "forward_looking_recommendation"

    # Statistical confirmation sentences (results/findings), not limitations.
    if _has_any(
        [
            r"\bplanned contrasts? confirmed\b",
            r"\bdid not differ\b",
            r"\bsignificantly differed\b",
            r"\bconfirmed that .{0,60} significantly\b",
        ],
        low,
    ):
        return False, "statistical_result"

    # Participant recruitment / sample description sentences (methodology), not limitations.
    if re.match(
        r"^participants?\s*\(",
        low,
        flags=re.IGNORECASE,
    ):
        return False, "participant_demographics"

    # "Were not above / did not exceed the threshold" validity-check sentences (method/results), not limitations.
    if _has_any(
        [
            r"\bwere not above the recommended\b",
            r"\bdid not exceed the recommended\b",
            r"\bnot above the threshold\b",
            r"\bhtmt\b",
        ],
        low,
    ):
        return False, "validity_check_result"

    # Convenience sampling alone (without explicit limitation language) describes methodology.
    if _has_any([r"\bconvenience sampling\b", r"\bsnowball sampling\b"], low):
        if not _has_any(
            [r"\bmay limit\b", r"\blimitation\b", r"\bgeneralizab\b", r"\bgeneralisab\b"],
            low,
        ):
            return False, "sampling_method_not_limitation"

    # Statistical-result sentences (t-test, F-test, beta coefficients) are
    # findings, NOT limitations, even when negation words appear.
    if _has_any(
        [
            r"\bt\s*\(\s*[\d.]+\s*\)\s*=",
            r"\bf\s*\(\s*[\d,]+\s*\)\s*=",
            r"\bchi.?square\s*=",
            r"\b(?:was|were|is|are)\s+(?:negative|positive)\s+and\s+statistically\s+significant\b",
            r"\bgames.howell\b",
            r"\bwas significant\b",
            r"\bwere significant\b",
            r"\b(?:contrast|comparison)\s+(?:was|were|is)\s+(?:not\s+)?significant\b",
        ],
        low,
    ):
        return False, "statistical_result"

    # Research-design / purpose statements describe what the study DID,
    # not what it was unable to do.
    if _has_any(
        [
            r"\bthe purpose of this\b",
            r"\bthe purpose of the\b",
            r"\bwas designed to\b",
            r"\bwere designed to\b",
            r"\bqualitative stage was\b",
            r"\binterviews? investigated\b",
            r"\baiming to clarify\b",
        ],
        low,
    ):
        return False, "research_design_statement"

    future_only = _has_any(
        [
            r"\bfuture research\b",
            r"\bfuture studies?\b",
            r"\bfuture work\b",
            r"\bresearch directions?\b",
            r"\bpromising avenues?\b",
            r"\bcan reposition\b",
            r"\bshould explore\b",
            r"\bshould examine\b",
            r"\bcould explore\b",
            r"\bcould examine\b",
        ],
        low,
    )
    if future_only and not _has_any(
        [
            r"\blimitation(?:s)?\b",
            r"\bweakness(?:es)?\b",
            r"\bconstraint(?:s)?\b",
            r"\bmay limit\b",
            r"\bmight limit\b",
            r"\bself[- ]reported\b",
            r"\bconvenience sampling\b",
            r"\bsnowball sampling\b",
            r"\bscenario[- ]based\b",
            r"\bdid not\b",
            r"\bnot directly measured\b",
            r"\brestricted to\b",
        ],
        low,
    ):
        return False, "forward_looking"

    # Phrases such as "lack of fit confidence is a major reason" are empirical
    # findings, not limitations, unless they explicitly describe a study
    # constraint.
    if _has_any(
        [
            r"\bmajor reason\b",
            r"\bmore likely\b",
            r"\bpositively evaluate\b",
            r"\bresults? (?:show|indicate|suggest)\b",
            r"\bfindings? (?:show|indicate|suggest)\b",
        ],
        low,
    ) and not _has_any(
        [
            r"\blimitation(?:s)?\b",
            r"\bconstraint(?:s)?\b",
            r"\bweakness(?:es)?\b",
            r"\bmay limit\b",
            r"\blimited (?:to|by|in)\b",
            r"\bdid not\b",
            r"\bnot directly measured\b",
        ],
        low,
    ):
        return False, "finding_not_limitation"

    # Cross-sectional / small-sample / demographic restrictions are acceptable
    # as limitations when the sentence is located in a limitation section. It
    # is unsafe to accept them everywhere because they also occur in methods.
    contextual_only = _has_any(
        [
            r"\bcross[- ]sectional\b",
            r"\bsmall sample\b",
            r"\byoung adults?\b",
            r"\bsingle (?:country|region|context)\b",
        ],
        low,
    ) and not positive_hits
    if contextual_only and not _has_any(
        [r"limitation", r"weakness", r"constraint", r"this study", r"our study", r"the present study"],
        section_low + " " + low,
    ):
        return False, "context_only_outside_limitation_context"

    if positive_hits:
        return True, "limitation_signal"
    return False, "no_limitation_signal"


def _future_valid(text: str, section: str, positive_hits: List[str]) -> Tuple[bool, str]:
    low = text.lower()
    if not positive_hits:
        return False, "no_future_signal"

    # A limitation statement is not future work unless the same sentence also
    # contains an explicit forward-looking recommendation.
    limitation_signal = _has_any(
        [
            r"\blimitation(?:s)?\b",
            r"\bweakness(?:es)?\b",
            r"\bconstraint(?:s)?\b",
            r"\bthis study did not\b",
            r"\bwas limited\b",
            r"\bwere limited\b",
        ],
        low,
    )
    forward_recommendation = _has_any(
        [
            r"\bfuture research\b",
            r"\bfuture studies?\b",
            r"\bfurther research\b",
            r"\bshould (?:examine|explore|investigate|test|replicate|extend|evaluate|address|assess|consider)\b",
            r"\bcould (?:examine|explore|investigate|test|replicate|extend|evaluate|address|assess|consider)\b",
            r"\brecommend(?:s|ed)?\b.*\b(?:future|further)\b",
        ],
        low,
    )
    if limitation_signal and not forward_recommendation:
        return False, "limitation_not_future_work"
    return True, "future_signal"


def _dataset_valid(text: str, section: str, positive_hits: List[str]) -> Tuple[bool, str]:
    low = text.lower()
    if not positive_hits:
        return False, "no_sample_signal"
    if _has_any(TASK_NEGATIVE["datasets"], low):
        return False, "measurement_not_sample"
    # A sentence that explicitly discusses questionnaire constructs is a
    # measurement sentence even when it also says "responses".
    if _has_any(
        [r"\bconstruct(?:s)?\b", r"\bmeasurement\b", r"\bitems?\b"], low
    ) and not _has_any(
        [r"\bparticipants?\b", r"\brespondents?\b", r"\bfinal sample\b", r"\bsample size\b"], low
    ):
        return False, "construct_or_measurement_sentence"
    return True, "sample_signal"


def _results_valid(text: str, section: str, positive_hits: List[str]) -> Tuple[bool, str]:
    low = text.lower()
    if not positive_hits:
        return False, "no_result_signal"
    if _has_any(
        [
            r"\bprocess macro\b.*\b(?:used|employed|conducted)\b",
            r"\bpca was (?:used|employed|conducted)\b",
            r"\bthe questionnaire was\b",
            r"\bdata (?:were|was) collected\b",
            r"\bparticipants? were randomly assigned\b",
            r"\bconsistent with (?:the )?(?:tam|technology acceptance model).*assumptions?\b",
        ],
        low,
    ) and not _has_any(
        [
            r"results? (?:show|indicate|suggest)\b",
            r"findings? (?:show|indicate|suggest)\b",
            r"found that\b",
            r"\bsignificant\b",
            r"\bβ\s*[=<>]",
            r"\bp\s*[<=>]",
            r"\bt\s*\(",
            r"\beffect\b",
            r"\bsupported\b",
            r"\bmore likely\b",
            r"\bmediation\b",
            r"\bmoderation\b",
        ],
        low,
    ):
        return False, "method_not_result"
    if re.search(r"\b(?:fig(?:ure)?\.?\s*\d+|has a fig(?:ure)?\.?\s*\d+)\b", low):
        return False, "figure_fragment"
    if re.search(r"\b(?:95%|90%|99%)\s+c\.i\.?\s*$", low):
        return False, "truncated_statistics"
    return True, "result_signal"


def _contribution_valid(text: str, section: str, positive_hits: List[str]) -> Tuple[bool, str]:
    low = text.lower()
    section_low = normalize_section(section)
    if _has_any(TASK_NEGATIVE["contributions"], low):
        return False, "method_or_authorship_not_contribution"
    explicit = _has_any(
        [r"\bcontribution(?:s)?\b", r"\bcontributes? to\b", r"\bnovel\b", r"\bextends?\b", r"\badvances?\b", r"\bintroduces?\b", r"\bintegrat(?:es|ing|ed)\b", r"\badds? to\b"],
        low,
    )
    section_signal = _has_any([r"contribution", r"implication", r"discussion"], section_low)
    conceptual = _has_any(
        [
            r"\b(?:tam|technology acceptance model)\b",
            r"\bprivacy calculus\b",
            r"\bvirtual try-on\b",
            r"\bsustainab\w+\b",
            r"\breturn waste\b",
            r"\bbrand[- ]related\b",
            r"\bboundary condition\b",
            r"\bmechanism\b",
        ],
        low,
    )
    if positive_hits and (explicit or (section_signal and conceptual)):
        return True, "contribution_signal"
    return False, "no_contribution_signal"


def _variables_valid(text: str, section: str, positive_hits: List[str]) -> Tuple[bool, str]:
    low = text.lower()

    # Literature-review openers are citations about OTHER work, not study variables.
    if re.match(r"^(?:recent|prior|previous|existing)\s+(?:research|studies|literature|work)\b", low, re.IGNORECASE):
        return False, "literature_citation"

    # Hypothesis-test results ("supporting H2", "consistent with H3") are
    # findings, not variable descriptions.
    if _has_any(
        [
            r"\bsupporting h\s*\d+\b",
            r"\bconsistent with h\s*\d+\b",
            r"\bh\s*\d+\s+(?:was|is|were)\s+(?:supported|confirmed|accepted|rejected)\b",
        ],
        low,
    ):
        return False, "hypothesis_test_result"

    # Results/findings framing sentences are not variable descriptions.
    if re.match(r"^results of\b", low, re.IGNORECASE):
        return False, "results_statement"

    # Factor loading / validity check sentences are not variable descriptions.
    if _has_any(
        [
            r"\bdemonstrated satisfactory loadings\b",
            r"\bfactor loadings?\b",
            r"\bwere therefore retained\b",
            r"\bcomposite reliability\b",
            r"\bcronbach.{0,5}alpha\b",
        ],
        low,
    ):
        return False, "psychometric_validity_check"

    # "These findings indicate / show / suggest" = results statement.
    if re.match(r"^these findings\b", low, re.IGNORECASE):
        return False, "findings_statement"

    # Statistical-analysis-results sentences describe findings, not variables.
    # Structural openers that always introduce a result/finding.
    if re.match(r"^results? (?:indicated?|showed?|demonstrated?|revealed?|suggested?)\b", low, re.IGNORECASE):
        return False, "results_opener"

    if re.match(r"^the model (?:explains?|accounts? for|fit)\b", low, re.IGNORECASE):
        return False, "model_fit_statement"

    if re.match(r"^overall[,.]?\s+the results?\b", low, re.IGNORECASE):
        return False, "results_summary"

    if re.match(r"^to further assess\b", low, re.IGNORECASE):
        return False, "analysis_procedure"

    if _has_any(
        [
            r"\baccounted for\s+[\d.]+%?\s+of\s+the\s+variance\b",
            r"\bexplains?\s+[\d.]+%?\s+of\s+the\s+variance\b",
            r"\bprocess macro\b",
            r"\bmodel\s+[47]\b",
            r"\bmoderated.mediation analysis\b",
            r"\bmediation analysis\b",
            r"\bpositively predicted\b",
            r"\bsignificantly predicted\b",
            r"\bnegatively predicted\b",
            r"\bconditional indirect effect\b",
            r"\bindex of moderat(?:or|ed) mediation\b",
            r"\bprovided support for all proposed hypotheses\b",
            r"\bconfirmed that greater acceptance\b",
        ],
        low,
    ):
        return False, "statistical_analysis_result"

    # Variance-explained / AVE / measurement-model validity checks.
    if re.match(r"^this confirms\b", low, re.IGNORECASE):
        return False, "measurement_model_validity"

    if _has_any(
        [
            r"\bover\s+50\s+percent\s+of\s+the\s+variance\b",
            r"\bsuitability of the model\b",
            r"\baverage variance extracted\b",
            r"\b\bave\b.*\bindicator\b",
        ],
        low,
    ):
        return False, "measurement_model_validity"

    # Any sentence containing a beta coefficient (beta = X.XX) is reporting
    # a statistical finding, never a variable description.
    if _has_any(
        [
            r"\u03b2\s*(?:[\w-]+\s*)?=\s*[\d.+-]",
            r"\bbeta\s*(?:[\w-]+\s*)?=\s*[\d.+-]",
            r"\bindirect effects?\s+of\b",
            r"\bremained a significant predictor\b",
            r"\bindicating partial mediation\b",
            r"\bCI\s*\[",
            r"\bbias.corrected\s+(?:ci|confidence interval)\b",
        ],
        text,
    ):
        return False, "statistical_coefficient_result"

    # Future-research / suggestion sentences belong to future_work, not variables.
    if _has_any(
        [
            r"\bfuture work should\b",
            r"\bfuture studies? should\b",
            r"\bfuture research should\b",
            r"\brobustness would be strengthened\b",
            r"\bwould be strengthened by testing\b",
            r"\btesting alternative outcome variables\b",
            r"\bto assess whether the effects generalize\b",
        ],
        low,
    ):
        return False, "future_work_not_variables"

    # Correlation-result sentences (r = X, p = Y) are statistical findings.
    if _has_any(
        [
            r"\br\s*=\s*-?[\d.]+,\s*p\s*[<>=]",
            r"\bnon.significant correlations?\b",
            r"\bnegligible\s+and\s+non.significant\b",
        ],
        low,
    ):
        return False, "correlation_result"

    # Ordinal-prefixed limitation sentences ("Fourth, the study did not...")
    if re.match(
        r"^(?:first|second|third|fourth|fifth|sixth|seventh)[,.]?\s+the study\b",
        low, re.IGNORECASE,
    ):
        return False, "ordinal_limitation"

    # Psychometric-bias / common-method checks are methodology quality tests,
    # NOT descriptions of study variables.
    if _has_any(
        [
            r"\bharman\b",
            r"\bmarker.variable approach\b",
            r"\bcommon method bias\b",
            r"\bcommon method variance\b",
            r"\bpodsakoff\b",
        ],
        low,
    ):
        return False, "psychometric_bias_check"

    if _has_any(TASK_NEGATIVE["variables"], low):
        return False, "measurement_validation_not_variables"
    if not positive_hits:
        return False, "no_variable_signal"

    # Require either explicit variable/construct language or a named construct
    # used in a relational/modeling context. This blocks generic measurement
    # quality statements.
    explicit = _has_any(
        [
            r"\bvariables?\b",
            r"\bconstructs?\b",
            r"\bindependent variable\b",
            r"\bdependent variable\b",
            r"\bmediator\b",
            r"\bmoderator\b",
            r"\bpredictor\b",
            r"\bmeasure(?:d|ment)?\b",
            r"\boperationali[sz]\b",
        ],
        low,
    )
    named_construct = _has_any(
        [
            r"\bperceived usefulness\b",
            r"\bperceived ease of use\b",
            r"\bprivacy concerns?\b",
            r"\battitude(?: toward| towards)? vto\b",
            r"\bwillingness to (?:buy|use|adopt)\b",
            r"\bpurchase intention\b",
            r"\breturn waste reduction\b",
            r"\baccuracy of product selection\b",
            r"\bproduct selection accuracy\b",
            r"\baps\b",
            r"\brwr\b",
            r"\bvto\b",
        ],
        low,
    )
    if explicit or named_construct:
        return True, "variable_signal"
    return False, "weak_variable_signal"


def task_decision(task: str, text: str, section: str = "") -> Dict[str, object]:
    """Return a conservative task-validation decision for one sentence."""
    task = normalize_task(task)
    text = normalize_text(text)
    section = normalize_section(section)

    if is_noise(text):
        return {"valid": False, "score": 0.0, "reason": "noise", "task": task}

    if task in TASK_NEGATIVE and _has_any(TASK_NEGATIVE[task], text.lower()):
        # Some tasks need custom handling where a negative pattern is allowed
        # in the presence of a stronger positive signal.
        if task not in {"limitations", "future_work", "results", "contributions", "variables", "datasets"}:
            return {"valid": False, "score": 0.0, "reason": "negative_signal", "task": task}

    patterns = TASK_PATTERNS.get(task, [])
    positive_hits = _matches(patterns, text.lower())

    if task == "limitations":
        valid, reason = _limitations_valid(text, section, positive_hits)
    elif task == "future_work":
        valid, reason = _future_valid(text, section, positive_hits)
    elif task == "datasets":
        valid, reason = _dataset_valid(text, section, positive_hits)
    elif task == "results":
        valid, reason = _results_valid(text, section, positive_hits)
    elif task == "contributions":
        valid, reason = _contribution_valid(text, section, positive_hits)
    elif task == "variables":
        valid, reason = _variables_valid(text, section, positive_hits)
    else:
        if not positive_hits:
            valid, reason = False, "no_task_signal"
        elif _has_any(TASK_NEGATIVE.get(task, []), text.lower()):
            valid, reason = False, "negative_signal"
        else:
            valid, reason = True, "task_signal"

    # Section support is a *secondary* signal. It never overrides a bad
    # sentence-level type decision.
    if valid:
        score = min(1.0, 0.55 + 0.08 * min(len(positive_hits), 4))
        if any(token in section for token in {
            task.replace("_", " "),
            "limitation" if task == "limitations" else "",
            "future" if task == "future_work" else "",
            "contribution" if task == "contributions" else "",
            "result" if task == "results" else "",
            "finding" if task == "results" else "",
            "sample" if task == "datasets" else "",
            "participant" if task == "datasets" else "",
            "design" if task == "methodology" else "",
            "method" if task == "methodology" else "",
            "variable" if task == "variables" else "",
            "construct" if task == "variables" else "",
        } - {""}):
            score = min(1.0, score + 0.12)
    else:
        score = 0.0

    return {"valid": valid, "score": score, "reason": reason, "task": task}


def is_task_valid(task: str, text: str, section: str = "") -> bool:
    return bool(task_decision(task, text, section).get("valid", False))


def filter_task_sentences(task: str, sentences: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Filter (sentence, section) pairs through the strict task gate."""
    output = []
    for sentence, section in sentences:
        if is_task_valid(task, sentence, section):
            output.append((normalize_text(sentence), normalize_section(section)))
    return output

# ---------------------------------------------------------------------------
# Backward-compatible API aliases
# ---------------------------------------------------------------------------
# Some versions of app/retrieval/task_retriever.py import this older helper
# name. Keep it available so the semantic layer remains compatible with both
# the older and newer retriever implementations.
def is_task_valid_sentence(arg1: str, arg2: str, section: str = "") -> bool:
    """Compatibility wrapper for older task-retriever code.

    Supports both common calling conventions:
        is_task_valid_sentence(task, sentence, section)
        is_task_valid_sentence(sentence, task, section)
    """
    known_tasks = set(TASK_PATTERNS) | set(TASK_ALIASES)
    first = str(arg1 or "").strip().lower()
    second = str(arg2 or "").strip().lower()

    if first in known_tasks:
        task_name = first
        sentence = arg2
    elif second in known_tasks:
        sentence = arg1
        task_name = second
    else:
        # Default to the newer API order: (task, sentence, section).
        task_name = first
        sentence = arg2

    return is_task_valid(task_name, sentence, section)


def task_semantic_score(task: str, text: str, section: str = "") -> float:
    """Return the normalized semantic task score used by retrieval code."""
    return float(task_decision(task, text, section).get("score", 0.0) or 0.0)


def get_task_semantic_score(task: str, text: str, section: str = "") -> float:
    """Backward-compatible alias for task_semantic_score()."""
    return task_semantic_score(task, text, section)


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

_ABBREVIATIONS_SS = (
    "et al.", "e.g.", "i.e.", "cf.", "vs.", "etc.", "fig.", "eq.",
    "approx.", "dr.", "prof.", "mr.", "mrs.", "ms.", "no.", "c.i.",
    "c.i", "s.d.", "s.d", "p.", "pp.", "vol.", "inc.", "U.S.",
)
_PLACEHOLDER_SS = "\uE001"


def split_sentences(text: str) -> List[str]:
    """Split text into sentences, preserving abbreviations and decimals."""
    value = normalize_text(text)
    if not value:
        return []
    for abbr in _ABBREVIATIONS_SS:
        safe = abbr.replace(".", _PLACEHOLDER_SS)
        value = re.sub(re.escape(abbr), safe, value, flags=re.IGNORECASE)
    # Protect decimal numbers like 0.05
    value = re.sub(r"(?<=\d)\.(?=\d)", _PLACEHOLDER_SS, value)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", value)
    output = []
    for part in parts:
        part = part.replace(_PLACEHOLDER_SS, ".")
        part = normalize_text(part)
        if len(part.split()) >= 6:
            output.append(part)
    return output


# ---------------------------------------------------------------------------
# Query expansion strings per task
# ---------------------------------------------------------------------------

_QUERY_EXPANSION_MAP: Dict[str, str] = {
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
    "statistical_analysis": (
        "pca principal component analysis regression anova t-test chi-square "
        "correlation mediation moderation bootstrapping structural equation modeling sem "
        "confidence interval variance inflation factor factor analysis"
    ),
    "theoretical_framework": (
        "technology acceptance model tam privacy calculus theoretical framework "
        "conceptual framework theory theoretical model"
    ),
    "question_answering": "",
}


def query_expansion(task: str) -> str:
    """Return a space-separated expansion string for the given task."""
    task = normalize_task(task)
    return _QUERY_EXPANSION_MAP.get(task, "")


# ---------------------------------------------------------------------------
# PROFILES dict — section_rules.py imports this
# ---------------------------------------------------------------------------

PROFILES: Dict[str, Dict] = {
    "methodology": {
        "preferred_sections": ["methodology", "method", "methods", "design", "procedure", "data collection", "study design"],
        "positive": list(_QUERY_EXPANSION_MAP["methodology"].split()),
        "excluded_sections": ["abstract", "references", "bibliography"],
    },
    "datasets": {
        "preferred_sections": ["sample", "participants", "data", "respondents"],
        "positive": list(_QUERY_EXPANSION_MAP["datasets"].split()),
        "excluded_sections": ["abstract", "references", "bibliography"],
    },
    "models": {
        "preferred_sections": ["model", "framework", "theory", "conceptual"],
        "positive": list(_QUERY_EXPANSION_MAP["models"].split()),
        "excluded_sections": ["abstract", "references", "bibliography"],
    },
    "results": {
        "preferred_sections": ["results", "findings", "analysis", "outcome"],
        "positive": list(_QUERY_EXPANSION_MAP["results"].split()),
        "excluded_sections": ["abstract", "references", "bibliography"],
    },
    "limitations": {
        "preferred_sections": ["limitation", "limitations", "weakness", "constraint"],
        "positive": list(_QUERY_EXPANSION_MAP["limitations"].split()),
        "excluded_sections": ["abstract", "references", "bibliography", "future"],
    },
    "future_work": {
        "preferred_sections": ["future", "future work", "future research", "recommendation"],
        "positive": list(_QUERY_EXPANSION_MAP["future_work"].split()),
        "excluded_sections": ["abstract", "references", "bibliography"],
    },
    "literature_review": {
        "preferred_sections": ["literature", "background", "related work", "prior research"],
        "positive": list(_QUERY_EXPANSION_MAP["literature_review"].split()),
        "excluded_sections": ["abstract", "references", "bibliography"],
    },
    "contributions": {
        "preferred_sections": ["contribution", "contributions", "implication", "discussion"],
        "positive": list(_QUERY_EXPANSION_MAP["contributions"].split()),
        "excluded_sections": ["abstract", "references", "bibliography", "method", "data"],
    },
    "variables": {
        "preferred_sections": ["variable", "construct", "hypothesis", "measure", "model"],
        "positive": list(_QUERY_EXPANSION_MAP["variables"].split()),
        "excluded_sections": ["abstract", "references", "bibliography"],
    },
    "statistical_analysis": {
        "preferred_sections": ["analysis", "results", "statistical", "method"],
        "positive": list(_QUERY_EXPANSION_MAP["statistical_analysis"].split()),
        "excluded_sections": ["abstract", "references", "bibliography"],
    },
    "theoretical_framework": {
        "preferred_sections": ["theory", "framework", "conceptual", "background"],
        "positive": list(_QUERY_EXPANSION_MAP["theoretical_framework"].split()),
        "excluded_sections": ["abstract", "references", "bibliography"],
    },
    "question_answering": {
        "preferred_sections": [],
        "positive": [],
        "excluded_sections": [],
    },
}


# ---------------------------------------------------------------------------
# Task detection and comparison detection
# ---------------------------------------------------------------------------

# Ordered list of (task_name, keyword_patterns) for deterministic routing.
# The first matching task wins.
_TASK_DETECTION_RULES: List[tuple] = [
    ("future_work", [
        r"\bfuture research\b", r"\bfuture work\b", r"\bfuture stud",
        r"\bfurther research\b", r"\bnext steps?\b", r"\bshould (?:explore|examine|investigate)\b",
        r"\bfuture direction\b",
    ]),
    ("limitations", [
        r"\blimitation", r"\bweakness(?:es)?\b", r"\bconstraint\b",
        r"\bshortcoming\b", r"\blimited\b.*\bstudy\b", r"\bstudy.*\blimited\b",
    ]),
    ("contributions", [
        r"\bcontributions?\b", r"\bcontribute\b", r"\bnovel\b.*\b(?:framework|insight)\b",
        r"\bwhat.*(?:adds?|advances?|extends?)\b", r"\btheoretical.*implication\b",
        r"\bmanagerial.*implication\b",
    ]),
    ("results", [
        r"\bresults?\b", r"\bfindings?\b", r"\bhypothes[ie]\b", r"\beffect\b",
        r"\boutcome\b", r"\bsignificant\b", r"\bsupported\b.*\bhypothes",
        r"\bwhat.*found\b", r"\bwhat.*show\b", r"\bwhat.*reveal\b",
    ]),
    ("methodology", [
        r"\bmethodolog(?:y|ies)\b", r"\bmethods?\b", r"\bresearch design\b", r"\bstudy design\b",
        r"\bhow.*(?:collect|conduct|design|recruit|sample)\b",
        r"\bdata.*collection\b", r"\bsurvey\b", r"\binterview\b",
        r"\bexperiment\b", r"\bprocedure\b",
    ]),
    ("datasets", [
        r"\bsample\b", r"\bparticipants?\b", r"\brespondents?\b",
        r"\bwho.*(?:participate|respond)\b", r"\bsample size\b",
        r"\bhow many\b.*\b(?:participant|respondent|subject)\b",
    ]),
    ("variables", [
        r"\bvariables?\b", r"\bconstructs?\b", r"\bpredictor\b",
        r"\bmediator\b", r"\bmoderator\b",
        r"\bindependent variable\b", r"\bdependent variable\b",
        r"\bwhat.*(?:measure|construct|variable|factor)\b", r"\bwhat factors\b", r"\bkey factors\b",
    ]),
    ("statistical_analysis", [
        r"\bpca\b", r"\bregression\b", r"\banova\b", r"\bt-test\b",
        r"\bchi-square\b", r"\bcorrelation\b", r"\bmediation\b", r"\bmoderation\b",
        r"\bbootstrap\b", r"\bstructural equation\b", r"\bsem\b",
        r"\bstatistical\b.*\banalysis\b",
    ]),
    ("models", [
        r"\bmodel\b", r"\bframework\b", r"\balgorithm\b",
        r"\btechnology acceptance\b", r"\btam\b", r"\bprivacy calculus\b",
        r"\bconceptual model\b", r"\btheoretical model\b",
    ]),
    ("theoretical_framework", [
        r"\btheoretical framework\b", r"\btheory\b.*\bused\b",
        r"\btheoretical basis\b", r"\bunderpinning\b",
    ]),
    ("literature_review", [
        r"\bliterature\b", r"\brelated work\b", r"\bprior research\b",
        r"\bprevious studies\b", r"\bexisting research\b",
    ]),
]


def detect_task(question: str) -> tuple:
    """Return (task_name, scores_dict) for a question using rule-based routing."""
    q = normalize_text(question).lower()
    scores: Dict[str, float] = {}
    for task_name, patterns in _TASK_DETECTION_RULES:
        hits = sum(1 for p in patterns if re.search(p, q, flags=re.IGNORECASE))
        if hits:
            scores[task_name] = float(hits)

    if scores:
        best = max(scores, key=lambda k: scores[k])
        return best, scores

    # Default to generic question answering
    return "question_answering", {"question_answering": 1.0}


def is_comparison_question(question: str) -> bool:
    """Return True if the question requests a comparison across papers."""
    q = normalize_text(question).lower()
    comparison_phrases = (
        "compare", "comparison", "differences between", "difference between",
        "similarities between", "similarities and differences", "across papers",
        "between papers", "between studies", "each paper", "both papers",
        "all papers", "all the papers", "the papers", "in the papers",
        "of the papers", "for each paper", "in each paper",
    )
    return any(phrase in q for phrase in comparison_phrases)
