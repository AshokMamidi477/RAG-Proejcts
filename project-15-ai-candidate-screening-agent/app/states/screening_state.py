from typing import TypedDict, Annotated
import operator
from pydantic import BaseModel
from app.states.jd_criteria import JDCriteria

class ScreeningState(TypedDict):
    jd_text: str
    company: str
    resumes: list[dict]           # [{"name": str, "text": str}]
    criteria: JDCriteria
    scores: Annotated[list, operator.add]
    outreach_drafts: Annotated[list, operator.add]
    top_candidates: list
