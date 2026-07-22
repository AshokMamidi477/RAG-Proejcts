
from typing import TypedDict 
from pydantic import BaseModel

class CandidateScore(BaseModel):
    name: str
    score: int          # 0-100
    matched_skills: list[str]
    missing_skills: list[str]
    reasoning: str
    recommend_outreach: bool