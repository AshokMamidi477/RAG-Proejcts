from typing import TypedDict, Optional
from pydantic import BaseModel

class JDCriteria(BaseModel):
    required_skills: list[str]
    nice_to_have: list[str]
    min_years_experience: Optional[int]
    role_title: str
    key_responsibilities: list[str]