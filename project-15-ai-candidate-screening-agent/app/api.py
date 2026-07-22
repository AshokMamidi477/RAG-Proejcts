"""api.py — FastAPI trigger endpoint"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.graphs.graph_builder import build_screening_graph

app = FastAPI(title="AI Candidate Screening Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
graph = build_screening_graph()


class ScreeningRequest(BaseModel):
    jd_text: str
    company: str = "Our Company"
    resumes: list[dict]   # [{"name": str, "text": str}]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/screen")
def screen_candidates(req: ScreeningRequest):
    if not req.jd_text.strip():
        raise HTTPException(422, "Job description cannot be empty.")
    if not req.resumes:
        raise HTTPException(422, "At least one resume is required.")

    result = graph.invoke({
        "jd_text":  req.jd_text,
        "company":  req.company,
        "resumes":  req.resumes,
        "scores":   [],
        "outreach_drafts": [],
        "top_candidates": [],
    })

    return {
        "role":   result["criteria"].role_title,
        "scores": [
            {"name": s.name, "score": s.score, "reasoning": s.reasoning,
             "recommend": s.recommend_outreach}
            for s in result["scores"]
        ],
        "outreach_drafts": result["outreach_drafts"],
    }
