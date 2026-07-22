"""api.py — FastAPI"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.coder import suggest_codes

app = FastAPI(title="Medical Coding Automation Tool")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class CodingRequest(BaseModel):
    clinical_note: str
    top_k: int = 8

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/suggest-codes")
def suggest(req: CodingRequest):
    if len(req.clinical_note.strip()) < 20:
        raise HTTPException(422, "Clinical note too short.")
    codes = suggest_codes(req.clinical_note, req.top_k)
    return {"suggestions": codes, "count": len(codes)}
