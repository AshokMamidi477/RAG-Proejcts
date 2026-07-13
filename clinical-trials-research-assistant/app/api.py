"""api.py — FastAPI endpoints"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.retriever import build_qa_chain

app = FastAPI(title="Clinical Trials Research Assistant")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

qa_chain = None

@app.on_event("startup")
def load_chain():
    global qa_chain
    try:
        qa_chain = build_qa_chain()
    except Exception as e:
        print(f"Warning: could not load QA chain: {e}")

class QueryRequest(BaseModel):
    question: str
    k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

@app.get("/health")
def health():
    return {"status": "ok", "index_loaded": qa_chain is not None}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not qa_chain:
        raise HTTPException(503, "Vector index not loaded. Run ingest.py first.")
    result   = qa_chain.invoke({"query": req.question})
    sources  = list({
        doc.metadata.get("source", "Unknown") + f" p.{doc.metadata.get('page', '?')}"
        for doc in result.get("source_documents", [])
    })
    return QueryResponse(answer=result["result"], sources=sources)
