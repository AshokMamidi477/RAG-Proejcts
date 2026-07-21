"""
api.py — FastAPI endpoints for Legal Precedent Finder

Stack:
------
- FastAPI
- LlamaIndex
- Gemini 2.0 Flash
- HuggingFace Embeddings
- FAISS Vector Store
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from app.query_engine import build_query_engine



# ---------------------------------------
# FastAPI App
# ---------------------------------------

app = FastAPI(
    title="Legal Precedent Finder",
    description="Legal RAG system using LlamaIndex + FAISS + Gemini",
    version="1.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



# ---------------------------------------
# Global Query Engine
# ---------------------------------------

engine = None



# ---------------------------------------
# Startup - Load RAG Engine
# ---------------------------------------

@app.on_event("startup")
def load_engine():

    global engine

    try:

        engine = build_query_engine()

        print(
            "✅ Legal RAG engine loaded"
        )


    except Exception as e:

        print(
            "❌ Could not load index:"
        )

        print(e)

        engine = None



# ---------------------------------------
# Request Models
# ---------------------------------------

class LegalQuery(BaseModel):

    question: str



# ---------------------------------------
# Health Check
# ---------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok",
        "index_loaded": engine is not None
    }



# ---------------------------------------
# Search Endpoint
# ---------------------------------------

@app.post("/search")
def legal_search(
    req: LegalQuery
):

    if engine is None:

        raise HTTPException(
            status_code=503,
            detail="Legal RAG engine not loaded"
        )


    try:

        print("--------------------------------")
        print("QUESTION:")
        print(req.question)
        print("--------------------------------")


        response = engine.query(
            req.question
        )


        print("--------------------------------")
        print("ANSWER:")
        print(response)
        print("--------------------------------")


        sources = []


        for node in response.source_nodes:

            sources.append(
                {
                    "text": node.node.get_content()[:300],
                    "score": node.score
                }
            )


        return {

            "answer": str(response),

            "sources": sources

        }



    except Exception as e:


        print("--------------------------------")
        print("QUERY ERROR")
        print(type(e).__name__)
        print(str(e))
        print("--------------------------------")


        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
        