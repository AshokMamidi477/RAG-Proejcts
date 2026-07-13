"""
api.py — FastAPI SEC Filing Analyst Bot
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from app.retriever import build_chain


app = FastAPI(
    title="SEC Filing Analyst Bot"
)

# -----------------------------
# CORS
# -----------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# -----------------------------
# Session Chains
# -----------------------------

chains = {}

# -----------------------------
# Models
# -----------------------------

class ChatRequest(BaseModel):
    session_id: str
    question: str
    ticker_filter: Optional[str] = None

class Source(BaseModel):
    page: Optional[int] = None
    ticker: Optional[str] = None
    year: Optional[str] = None
    filing_type: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]

# -----------------------------
# Health
# -----------------------------

@app.get("/health")
def health():
    return {
        "status": "ok"
    }

# -----------------------------
# Chat
# -----------------------------

@app.post(
    "/chat",
    response_model=ChatResponse
)
def chat(req: ChatRequest):
    # Create conversation chain once per session
    if req.session_id not in chains:
        print(
            f"Creating new chain for session {req.session_id}"
        )

        chains[req.session_id] = build_chain(
            ticker_filter=req.ticker_filter
        )

    chain = chains[req.session_id]

    # New LangChain invocation
    print("Starting chain invoke...")
    try:
        response = chain.invoke(
            {
                "question": req.question
            }
        )

    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "sources": []
        }
    print("Chain completed")

    sources = []

    for doc in response.get(
        "source_documents",
        []
    ):

        sources.append(
            Source(
                page=doc.metadata.get(
                    "page"
                ),

                ticker=doc.metadata.get(
                    "ticker"
                ),

                year=doc.metadata.get(
                    "year"
                ),

                filing_type=doc.metadata.get(
                    "filing_type"
                )

            )
        )

    return ChatResponse(
        answer=response.get(
            "answer",
            ""
        ),
        sources=sources
    )

# -----------------------------
# Clear Session
# -----------------------------

@app.delete("/session/{session_id}")
def clear_session(session_id: str):

    chains.pop(
        session_id,
        None
    )


    return {
        "cleared": session_id
    }