"""api.py — FastAPI with session management"""
import uuid
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.agent import build_support_agent, AgentState

app   = FastAPI(title="Customer Support Agent")
agent = build_support_agent()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
sessions: dict[str, list] = {}   # session_id -> conversation history

class ChatRequest(BaseModel):
    session_id: str = ""
    message: str

@app.get("/health")
def health(): return {"status":"ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    history    = sessions.get(session_id, [])
    history.append({"role":"user","content":req.message})

    result = agent.invoke({
        "session_id":   session_id,
        "user_message": req.message,
        "intent":       "",
        "kb_context":   "",
        "response":     "",
        "history":      history,
        "escalated":    False,
    })
    response = result["response"]
    history.append({"role":"assistant","content":response})
    sessions[session_id] = history

    return {"session_id":session_id,"response":response,
            "escalated":result.get("escalated",False)}
