"""
agent.py — LangGraph AI Customer Support Agent

Architecture:

User
 |
FastAPI
 |
LangGraph Agent
 |
+----------------+
| Intent Router  |
+----------------+
        |
+----------------+
| Pinecone RAG   |
+----------------+
        |
+----------------+
| Groq LLM       |
+----------------+
        |
Ticket Escalation


Memory:

Short Term:
LangGraph Checkpointer

Long Term:
Redis/PostgreSQL (future)
"""

import os
from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()


# =====================================================
# Groq LLM Configuration
# =====================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    groq_api_key=os.getenv("GROQ_API_KEY")
)



# =====================================================
# Pinecone Configuration
# =====================================================

INDEX_NAME = os.getenv(
    "PINECONE_INDEX",
    "support-kb"
)



# =====================================================
# HuggingFace Embedding Model
# MUST MATCH ingestion model
# =====================================================

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)


embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)



# =====================================================
# LangGraph State
# =====================================================

class AgentState(TypedDict):
    session_id: str
    user_message: str
    intent: str
    kb_context: str
    response: str
    messages: Annotated[list, add_messages]
    escalated: bool



# =====================================================
# Intent Classification Node
# =====================================================

def classify_intent(state: AgentState):
    message = state["user_message"].lower()
    escalation_words = [
        "speak to human",
        "human agent",
        "representative",
        "manager",
        "complaint",
        "escalate"
    ]


    product_words = [
        "how",
        "what",
        "why",
        "price",
        "cost",
        "feature",
        "plan",
        "billing",
        "cancel",
        "integration"
    ]


    if any(word in message for word in escalation_words):

        return {
            "intent": "escalation"
        }


    if any(word in message for word in product_words):

        return {
            "intent": "product_question"
        }


    return {
        "intent": "general"
    }




# =====================================================
# Pinecone Retrieval Node
# =====================================================

def retrieve_kb(state: AgentState):


    if state["intent"] != "product_question":

        return {
            "kb_context": ""
        }



    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings
    )



    documents = vectorstore.similarity_search(
        state["user_message"],
        k=4
    )



    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )



    return {

        "kb_context": context

    }




# =====================================================
# Groq Response Generation Node
# =====================================================

def generate_response(state: AgentState):


    conversation_history = "\n".join(

        f"{m.type}: {m.content}"

        for m in state.get("messages", [])[-6:]

    )


    if state["intent"] == "escalation":

        return {

            "response":
            "I understand you need human assistance. "
            "I will create a support ticket for our team.",

            "escalated": True

        }



    context = ""


    if state["kb_context"]:

        context = f"""

Knowledge Base:

{state["kb_context"]}

"""



    prompt = f"""

You are an intelligent customer support AI assistant.


{context}


Conversation History:

{conversation_history}



Customer Question:

{state["user_message"]}



Instructions:

1. Answer professionally.
2. Use knowledge base information.
3. Do not hallucinate.
4. If information is unavailable, recommend escalation.

"""

    result = llm.invoke(prompt)
    return {
        "response": result.content,
        "messages": [
            {
                "role": "assistant",
                "content": result.content
            }
        ],

        "escalated": False

    }


# =====================================================
# Escalation Router
# =====================================================

def should_escalate(state: AgentState):
    if state.get("escalated"):
        return "escalate"
    return "done"


# =====================================================
# Ticket Creation
# =====================================================

def handle_escalation(state: AgentState):

    from app.ticket_logger import log_ticket

    ticket_id = log_ticket(
        state["session_id"],
        state["user_message"],
        state.get("messages", [])
    )

    return {
        "response":
        state["response"]
        +
        f"\n\nSupport Ticket ID: {ticket_id}"
    }





# =====================================================
# Build LangGraph Agent
# =====================================================

def build_support_agent():
    graph = StateGraph(AgentState)
    graph.add_node(
        "classify",
        classify_intent
    )

    graph.add_node(
        "retrieve",
        retrieve_kb
    )

    graph.add_node(
        "respond",
        generate_response
    )

    graph.add_node(
        "escalate",
        handle_escalation
    )

    graph.set_entry_point(
        "classify"
    )

    graph.add_edge(
        "classify",
        "retrieve"
    )

    graph.add_edge(
        "retrieve",
        "respond"
    )

    graph.add_conditional_edges(
        "respond",
        should_escalate,
        {
            "escalate": "escalate",
            "done": END
        }
    )


    graph.add_edge(
        "escalate",
        END
    )


    # LangGraph short-term memory
    memory = MemorySaver()

    return graph.compile(
        checkpointer=memory
    )