
from app.llm.google_gemini_llm import GoogleGeminiLLM
from langgraph.graph import StateGraph, END
from app.nodes.screening_node import ScreeningNode
from app.states.screening_state import ScreeningState
from app.nodes.airtable_node import AirtableNode

def build_screening_graph():
    llm = GoogleGeminiLLM().get_llm()
    screening_node = ScreeningNode(llm)
    airtable_node = AirtableNode()
    g = StateGraph(ScreeningState)

    g.add_node("analyse_jd",      screening_node.node_analyse_jd)
    g.add_node("score_resumes",   screening_node.node_score_resumes)
    g.add_node("draft_outreach",  screening_node.node_draft_outreach)
    g.add_node("save_candidates", airtable_node.save_candidates)

    g.set_entry_point("analyse_jd")
    g.add_edge("analyse_jd",     "score_resumes")
    g.add_edge("score_resumes",  "draft_outreach")
    g.add_edge("draft_outreach",  "save_candidates")
    g.add_edge("save_candidates", END)

    return g.compile()