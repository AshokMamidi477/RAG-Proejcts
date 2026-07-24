"""test_agent.py"""
import sys; sys.path.insert(0,"app")
from agent import classify_intent, AgentState

def make_state(msg):
    return AgentState(session_id="test",user_message=msg,intent="",kb_context="",
                      response="",history=[],escalated=False)

def test_escalation_intent():
    r = classify_intent(make_state("I want to speak to a human agent"))
    assert r["intent"] == "escalation"

def test_product_question_intent():
    r = classify_intent(make_state("How much does the Professional plan cost?"))
    assert r["intent"] == "product_question"

def test_general_intent():
    r = classify_intent(make_state("Hello there"))
    assert r["intent"] == "general"
