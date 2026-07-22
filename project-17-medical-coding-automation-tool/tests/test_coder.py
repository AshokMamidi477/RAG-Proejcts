"""test_coder.py"""

def test_coding_prompt_has_placeholders():
    import sys; sys.path.insert(0,"app")
    from coder import CODING_PROMPT
    assert "{reference}" in CODING_PROMPT
    assert "{note}" in CODING_PROMPT

def test_code_structure():
    sample = {"code":"I10","description":"Essential hypertension","confidence":0.9,"reasoning":"Hypertension mentioned."}
    assert "code" in sample
    assert 0 <= sample["confidence"] <= 1
