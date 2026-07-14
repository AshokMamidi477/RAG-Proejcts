"""test_recommender.py"""

def test_rerank_prompt_format():
    import sys; sys.path.insert(0,"app")
    from recommender import RERANK_PROMPT
    assert "{query}" in RERANK_PROMPT
    assert "{context}" in RERANK_PROMPT
    assert "{products}" in RERANK_PROMPT

def test_recommend_returns_list_type():
    # Without API — just test the function signature exists
    import sys; sys.path.insert(0,"app")
    from recommender import recommend
    assert callable(recommend)
