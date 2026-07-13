"""test_retriever.py — unit tests (no API needed)"""

def test_chunk_size_constant():
    import sys; sys.path.insert(0,"app")
    from ingest import CHUNK_SIZE, CHUNK_OVERLAP
    assert CHUNK_SIZE > CHUNK_OVERLAP, "Chunk size must be greater than overlap"
    assert CHUNK_OVERLAP > 0

def test_index_path_defined():
    import sys; sys.path.insert(0,"app")
    from ingest import INDEX_PATH
    assert "vector_store" in INDEX_PATH

def test_prompt_template_has_variables():
    import sys; sys.path.insert(0,"app")
    from retriever import PROMPT_TEMPLATE
    assert "{context}" in PROMPT_TEMPLATE
    assert "{question}" in PROMPT_TEMPLATE
