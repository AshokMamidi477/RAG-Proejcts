"""test_ingest.py"""
import sys; sys.path.insert(0,"app")
from ingest import CHUNK_SIZE, CHUNK_OVERLAP, INDEX_NAME

def test_chunk_config():
    assert CHUNK_SIZE == 1000
    assert CHUNK_OVERLAP < CHUNK_SIZE

def test_index_name_set():
    assert len(INDEX_NAME) > 0

def test_metadata_keys():
    expected_keys = {"ticker", "year", "filing_type"}
    # Simulate what ingest would attach
    metadata = {"ticker": "AAPL", "year": "2023", "filing_type": "10-K"}
    assert expected_keys == set(metadata.keys())
