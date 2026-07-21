"""
ingest.py — Build LlamaIndex FAISS index from legal documents

Stack:
------
- LlamaIndex
- HuggingFace Sentence Transformer Embeddings
- FAISS Vector Store

Embedding Model:
----------------
sentence-transformers/all-MiniLM-L6-v2

Embedding Dimension:
--------------------
384
"""

import os
import argparse
import faiss

from dotenv import load_dotenv

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings
)

from llama_index.core.node_parser import SentenceSplitter

from llama_index.vector_stores.faiss import FaissVectorStore

from llama_index.embeddings.huggingface import HuggingFaceEmbedding


load_dotenv()


# ---------------------------------------
# Configuration
# ---------------------------------------

INDEX_DIR = "vector_store/legal"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

EMBED_DIMENSION = 384



def build_index(source_dir: str):

    print("----------------------------------")
    print("Starting legal document ingestion...")
    print("----------------------------------")


    # Create storage directory

    os.makedirs(
        INDEX_DIR,
        exist_ok=True
    )


    # ---------------------------------------
    # Configure HuggingFace Embeddings
    # ---------------------------------------

    print(
        "Loading HuggingFace embedding model..."
    )


    embed_model = HuggingFaceEmbedding(
        model_name=
        "sentence-transformers/all-MiniLM-L6-v2"
    )


    Settings.embed_model = embed_model



    # ---------------------------------------
    # Configure Chunking
    # ---------------------------------------

    Settings.node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )



    # ---------------------------------------
    # Load Legal Documents
    # ---------------------------------------

    print(
        "Loading documents..."
    )


    documents = SimpleDirectoryReader(
        source_dir,
        recursive=True
    ).load_data()


    print(
        f"Loaded {len(documents)} documents"
    )



    # ---------------------------------------
    # Create FAISS Index
    #
    # MiniLM produces 384 dimensions
    # ---------------------------------------

    print(
        "Creating FAISS index (384 dimensions)..."
    )


    faiss_index = faiss.IndexFlatL2(
        EMBED_DIMENSION
    )


    vector_store = FaissVectorStore(
        faiss_index=faiss_index
    )



    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )



    # ---------------------------------------
    # Create LlamaIndex Vector Index
    # ---------------------------------------

    print(
        "Generating embeddings..."
    )


    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )



    # ---------------------------------------
    # Save LlamaIndex metadata
    # ---------------------------------------

    print(
        "Persisting LlamaIndex storage..."
    )


    index.storage_context.persist(
        persist_dir=INDEX_DIR
    )



    # ---------------------------------------
    # Save FAISS binary index
    # ---------------------------------------

    print(
        "Saving FAISS binary index..."
    )


    faiss.write_index(
        vector_store._faiss_index,
        f"{INDEX_DIR}/faiss.index"
    )


    print("----------------------------------")
    print("FAISS index saved successfully")
    print(f"Location: {INDEX_DIR}")
    print("----------------------------------")



if __name__ == "__main__":


    parser = argparse.ArgumentParser(
        description="Build legal FAISS index"
    )


    parser.add_argument(
        "--source",
        default="samples/case_law/",
        help="Directory containing legal documents"
    )


    args = parser.parse_args()


    build_index(
        args.source
    )