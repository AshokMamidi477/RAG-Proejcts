"""
query_engine.py

Legal RAG Query Engine

Stack:
------
- LlamaIndex
- Gemini 2.0 Flash
- HuggingFace Embeddings
- FAISS Vector Store
"""

import os
import faiss

from dotenv import load_dotenv

from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
    PromptTemplate
)

from llama_index.vector_stores.faiss import FaissVectorStore

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.llms.gemini import Gemini


load_dotenv()


INDEX_DIR = "vector_store/legal"


LEGAL_PROMPT = """

You are a legal research assistant.

Answer ONLY using the provided legal context.

Rules:
- Do not invent cases.
- Do not invent statutes.
- Do not provide legal advice.
- If information is not found, say:
"No relevant precedent found in the available legal corpus."


Context:

{context_str}


Question:

{query_str}


Answer:

"""



def build_query_engine(similarity_top_k: int = 5):

    print("Loading HuggingFace embeddings...")


    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    print("Loading Gemini...")


    Settings.llm = Gemini(
        model="models/gemini-2.0-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.0
    )


    print("Loading FAISS index...")


    # Load FAISS binary index
    faiss_index = faiss.read_index(
        f"{INDEX_DIR}/faiss.index"
    )


    vector_store = FaissVectorStore(
        faiss_index=faiss_index
    )


    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=INDEX_DIR
    )


    print("Loading LlamaIndex storage...")


    index = load_index_from_storage(
        storage_context
    )


    print("Creating query engine...")


    query_engine = index.as_query_engine(
        similarity_top_k=similarity_top_k
    )


    query_engine.update_prompts(
        {
            "response_synthesizer:text_qa_template":
            PromptTemplate(
                LEGAL_PROMPT
            )
        }
    )


    print("✅ Legal RAG engine loaded")


    return query_engine