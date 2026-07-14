"""recommender.py — Two-stage semantic retrieval + Gemini LLM re-ranking"""

import json
import os

from pinecone import Pinecone
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from google import genai


load_dotenv()


# -----------------------------
# Configuration
# -----------------------------

INDEX_NAME = os.getenv(
    "PINECONE_INDEX",
    "product-catalogue"
)


GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)

# Gemini client
gemini_client = genai.Client(
    api_key=os.getenv(
        "GOOGLE_API_KEY"
    )
)


# HuggingFace embedding model

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)



# -----------------------------
# Re-ranking Prompt
# -----------------------------

RERANK_PROMPT = """
You are a smart product recommendation assistant.

Customer request:
"{query}"

Additional customer context:
"{context}"


From the products below, select the TOP 5 most relevant products.

Rank them from 1-5.

For each product provide:
- rank
- product_id
- product name
- one sentence explaining why it matches the customer's need


Products:

{products}


Return ONLY valid JSON:

[
 {{
   "rank":1,
   "product_id":"123",
   "name":"Product Name",
   "reason":"Why this product matches"
 }}
]
"""



# -----------------------------
# Generate Query Embedding
# -----------------------------

def get_query_embedding(
    text: str
) -> list[float]:

    return embedding_model.embed_query(
        text
    )



# -----------------------------
# Semantic Retrieval
# -----------------------------

def semantic_retrieve(
    query: str,
    top_k: int = 20,
    category_filter: str = None
) -> list[dict]:


    pc = Pinecone(
        api_key=os.getenv(
            "PINECONE_API_KEY"
        )
    )


    index = pc.Index(
        INDEX_NAME
    )


    query_vector = get_query_embedding(
        query
    )


    filter_dict = None

    if category_filter:
        filter_dict = {
            "category": {
                "$eq": category_filter
            }
        }



    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict
    )


    products = []


    for match in results.matches:

        products.append(

            {
                "product_id":
                    match.id,

                "score":
                    match.score,

                **match.metadata
            }

        )


    return products



# -----------------------------
# Gemini Re-ranking
# -----------------------------

def llm_rerank(
    query: str,
    context: str,
    candidates: list[dict]
) -> list[dict]:


    products_text = "\n".join(

        f"""
ID: {c['product_id']}
Name: {c.get('name')}
Price: ${c.get('price','?')}
Category: {c.get('category','')}
Description: {c.get('description','')}
Tags: {c.get('tags','')}
"""

        for c in candidates[:20]

    )


    prompt = RERANK_PROMPT.format(

        query=query,

        context=context,

        products=products_text

    )



    response = gemini_client.models.generate_content(

        model=GEMINI_MODEL,

        contents=prompt,

        config={
            "temperature": 0.2
        }

    )


    content = response.text.strip()


    # Remove markdown JSON wrapper if Gemini adds it

    if content.startswith("```"):

        content = (
            content
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )


    result = json.loads(
        content
    )


    if isinstance(result, list):

        return result


    return result.get(
        "recommendations",
        []
    )



# -----------------------------
# Complete Recommendation Flow
# -----------------------------

def recommend(
    query: str,
    context: str = "",
    category: str = None
) -> list[dict]:


    # Stage 1:
    # Vector similarity search

    candidates = semantic_retrieve(

        query,

        top_k=20,

        category_filter=category

    )


    if not candidates:

        return []



    # Stage 2:
    # Gemini intelligent ranking

    ranked = llm_rerank(

        query,

        context,

        candidates

    )



    # Attach Pinecone metadata

    metadata_map = {

        c["product_id"]: c

        for c in candidates

    }



    for item in ranked:

        product_id = item.get(
            "product_id"
        )


        item.update(

            metadata_map.get(

                product_id,

                {}

            )

        )


    return ranked