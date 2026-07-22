"""coder.py — RAG medical coding chain using HuggingFace embeddings + Gemini"""

import json
import os

from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


# Pinecone index
INDEX_NAME = os.getenv("PINECONE_INDEX", "medicalcoding")


# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.1
)


# Hugging Face Sentence Transformer Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


CODING_PROMPT = """
You are a certified medical coder (CPC).

Using the ICD-10 reference below, suggest the most appropriate diagnostic
codes for this clinical note.

ICD-10 Reference:
{reference}

Clinical Note:
{note}


Return ONLY valid JSON.

Format:

[
 {{
   "code":"X00.0",
   "description":"Official ICD-10 description",
   "confidence":0.95,
   "reasoning":"One sentence explanation"
 }}
]


Rules:
- Only suggest codes supported by the clinical note
- Order suggestions by confidence
- Include complete ICD-10 code
- Use only codes available in the reference
- Never hallucinate codes
"""


def suggest_codes(clinical_note: str, top_k: int = 8) -> list[dict]:

    # Connect to Pinecone vector database
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings
    )


    # Retrieve relevant ICD-10 documents
    docs = vectorstore.similarity_search(
        clinical_note,
        k=top_k
    )


    reference = "\n".join(
        doc.page_content
        for doc in docs
    )


    prompt = CODING_PROMPT.format(
        reference=reference,
        note=clinical_note
    )


    # Gemini call
    response = llm.invoke(prompt)


    content = response.content


    # Remove markdown JSON formatting if Gemini returns it
    content = content.replace("```json", "")
    content = content.replace("```", "")
    content = content.strip()


    result = json.loads(content)


    return (
        result
        if isinstance(result, list)
        else result.get("suggestions", [])
    )