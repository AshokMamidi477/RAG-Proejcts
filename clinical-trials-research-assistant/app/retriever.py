"""
retriever.py
Loads the FAISS index and builds a LangChain RetrievalQA chain using Gemini.

The embedding model MUST match the one used during ingestion.
"""

import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv


load_dotenv()


INDEX_PATH = "vector_store/clinical_trials"


PROMPT_TEMPLATE = """
You are a clinical research assistant.

Answer the question using ONLY the context provided below.

If the answer is not present in the context, say:
"I could not find relevant information in the loaded trials."

Cite the source document name at the end of your answer.

Context:
{context}

Question:
{question}

Answer:
"""


def build_qa_chain(k: int = 5) -> RetrievalQA:

    # IMPORTANT:
    # This MUST be the same embedding model used in ingest.py
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    # Load FAISS Vector Store
    vectorstore = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


    # Retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k
        }
    )


    # Prompt
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=[
            "context",
            "question"
        ]
    )


    # Gemini for answer generation only
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )


    # RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt
        }
    )


    return qa_chain