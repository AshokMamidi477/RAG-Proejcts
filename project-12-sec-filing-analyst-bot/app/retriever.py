"""
retriever.py

Loads Pinecone vector store and creates
a Conversational Retrieval QA chain.

Embedding:
    HuggingFace all-MiniLM-L6-v2

LLM:
    Google Gemini

Vector DB:
    Pinecone
"""

import os

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain_core.prompts import PromptTemplate


load_dotenv()


# -----------------------------
# Configuration
# -----------------------------

INDEX_NAME = os.getenv(
    "PINECONE_INDEX",
    "filinganalystbot"
)


EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)



# -----------------------------
# Load Embeddings
# -----------------------------

def load_embeddings():

    print("Loading HuggingFace embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    return embeddings



# -----------------------------
# Load Pinecone Vector Store
# -----------------------------

def load_vectorstore():

    print("Connecting to Pinecone...")


    embeddings = load_embeddings()


    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings
    )


    return vectorstore



# -----------------------------
# Load Gemini
# -----------------------------

def load_llm():

    print("Loading Gemini LLM...", GEMINI_MODEL)


    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.2,
        google_api_key=os.getenv(
            "GOOGLE_API_KEY"
        )
    )


    return llm



# -----------------------------
# Build RAG Chain
# -----------------------------

def build_chain(
    ticker_filter=None,
    year_filter=None,
    filing_type_filter=None
):

    vectorstore = load_vectorstore()


    llm = load_llm()


    # -----------------------------
    # Pinecone Metadata Filters
    # -----------------------------

    pinecone_filter = {}


    if ticker_filter:

        pinecone_filter["ticker"] = {
            "$eq": ticker_filter.upper()
        }


    if year_filter:

        pinecone_filter["year"] = {
            "$eq": str(year_filter)
        }


    if filing_type_filter:

        pinecone_filter["filing_type"] = {
            "$eq": filing_type_filter
        }



    print(
        "Pinecone filter:",
        pinecone_filter
    )



    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 3,
            "filter": pinecone_filter
        }
    )



    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )



    prompt_template = """

You are an SEC filing analyst assistant.

Answer the user's question using only the SEC filing context.

If the answer is not present in the context,
say:

"I could not find this information in the SEC filing."


SEC Filing Context:

{context}


Question:

{question}


Answer:

"""


    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=[
            "context",
            "question"
        ]
    )



    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": prompt
        },
        return_source_documents=True
    )


    return chain