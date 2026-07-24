"""kb_ingest.py — Ingest knowledge base into Pinecone"""

import os

from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Pinecone Key:", os.getenv("PINECONE_API_KEY"))

INDEX_NAME = os.getenv("PINECONE_INDEX", "support-kb")

# HuggingFace embedding model
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-small-en-v1.5"
)

# all-MiniLM-L6-v2 produces 384-dimensional embeddings
EMBEDDING_DIMENSION = 384


def ingest_kb(source: str = "samples/sample_knowledge_base.txt"):

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    existing_indexes = [index.name for index in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:

        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    print("Loading knowledge base...")

    docs = TextLoader(source).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )

    chunks = splitter.split_documents(docs)

    print(f"Created {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME
    )

    print(f"Ingested {len(chunks)} chunks into Pinecone")


if __name__ == "__main__":
    ingest_kb()