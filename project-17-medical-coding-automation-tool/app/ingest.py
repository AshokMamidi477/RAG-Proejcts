"""ingest.py — Ingest ICD-10 reference corpus into Pinecone"""

import os

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone, ServerlessSpec


load_dotenv()


INDEX_NAME = os.getenv(
    "PINECONE_INDEX",
    "medicalcoding"
)


# Hugging Face embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def ingest_icd10(
    source: str = "samples/icd10_reference.txt"
):

    pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY")
    )


    # Create Pinecone index if not exists
    existing_indexes = [
        index.name
        for index in pc.list_indexes()
    ]


    if INDEX_NAME not in existing_indexes:

        pc.create_index(
            name=INDEX_NAME,
            dimension=384,   # all-MiniLM-L6-v2 dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        print(
            f"Created Pinecone index: {INDEX_NAME}"
        )

    else:
        print(
            f"Pinecone index already exists: {INDEX_NAME}"
        )


    # Load ICD-10 reference file
    loader = TextLoader(
        source,
        encoding="utf-8"
    )

    docs = loader.load()


    # Split into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=[
            "\n\n",
            "\n",
            ". "
        ]
    )


    chunks = splitter.split_documents(
        docs
    )


    # Hugging Face embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


    # Store vectors in Pinecone
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME
    )


    print(
        f"Ingested {len(chunks)} ICD-10 reference chunks into {INDEX_NAME}"
    )



if __name__ == "__main__":
    ingest_icd10()