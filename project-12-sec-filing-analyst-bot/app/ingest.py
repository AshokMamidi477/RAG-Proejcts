"""ingest.py — Ingest SEC filings into Pinecone with metadata"""

import os
import argparse
from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader,
    PyMuPDFLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv


load_dotenv()


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

INDEX_NAME = os.getenv(
    "PINECONE_INDEX",
    "filinganalystbot"
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def init_pinecone():
    pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY")
    )

    existing_indexes = [
        i.name
        for i in pc.list_indexes()
    ]

    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        print(f"Created Pinecone index: {INDEX_NAME}")
    else:
        print(f"Using existing index: {INDEX_NAME}")
    return pc


def ingest_filing(
    source_dir: str,
    ticker: str,
    year: str,
    filing_type: str
):

    init_pinecone()

    docs = []

    for path in Path(source_dir).rglob("*"):
        if path.suffix.lower() == ".pdf":
            docs.extend(
                PyMuPDFLoader(
                    str(path)
                ).load()
            )

        elif path.suffix.lower() == ".txt":
            docs.extend(
                TextLoader(
                    str(path)
                ).load()
            )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )


    chunks = splitter.split_documents(docs)


    for chunk in chunks:
        chunk.metadata.update(
            {
                "ticker": ticker.upper(),
                "year": year,
                "filing_type": filing_type
            }
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME
    )

    print(
        f"Ingested {len(chunks)} chunks "
        f"for {ticker} {year} {filing_type}"
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--source", required=True)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--year", required=True)
    parser.add_argument("--type", default="10-K")

    args = parser.parse_args()

    ingest_filing(
        args.source,
        args.ticker,
        args.year,
        args.type
    )