"""catalogue_ingest.py — Embed and upsert product catalogue to Pinecone"""

import os
import argparse
import pandas as pd

from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# -----------------------------
# Configuration
# -----------------------------

INDEX_NAME = os.getenv(
    "PINECONE_INDEX",
    "product-catalogue"
)

BATCH_SIZE = 50

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------
# Load Embedding Model
# -----------------------------

def load_embeddings():
    print(
        "Loading HuggingFace embeddings..."
    )
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    return embeddings



# -----------------------------
# Generate Embeddings
# -----------------------------

def embed_texts(embeddings, texts: list[str]):
    return embeddings.embed_documents( texts )

# -----------------------------
# Initialize Pinecone
# -----------------------------

def init_pinecone():
    pc = Pinecone(
        api_key=os.getenv(
            "PINECONE_API_KEY"
        )
    )

    existing_indexes = [
        index.name
        for index in pc.list_indexes()
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
        print(f"Created index {INDEX_NAME}")

    else:
        print(
            f"Using existing index {INDEX_NAME}"
        )

    return pc



# -----------------------------
# Ingest Catalogue
# -----------------------------

def ingest_catalogue(
    csv_path: str
):
    init_pinecone()
    pc = Pinecone(
        api_key=os.getenv(
            "PINECONE_API_KEY"
        )
    )

    index = pc.Index(
        INDEX_NAME
    )

    embeddings = load_embeddings()

    df = pd.read_csv(
        csv_path
    )


    # Combine product information
    # into embedding text

    df["embed_text"] = (
        df["name"].fillna("")
        + ". "
        + df["category"].fillna("")
        + ". "
        + df["description"].fillna("")
        + ". Tags: "
        + df["tags"].fillna("")
    )


    for start in range(
        0,
        len(df),
        BATCH_SIZE
    ):

        batch = df.iloc[
            start:start+BATCH_SIZE
        ]
        texts = batch[
            "embed_text"
        ].tolist()

        vectors_embeddings = embed_texts(
            embeddings,
            texts
        )

        vectors = []

        for i, (_, row) in enumerate(
            batch.iterrows()
        ):

            vectors.append({

                "id":
                    str(
                        row["product_id"]
                    ),


                "values":
                    vectors_embeddings[i],


                "metadata": {

                    "name":
                        row["name"],


                    "category":
                        row["category"],


                    "price":
                        float(
                            row["price"]
                        ),


                    "rating":
                        float(
                            row.get(
                                "rating",
                                4.0
                            )
                        ),


                    "description":
                        row["description"][:500],


                    "tags":
                        row.get(
                            "tags",
                            ""
                        )
                }

            })


        index.upsert(
            vectors=vectors
        )


        print(
            f"Upserted batch "
            f"{start}-{start+len(batch)}"
        )


    print(
        f"Completed ingestion: {len(df)} products"
    )



# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--csv",
        default="samples/sample_catalogue.csv"
    )


    args = parser.parse_args()


    ingest_catalogue(
        args.csv
    )