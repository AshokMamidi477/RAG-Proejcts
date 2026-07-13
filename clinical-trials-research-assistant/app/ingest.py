"""
ingest.py
Ingests PDF or TXT files, chunks them, embeds with Gemini, and saves a FAISS index.
Usage: python app/ingest.py --source samples/sample_trials/
"""
import os, argparse
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import pytesseract
from pdf2image import convert_from_path
from dotenv import load_dotenv
from PIL import Image
load_dotenv()

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200
INDEX_PATH    = "vector_store/clinical_trials"


def load_documents(source_dir: str) -> list:
    docs = []

    for path in Path(source_dir).rglob("*"):

        if path.suffix.lower() == ".pdf":
            pdf_docs = PyMuPDFLoader(str(path)).load()

            extracted_text = "".join(
                doc.page_content for doc in pdf_docs
            ).strip()
            if extracted_text:
                print(f"Text PDF: {path.name}")
                for doc in pdf_docs:
                    doc.metadata.update({
                        "source": str(path),
                        "document_type": "text_pdf",
                        "ocr": False
                    })
                docs.extend(pdf_docs)
            else:
                print(f"Scanned PDF detected: {path.name}")
                ocr_docs = load_scanned_pdf(str(path))
                docs.extend(ocr_docs)

        elif path.suffix.lower() == ".txt":
            txt_docs = TextLoader(str(path)).load()
            for doc in txt_docs:
                doc.metadata.update({
                    "source": str(path),
                    "document_type": "text_txt",
                    "ocr": False
                })
            docs.extend(txt_docs)

        elif path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            print(f"Image detected: {path.name}")

            text = pytesseract.image_to_string(Image.open(path))

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(path),
                        "document_type": "image",
                        "ocr": True,
                    },
                )
            )

    print(f"Loaded {len(docs)} document pages from {source_dir}")
    return docs

def load_scanned_pdf(pdf_path: str) -> list:
    """
    Convert scanned PDF pages to images and extract text using OCR.
    """

    print(f"Performing OCR on scanned PDF: {pdf_path}")

    documents = []
    pages = convert_from_path(
        pdf_path,
        dpi=300
    )

    for page_number, page_image in enumerate(pages):
        text = pytesseract.image_to_string(
        page_image,
        lang="eng"
    )

        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                    "source": pdf_path,
                    "page": page_number + 1,
                    "document_type": "clinical_scan",
                    "ocr": True
                    }
                )
            )
    print(f"OCR completed. Extracted {len(documents)} pages from {pdf_path}")
    return documents

def chunk_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def build_index(chunks: list) -> None:
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    vectorstore.save_local(INDEX_PATH)
    print(f"FAISS index saved to {INDEX_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="samples/sample_trials/", help="Directory of PDFs/TXTs")
    args = parser.parse_args()
    docs   = load_documents(args.source)


for doc in docs:
    print("\nSOURCE:", doc.metadata)
    print(doc.page_content[:1000])

    # chunks = chunk_documents(docs)
    # build_index(chunks)
