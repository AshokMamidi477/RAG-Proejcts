# 🛒 Project 13 — Intelligent Product Recommendation Engine

![Level](https://img.shields.io/badge/Level-Intermediate-blue)
![Industry](https://img.shields.io/badge/Industry-E--commerce-green)
![Stack](https://img.shields.io/badge/Stack-LangChain%20%7C%20Pinecone%20%7C%20Gemini%20%7C%20FastAPI-orange)

An AI-powered product recommendation engine that understands customer intent using embeddings, semantic search, and Large Language Models.

The system uses a **two-stage retrieval architecture**:

1. **Semantic Retrieval**
   * Convert product catalogue into HuggingFace embeddings.
   * Store vectors in Pinecone.
   * Retrieve top relevant products using similarity search.

2. **LLM Re-ranking**
   * Send retrieved candidates to Gemini.
   * Gemini analyzes customer intent and ranks the best products.
   * Generates personalized recommendation explanations.

---

# 🧩 Business Problem

Traditional e-commerce search relies on keyword matching. 

**Example Customer Query:**
> "I need waterproof shoes for hiking under $100"

Standard keyword search may incorrectly return:
* Waterproof phone cases
* Hiking books
* Rain jackets

This system understands the actual intent:
* **Category:** Outdoor Footwear
* **Features:** Waterproof, Durable, Hiking suitable
* **Constraint:** Price < $100

This project solves this problem using semantic AI search.

---

# 🎯 Project Objective

Build an intelligent recommendation engine that:

✅ Embeds product catalogue using HuggingFace models  
✅ Stores embeddings in Pinecone Vector Database  
✅ Performs semantic similarity search  
✅ Retrieves top-20 candidate products  
✅ Uses Gemini LLM for intelligent re-ranking  
✅ Returns top-5 personalized recommendations  
✅ Provides reasoning behind recommendations  
✅ Exposes FastAPI APIs  
✅ Provides Streamlit user interface  

---

# 🏗 System Architecture

```
                 User Query
         "waterproof hiking shoes"
                     |
                     v
        HuggingFace Embedding Model
         (Sentence Transformer, 384d)
                     |
                     v
             Pinecone Vector DB
       (Product Embeddings + Metadata)
                     |
                     v
         Semantic Similarity Search
             (Retrieve Top 20)
                     |
                     v
                 Gemini LLM
        (Contextual Product Ranking)
                     |
                     v
           Top 5 Recommendations
               + Explanation
                     |
                     v
                FastAPI API
                     |
                     v
                Streamlit UI
```

---

# 🛠 Technology Stack

| Layer | Technology |
|---|---|
| **LLM** | Google Gemini |
| **Embeddings** | HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`) |
| **Embedding Dimension** | 384 |
| **Vector Database** | Pinecone |
| **Framework** | LangChain |
| **API** | FastAPI |
| **Server** | Uvicorn |
| **Frontend** | Streamlit |
| **Data Processing** | Pandas |
| **Language** | Python 3.10+ |

---

# 🤖 AI Models Used

## LLM - Gemini
Used for product comparison, context understanding, recommendation ranking, and explanation generation.

**Example Input:**
> "I need a lightweight laptop for programming"

**Gemini Analysis & Output:**
```json
[
 {
  "rank": 1,
  "product": "Developer Pro Laptop",
  "reason": "Lightweight laptop with strong CPU suitable for programming"
 }
]
```

## Embedding Model - HuggingFace
The embedding model converts text descriptions into numerical vectors representing semantic meaning.

**Example Input:**
> "Waterproof hiking boots with strong outdoor grip"

**Output:**
```
[0.023, -0.142, 0.321, ... 384 dimensions]
```

### Why 384 dimensions?
* Lower storage requirement
* Faster similarity search
* Reduced latency
* High semantic accuracy for product search

---

# 📁 Project Structure

```
project-13-intelligent-product-recommendation-engine/
├── app/
│   ├── catalogue_ingest.py    # Generates embeddings & uploads products to Pinecone
│   ├── recommender.py         # Handles semantic retrieval & Gemini reranking
│   ├── api.py                 # FastAPI endpoints
│   └── ui.py                  # Streamlit user interface
├── samples/
│   └── sample_catalogue.csv   # Mock product data
├── tests/
│   └── test_recommender.py    # Recommendation pipeline tests
├── .env.example
├── README.md
└── requirements.txt
```

---

# ⚙️ Setup

## 1. Create Virtual Environment

**Mac/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

## 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## 3. Environment Variables
Create a `.env` file in the root directory and add your credentials:
```env
GEMINI_API_KEY=<your-gemini-key>
PINECONE_API_KEY=<your-pinecone-key>
PINECONE_INDEX=product-catalogue
HF_TOKEN=<optional-huggingface-token>
```

---

# 📦 Dependency Requirements (`requirements.txt`)

```text
# LangChain Core
langchain>=0.2.0
langchain-community>=0.2.0

# Gemini LLM
langchain-google-genai>=2.0.0
google-generativeai>=0.8.0

# HuggingFace Embeddings
langchain-huggingface>=0.1.0
sentence-transformers>=3.0.0

# Vector Database
pinecone-client>=3.0.0
langchain-pinecone>=0.1.0

# API
fastapi>=0.110.0
uvicorn>=0.29.0

# Frontend
streamlit>=1.35.0

# Data Processing
pandas>=2.0.0

# Validation
pydantic>=2.0.0

# Environment Variables
python-dotenv>=1.0.0

# Testing
pytest>=8.0.0
```

---

# 🚀 Running The Application

### Step 1 — Load Product Catalogue
```bash
python app/catalogue_ingest.py --csv samples/sample_catalogue.csv
```
*Expected Output:*
```text
Loading HuggingFace embeddings...
Embedding dimension: 384
Upserted batch 0-50
Ingestion complete
```

### Step 2 — Start FastAPI Backend
```bash
uvicorn app.api:app --reload --port 8000
```
* Local API Docs: `http://127.0.0.1:8000/docs`
* Health Check: `GET /health`

### Step 3 — Run Streamlit UI
```bash
streamlit run app/ui.py
```

---

# 🔍 Example Queries

* **Query 1:** `waterproof hiking shoes under $100`
  * *Expected Focus:* Outdoor footwear, waterproof boots, trail gear meeting price constraints.
* **Query 2:** `gift for someone who loves cooking`
  * *Expected Focus:* Kitchen utilities, cooking accessories, high-tier cookware.

---

# 🧠 Why Two-Stage Retrieval?

A standalone LLM cannot evaluate thousands or millions of inventory rows efficiently or cost-effectively. 

```
[Stage 1: Fast Retrieval]
Query ──> Embedding ──> Pinecone Vector Search ──> Top 20 Candidates

[Stage 2: Intelligent Ranking]
Top 20 Candidates ──> Gemini LLM Reasoning ──> Top 5 Personalized Recommendations + Explanations
```

### Key Benefits
* ⚡ **Performance:** Highly scalable, fast search execution.
* 💰 **Cost-Efficiency:** Minimal token overhead hitting the LLM.
* 🎯 **Precision:** Combines lightning-fast vector parsing with nuanced LLM reasoning.

---

# 🔄 Difference From Document RAG Projects

| Feature | Project 12: SEC Filing Analyst Bot | Clinical Research Assistant | Project 13: Product Recommendation Engine |
|---|---|---|---|
| **Primary Purpose** | Answer questions from financial reports | Parse & query clinical trials | Recommend ideal inventory options |
| **Pipeline Core** | PDF ➔ Chunking ➔ Embedding ➔ Vector Search ➔ Generation | Documents ➔ OCR ➔ Embedding ➔ Vector Search ➔ Summary | Catalogue ➔ Embedding ➔ Vector Search ➔ **Gemini Re-ranking** |
| **RAG Blueprint Type**| Knowledge Retrieval RAG | Research / Synthesis RAG | **Decision / Recommendation RAG** |

---

# 🎤 Interview Questions

### 1. Explain your architecture.
> **Answer:** I implemented a two-stage recommendation system. Product specifications and metadata are stored as 384-dimensional HuggingFace embeddings within Pinecone. User inputs prompt a semantic proximity search to extract the top 20 candidate matches. Those entries are then funneled into Gemini, which applies analytical filters based on constraints, context, and intent to yield the top 5 results complete with logical reasoning.

### 2. Why use HuggingFace embeddings over premium cloud endpoints?
> **Answer:** Local HuggingFace inference ensures total cost predictability and zero rate-limiting constraints during baseline vector construction. For retail matching, an optimized model like `all-MiniLM-L6-v2` balances performance with highly accurate cluster groupings.

### 3. Why not directly feed the catalogue to Gemini?
> **Answer:** Processing millions of database items directly via an LLM context window causes astronomical latency spikes and massive operational API expenses. Vector filtering scales gracefully; LLMs are best reserved for structural reasoning over compact candidate pools.

### 4. Why stick with 384 dimensions?
> **Answer:** Higher vector dimension models yield larger memory footprints and compute footprints. 384 dimensions provide the optimal structural midpoint for consumer catalog indexing, retaining contextual integrity while running efficiently.

### 5. How would you scale this pipeline to millions of inventory items?
> **Answer:** I would incorporate:
> * Decoupled, asynchronous batch embedding generation (e.g., using Celery or AWS Lambda).
> * Event-driven streaming tools like Kafka for immediate inventory indexing adjustments.
> * Hybrid indexing strategies combined with advanced metadata filtering expressions during vector lookups.

---

# 📈 Future Enhancements

* 👤 Integration of personalized historic user purchasing data.
* 🔄 Real-time catalog updates via database triggers.
* 🔍 Hybrid Search architecture combining BM25 keywords + Dense Vectors.
* 📊 Comprehensive A/B testing infrastructure evaluating conversion rates.
* 📉 Live recommendation performance tracking analytics panel.

---

# 👨‍💻 Author
**Ashok Mamidi**  
GenAI Engineer | AI Architect | iOS + AI Developer 
