
Conversation with Gemini
generate README.md file with below

# 📈 Project 12 — SEC Filing Analyst Bot



![Level](https://img.shields.io/badge/Level-Intermediate-blue)

![Industry](https://img.shields.io/badge/Industry-Finance-green)

![Stack](https://img.shields.io/badge/Stack-LangChain%20%7C%20Pinecone%20%7C%20Gemini%20%7C%20FastAPI-orange)



A conversational RAG (Retrieval Augmented Generation) application that allows financial analysts to query SEC filings such as 10-K and 10-Q documents using natural language.



The system retrieves relevant filing sections from Pinecone and uses Google Gemini to generate answers grounded only in the retrieved context.



---



# 🧩 Business Problem



Financial analysts spend hours manually reading SEC filings to extract:



- Revenue information

- Risk factors

- Business performance

- Financial metrics

- Management discussions

- Future outlook



When companies have multiple filings across years, searching manually becomes inefficient.



This project solves this problem by creating an AI assistant that can answer questions from SEC filings and maintain conversation context for follow-up questions.



Example:



User:



What were ACME's major risk factors?





Assistant:



ACME identified supply chain disruption and market competition as major risks.





Follow-up:



Which risk impacted revenue the most?





The system understands that "which risk" refers to the previous answer.



---



# 🎯 Project Objectives



The application supports:



✅ SEC filing ingestion

✅ PDF and TXT document loading

✅ Document chunking

✅ HuggingFace embeddings

✅ Pinecone vector storage

✅ Metadata filtering

✅ Conversational memory

✅ Gemini-powered responses

✅ FastAPI backend

✅ Streamlit chat interface



---



# 🏗 Architecture



SEC Filings

PDF / TXT Files

|

|

v



ingest.py



PyMuPDFLoader / TextLoader



|

|



RecursiveCharacterTextSplitter



chunk_size = 1000

overlap = 150



|

|



Add Metadata



{

ticker: ACME,

year: 2025,

filing_type: 10-K

}



|

|



HuggingFace Embeddings



sentence-transformers/all-MiniLM-L6-v2



|

|



Pinecone Vector DB



|

|



ConversationalRetrievalChain



|

+-----------+-----------+

| |

| |

v v



Retriever ConversationBufferMemory



| |

+-----------+-----------+



|

|



Gemini 2.5 Flash



|

|



FastAPI



|

|



Streamlit UI





---



# 🛠 Technology Stack



| Component | Technology |

|---|---|

| LLM | Google Gemini 2.5 Flash |

| Embeddings | HuggingFace all-MiniLM-L6-v2 |

| Vector Database | Pinecone |

| RAG Framework | LangChain |

| Memory | ConversationBufferMemory |

| PDF Loader | PyMuPDFLoader |

| Text Loader | TextLoader |

| API | FastAPI |

| UI | Streamlit |

| Language | Python |



---



# 📁 Project Structure



project-12-sec-filing-analyst-bot/

│



├── app/



│ │



│ ├── ingest.py



│ │ Loads SEC documents



│ │ Creates embeddings



│ │ Stores vectors in Pinecone



│ │



│ ├── retriever.py



│ │ Creates RAG chain



│ │ Loads Gemini



│ │ Handles memory



│ │



│ ├── api.py



│ │ FastAPI chat endpoint



│ │



│ └── ui.py



│ Streamlit interface



│



├── samples/



│ └── sample_filings/



│



├── tests/



│



├── .env



├── requirements.txt



└── README.md



---



# ⚙️ Setup



## Create Virtual Environment



Mac/Linux:



```bash

python -m venv venv



source venv/bin/activate



Windows:





venv\Scripts\activate





Install Dependencies



pip install -r requirements.txt





🔐 Environment Configuration

Create .env

Example:



GOOGLE_API_KEY=your_google_gemini_key



PINECONE_API_KEY=your_pinecone_key



PINECONE_INDEX=filinganalystbot



GEMINI_MODEL=gemini-2.5-flash



📥 Document Ingestion

Command:





python app/ingest.py \--source samples/sample_filings \--ticker ACME \--year 2025 \--type 10-K



How Ingestion Works

1. Document Loading

The application scans the source folder.

Supported files:



.pdf

.txt



PDF:





PyMuPDFLoader()





TXT:





TextLoader()





Example:



samples/sample_filings/



ACME_2025_10K.txt



will be loaded.

2. Document Chunking

Large filings are split into smaller sections.

Configuration:





CHUNK_SIZE = 1000CHUNK_OVERLAP = 150



Example:



Original Document



10000 characters





Chunk 1

0-1000





Chunk 2

850-1850





Chunk 3

1700-2700



Overlap keeps context between chunks.

3. Metadata Creation

Each chunk receives:





{

"ticker":"ACME",

"year":"2025",

"filing_type":"10-K"

}



This enables filtering.

Example:

Search only:



ACME 2025 10-K





instead of searching all companies.

4. Embedding Generation

Model:



sentence-transformers/all-MiniLM-L6-v2





Example:

Text:



Revenue increased by 20 percent





becomes a vector:



[0.23,0.81,...]



384 dimensions



The vector is stored in Pinecone.

🔎 Retrieval Flow

When user asks:



What was ACME revenue?





The flow:



Question



|

v



Create Query Embedding



|

v



Search Pinecone



|

v



Retrieve Similar Chunks



|

v



Send Context + Question



|

v



Gemini Generates Answer



Pinecone Retriever

Code:





retriever = vectorstore.as_retriever(

search_kwargs={

"k":3,

"filter":pinecone_filter

}

)



search_kwargs Explanation

search_kwargs controls retrieval behavior.

k

Example:





"k":3





Means:

Return top 3 matching chunks.

Example:

Question:



What are company risks?





Pinecone returns:



Chunk 1 Risk Factors



Chunk 2 Market Risk



Chunk 3 Business Risk



filter

Example:





{

"ticker":{

"$eq":"ACME"

}

}



Means:

Only search ACME documents.

Ignore:



Tesla

Microsoft

Apple



💬 ConversationBufferMemory

Code:





memory = ConversationBufferMemory(

memory_key="chat_history",

return_messages=True,

output_key="answer"

)



Purpose:

Stores previous conversation messages.

Example:

First question:



What was revenue?





Memory:



User:

What was revenue?



Assistant:

Revenue was $5M



Follow-up:



How did it compare with last year?





The chain understands:

"it" = revenue

🤖 Gemini Integration

Gemini is loaded here:





ChatGoogleGenerativeAI(

model="gemini-2.5-flash"

)



The query reaches Gemini through:





ConversationalRetrievalChain.from_llm()





Flow:



User Question



+



Conversation History



+



Retrieved Pinecone Documents



|



v



Prompt Template



|



v



Gemini 2.5 Flash



|



v



Final Answer



Important:

Gemini does not search documents.

Pinecone retrieves.

Gemini generates.

🚀 Running Application

Step 1: Ingest Documents



python app/ingest.py \--source samples/sample_filings \--ticker ACME \--year 2025 \--type 10-K



Step 2: Start API



uvicorn app.api:app --reload --port 8000





API:



http://127.0.0.1:8000





Step 3: Start UI



streamlit run app/ui.py





Example Questions

What are the major risk factors?





What was the revenue growth?





How does this compare with previous year?





What are the company's future challenges?





⚠️ Troubleshooting

Gemini 404 Model Error

Error:



gemini-2.5-flash-lite is no longer available





Fix:

Change:



GEMINI_MODEL=gemini-2.5-flash





Gemini Quota Error

Error:



429 ResourceExhausted

Quota exceeded



Cause:

Free Gemini API has request limits.

Solutions:





Wait for quota reset



Use another API key



Enable billing



Reduce testing requests

Pinecone No Results

Check:





Index name

PINECONE_INDEX







Metadata values

Example:



ACME





not



acme







Correct ingestion command

🎤 Interview Discussion Points

Why RAG?

LLMs do not know private company filings.

RAG provides:





Relevant documents



Grounded answers



Reduced hallucination

Why Pinecone?

Because it supports:





Vector similarity search



Metadata filtering



Large scale storage

Why ConversationRetrievalChain?

Normal RetrievalQA:



Question → Answer





No memory.

Conversational Retrieval:



Question

+

Previous conversation

+

Documents

→ Answer



Production Improvements

Future enhancements:





Redis conversation storage



User authentication



SEC EDGAR API integration



PDF page citations



Hybrid search



Financial calculation agents



Cloud deployment

Final Architecture Summary

SEC Documents



↓



Chunking



↓



HuggingFace Embeddings



↓



Pinecone Vector Database



↓



Retriever



↓



Conversation Memory



↓



Gemini 2.5 Flash



↓



FastAPI



↓



Streamlit Chat Application



End

SEC Filing Analyst Bot demonstrates an enterprise-style RAG pipeline for financial document intelligence.





This version matches your current implementation instead of the original OpenAI-based README.

         SEC Filings
        PDF / TXT Files
              |
              |
              v

          ingest.py

  PyMuPDFLoader / TextLoader

              |
              |

RecursiveCharacterTextSplitter

      chunk_size = 1000
      overlap    = 150

              |
              |

      Add Metadata

{
  ticker: ACME,
  year: 2025,
  filing_type: 10-K
}

              |
              |

HuggingFace Embeddings
sentence-transformers/all-MiniLM-L6-v2

              |
              |

      Pinecone Vector DB

              |
              |

ConversationalRetrievalChain

              |
  +-----------+-----------+
  |                       |
  |                       |
  v                       v
Retriever                ConversationBufferMemory

  |                       |
  +-----------+-----------+

              |
              |

      Gemini 2.5 Flash

              |
              |

          FastAPI

              |
              |

      Streamlit UI

---

# 🛠 Technology Stack

| Component | Technology |
|---|---|
| LLM | Google Gemini 2.5 Flash |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector Database | Pinecone |
| RAG Framework | LangChain |
| Memory | ConversationBufferMemory |
| PDF Loader | PyMuPDFLoader |
| Text Loader | TextLoader |
| API | FastAPI |
| UI | Streamlit |
| Language | Python |

---

# 📁 Project Structure

project-12-sec-filing-analyst-bot/
│
├── app/
│   ├── ingest.py            # Loads SEC documents, creates embeddings, stores vectors
│   ├── retriever.py         # Creates RAG chain, loads Gemini, handles memory
│   ├── api.py               # FastAPI chat endpoint
│   └── ui.py                # Streamlit interface
│
├── samples/
│   └── sample_filings/
│
├── tests/
│
├── .env
├── requirements.txt
└── README.md


---

# ⚙️ Setup

## Create Virtual Environment

Mac/Linux:
```bash
python -m venv venv
source venv/bin/activate
Windows:

Bash
venv\\Scripts\\activate
Install Dependencies
Bash
pip install -r requirements.txt
🔐 Environment Configuration
Create a .env file in the root directory:

Code snippet
GOOGLE_API_KEY=your_google_gemini_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX=filinganalystbot
GEMINI_MODEL=gemini-2.5-flash
📥 Document Ingestion
Command:

Bash
python app/ingest.py --source samples/sample_filings --ticker ACME --year 2025 --type 10-K
How Ingestion Works
1. Document Loading
The application scans the source folder. Supported files: .pdf, .txt

PDF: PyMuPDFLoader()

TXT: TextLoader()

Example: samples/sample_filings/ACME_2025_10K.txt will be loaded.

2. Document Chunking
Large filings are split into smaller sections.
Configuration:

Python
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
Original Document: 10,000 characters

Chunk 1: 0-1000

Chunk 2: 850-1850

Chunk 3: 1700-2700

Overlap keeps context between chunks.

3. Metadata Creation
Each chunk receives:

JSON
{
 "ticker": "ACME",
 "year": "2025",
 "filing_type": "10-K"
}
This enables filtering, allowing search only across ACME 2025 10-K instead of searching all companies.

4. Embedding Generation
Model: sentence-transformers/all-MiniLM-L6-v2

Text: "Revenue increased by 20 percent"

Vector: [0.23, 0.81, ...] (384 dimensions)

The vector is stored in Pinecone.

🔎 Retrieval Flow
When a user asks: "What was ACME revenue?"

Question -> Create Query Embedding -> Search Pinecone -> Retrieve Similar Chunks -> Send Context + Question -> Gemini Generates Answer
Pinecone Retriever
Python
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3,
        "filter": pinecone_filter
    }
)
search_kwargs Explanation
Controls retrieval behavior:

k: 3 — Return top 3 matching chunks.

filter — Only search specified documents (e.g., {"ticker": {"$eq": "ACME"}}), ignoring other companies.

💬 ConversationBufferMemory
Python
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)
Purpose: Stores previous conversation messages so the chain understands that follow-up pronouns (e.g., "it") refer to entities in the previous exchange.

🤖 Gemini Integration
Gemini is loaded here:

Python
ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)
The query reaches Gemini through ConversationalRetrievalChain.from_llm():

User Question + Conversation History + Retrieved Pinecone Documents 
                       ↓
                Prompt Template
                       ↓
               Gemini 2.5 Flash
                       ↓
                 Final Answer
Important: Gemini does not search documents. Pinecone retrieves, Gemini generates.

🚀 Running Application
Step 1: Ingest Documents
Bash
python app/ingest.py --source samples/sample_filings --ticker ACME --year 2025 --type 10-K
Step 2: Start API
Bash
uvicorn app.api:app --reload --port 8000
API Endpoint: http://127.0.0.1:8000

Step 3: Start UI
Bash
streamlit run app/ui.py
Example Questions
What are the major risk factors?

What was the revenue growth?

How does this compare with previous year?

What are the company's future challenges?

⚠️ Troubleshooting
Gemini 404 Model Error
Error: gemini-2.5-flash-lite is no longer available

Fix: Change to GEMINI_MODEL=gemini-2.5-flash

Gemini Quota Error
Error: 429 ResourceExhausted Quota exceeded

Cause: Free Gemini API has request limits.

Solutions: Wait for quota reset, use another API key, enable billing, or reduce testing requests.

Pinecone No Results
Check index name (PINECONE_INDEX)

Check metadata values (e.g., ACME instead of acme)

Verify the correct ingestion command was executed.

🎤 Interview Discussion Points
Why RAG?
LLMs do not know private or specific company filings. RAG provides relevant documents, grounded answers, and reduced hallucinations.

Why Pinecone?
Because it natively supports vector similarity search, metadata filtering, and large scale storage.

Why ConversationalRetrievalChain?
Normal RetrievalQA handles Question → Answer without memory. ConversationalRetrievalChain handles Question + Previous conversation + Documents → Answer.

Production Improvements
Redis conversation storage

User authentication

SEC EDGAR API integration

PDF page citations

Hybrid search

Financial calculation agents

Cloud deployment

🏗 Final Architecture Summary
SEC Documents -> Chunking -> HuggingFace Embeddings -> Pinecone Vector Database -> Retriever -> Conversation Memory -> Gemini 2.5 Flash -> FastAPI -> Streamlit Chat Application
SEC Filing Analyst Bot demonstrates an enterprise-style RAG pipeline for financial document intelligence.
"""

with open("README.md", "w", encoding="utf-8") as f:
f.write(content)
print("File successfully generated.")


```text?code_stdout&code_event_index=1
File successfully generated.

Your Markdown file is ready
MD icon
README
 MD 
Open

I have successfully structured and formatted the README.md file for Project 12 — SEC Filing Analyst Bot, converting the plain text diagrams and lists into clean, professional Markdown with proper syntax highlighting, formatting, and structural alignments.




Gemini is AI and can make mistakes.

# 📈 Project 12 — SEC Filing Analyst Bot

![Level](https://img.shields.io/badge/Level-Intermediate-blue)
![Industry](https://img.shields.io/badge/Industry-Finance-green)
![Stack](https://img.shields.io/badge/Stack-LangChain%20%7C%20Pinecone%20%7C%20Gemini%20%7C%20FastAPI-orange)

A conversational RAG (Retrieval Augmented Generation) application that allows financial analysts to query SEC filings such as 10-K and 10-Q documents using natural language.

The system retrieves relevant filing sections from Pinecone and uses Google Gemini to generate answers grounded only in the retrieved context.

---

# 🧩 Business Problem

Financial analysts spend hours manually reading SEC filings to extract:

- Revenue information
- Risk factors
- Business performance
- Financial metrics
- Management discussions
- Future outlook

When companies have multiple filings across years, searching manually becomes inefficient.

This project solves this problem by creating an AI assistant that can answer questions from SEC filings and maintain conversation context for follow-up questions.

Example:

User:
What were ACME's major risk factors?

Assistant:
ACME identified supply chain disruption and market competition as major risks.

Follow-up:
Which risk impacted revenue the most?

The system understands that "which risk" refers to the previous answer.

---

# 🎯 Project Objectives

The application supports:

✅ SEC filing ingestion  
✅ PDF and TXT document loading  
✅ Document chunking  
✅ HuggingFace embeddings  
✅ Pinecone vector storage  
✅ Metadata filtering  
✅ Conversational memory  
✅ Gemini-powered responses  
✅ FastAPI backend  
✅ Streamlit chat interface  

---

# 🏗 Architecture

```
             SEC Filings
            PDF / TXT Files
                  |
                  |
                  v

              ingest.py

      PyMuPDFLoader / TextLoader

                  |
                  |

    RecursiveCharacterTextSplitter

          chunk_size = 1000
          overlap    = 150

                  |
                  |

          Add Metadata

    {
      ticker: ACME,
      year: 2025,
      filing_type: 10-K
    }

                  |
                  |

    HuggingFace Embeddings

sentence-transformers/all-MiniLM-L6-v2

                  |
                  |

          Pinecone Vector DB

                  |
                  |

    ConversationalRetrievalChain

                  |
      +-----------+-----------+
      |                       |
      |                       |
      v                       v

Retriever                ConversationBufferMemory

      |                       |
      +-----------+-----------+

                  |
                  |

          Gemini 2.5 Flash

                  |
                  |

              FastAPI

                  |
                  |

          Streamlit UI
```

---

# 🛠 Technology Stack

| Component | Technology |
|---|---|
| LLM | Google Gemini 2.5 Flash |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector Database | Pinecone |
| RAG Framework | LangChain |
| Memory | ConversationBufferMemory |
| PDF Loader | PyMuPDFLoader |
| Text Loader | TextLoader |
| API | FastAPI |
| UI | Streamlit |
| Language | Python |

---

# 📁 Project Structure

```
project-12-sec-filing-analyst-bot/
│
├── app/
│   ├── ingest.py            # Loads SEC documents, creates embeddings, stores vectors
│   ├── retriever.py         # Creates RAG chain, loads Gemini, handles memory
│   ├── api.py               # FastAPI chat endpoint
│   └── ui.py                # Streamlit interface
│
├── samples/
│   └── sample_filings/
│
├── tests/
│
├── .env
├── requirements.txt
└── README.md
```

---

# ⚙️ Setup

## Create Virtual Environment

Mac/Linux:
```bash
python -m venv venv
source venv/bin/activate
```

Windows:
```bash
venv\Scripts\activate
```

## Install Dependencies
```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Configuration
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_google_gemini_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX=filinganalystbot
GEMINI_MODEL=gemini-2.5-flash
```

---

# 📥 Document Ingestion
Command:
```bash
python app/ingest.py --source samples/sample_filings --ticker ACME --year 2025 --type 10-K
```

## How Ingestion Works
### 1. Document Loading
The application scans the source folder. Supported files: `.pdf`, `.txt`
* PDF: `PyMuPDFLoader()`
* TXT: `TextLoader()`

Example: `samples/sample_filings/ACME_2025_10K.txt` will be loaded.

### 2. Document Chunking
Large filings are split into smaller sections.
Configuration:
```python
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
```

* **Original Document:** 10,000 characters
* **Chunk 1:** 0-1000
* **Chunk 2:** 850-1850
* **Chunk 3:** 1700-2700

*Overlap keeps context between chunks.*

### 3. Metadata Creation
Each chunk receives:
```json
{
 "ticker": "ACME",
 "year": "2025",
 "filing_type": "10-K"
}
```
This enables filtering, allowing search only across `ACME 2025 10-K` instead of searching all companies.

### 4. Embedding Generation
Model: `sentence-transformers/all-MiniLM-L6-v2`

* **Text:** `"Revenue increased by 20 percent"`
* **Vector:** `[0.23, 0.81, ...]` (384 dimensions)

The vector is stored in Pinecone.

---

# 🔎 Retrieval Flow
When a user asks: `"What was ACME revenue?"`

```
Question -> Create Query Embedding -> Search Pinecone -> Retrieve Similar Chunks -> Send Context + Question -> Gemini Generates Answer
```

### Pinecone Retriever
```python
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3,
        "filter": pinecone_filter
    }
)
```

### `search_kwargs` Explanation
Controls retrieval behavior:
* **`k`: 3** — Return top 3 matching chunks.
* **`filter`** — Only search specified documents (e.g., `{"ticker": {"$eq": "ACME"}}`), ignoring other companies.

---

# 💬 ConversationBufferMemory
```python
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)
```
**Purpose:** Stores previous conversation messages so the chain understands that follow-up pronouns (e.g., "it") refer to entities in the previous exchange.

---

# 🤖 Gemini Integration
Gemini is loaded here:
```python
ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)
```
The query reaches Gemini through `ConversationalRetrievalChain.from_llm()`:
```
User Question + Conversation History + Retrieved Pinecone Documents 
                       ↓
                Prompt Template
                       ↓
               Gemini 2.5 Flash
                       ↓
                 Final Answer
```
*Important: Gemini does not search documents. Pinecone retrieves, Gemini generates.*

---

# 🚀 Running Application

### Step 1: Ingest Documents
```bash
python app/ingest.py --source samples/sample_filings --ticker ACME --year 2025 --type 10-K
```

### Step 2: Start API
```bash
uvicorn app.api:app --reload --port 8000
```
API Endpoint: `http://127.0.0.1:8000`

### Step 3: Start UI
```bash
streamlit run app/ui.py
```

### Example Questions
* What are the major risk factors?
* What was the revenue growth?
* How does this compare with previous year?
* What are the company's future challenges?

---

# ⚠️ Troubleshooting

### Gemini 404 Model Error
* **Error:** `gemini-2.5-flash-lite is no longer available`
* **Fix:** Change to `GEMINI_MODEL=gemini-2.5-flash`

### Gemini Quota Error
* **Error:** `429 ResourceExhausted Quota exceeded`
* **Cause:** Free Gemini API has request limits.
* **Solutions:** Wait for quota reset, use another API key, enable billing, or reduce testing requests.

### Pinecone No Results
* Check index name (`PINECONE_INDEX`)
* Check metadata values (e.g., `ACME` instead of `acme`)
* Verify the correct ingestion command was executed.

---

# 🎤 Interview Discussion Points

### Why RAG?
LLMs do not know private or specific company filings. RAG provides relevant documents, grounded answers, and reduced hallucinations.

### Why Pinecone?
Because it natively supports vector similarity search, metadata filtering, and large scale storage.

### Why ConversationalRetrievalChain?
Normal `RetrievalQA` handles `Question → Answer` without memory. `ConversationalRetrievalChain` handles `Question + Previous conversation + Documents → Answer`.

### Production Improvements
* Redis conversation storage
* User authentication
* SEC EDGAR API integration
* PDF page citations
* Hybrid search
* Financial calculation agents
* Cloud deployment

---

# 🏗 Final Architecture Summary
```
SEC Documents -> Chunking -> HuggingFace Embeddings -> Pinecone Vector Database -> Retriever -> Conversation Memory -> Gemini 2.5 Flash -> FastAPI -> Streamlit Chat Application
```

*SEC Filing Analyst Bot demonstrates an enterprise-style RAG pipeline for financial document intelligence.*
README.md
Displaying README.md.