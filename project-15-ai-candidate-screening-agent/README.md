# 🤖 AI Candidate Screening Agent

An AI-powered candidate screening platform built using **LangGraph, LangChain, LLMs, FastAPI, LangSmith, and Airtable**.

The system automates the initial recruitment workflow by analyzing job descriptions, evaluating resumes against hiring criteria, generating recruiter outreach messages, and storing candidate evaluation results.

---

# 🚀 Project Overview

Traditional candidate screening requires recruiters to manually review hundreds of resumes.

This project uses Generative AI to create an intelligent screening workflow:

1. Analyze Job Description
2. Extract hiring criteria
3. Evaluate candidate resumes
4. Generate candidate scores and reasoning
5. Draft personalized recruiter outreach emails
6. Store candidate results in Airtable

The workflow is orchestrated using **LangGraph** and monitored using **LangSmith observability**.

---

# 🏗️ Architecture
                     Client
                       |
                       |
                  FastAPI API
                       |
                       |
                  LangGraph
                       |
    -----------------------------------------
    |                  |                    |
    v                  v                    v
    Analyze     JD Score Resume       Draft Outreach
    v
  Save Candidate Results
    |
    v
Airtable DB          LangSmith
                       |
                       |
              Trace & Monitor
              Agent Execution


---

# ✨ Features

## 📌 Job Description Analysis

The AI extracts:

- Role title
- Required skills
- Nice-to-have skills
- Minimum experience
- Key responsibilities

Example:

Input: Senior React Native Engineer with AWS experience

Output:

```json
{
  "role_title": "Senior React Native Engineer",
  "required_skills": [
    "React Native",
    "TypeScript",
    "AWS"
  ],
  "min_years_experience": 5
}

📌 Resume Screening

The system evaluates resumes against job requirements.

Generates:

Candidate score (0-100)
Matched skills
Missing skills
Evaluation reasoning
Outreach recommendation

Example:
{
  "candidate": "John Smith",
  "score": 87,
  "matched_skills": [
    "React Native",
    "Redux",
    "AWS"
  ],
  "recommend_outreach": true
}

📌 AI Recruiter Outreach

For qualified candidates, the system generates personalized recruiter outreach emails.

Example:
Hello John,

We noticed your experience building React Native applications
and believe your background aligns with our Senior Mobile
Engineer role.

We would love to discuss this opportunity with you.

📌 Airtable Integration

Candidate evaluation results are automatically stored in Airtable.

Stored information:

Candidate Name
Score
Matched Skills
Missing Skills
Reasoning
Outreach Recommendation
Submission Timestamp

📌 LangSmith Observability

LangSmith provides visibility into:

LangGraph execution flow
LLM calls
Prompts
Outputs
Errors
Latency
Token usage

Example trace:

Candidate Screening Workflow

 |
 |-- analyse_jd
 |       |
 |       ---> Gemini LLM Call
 |
 |-- score_resumes
 |       |
 |       ---> Gemini LLM Calls
 |
 |-- draft_outreach
 |       |
 |       ---> Gemini LLM Call
 |
 |-- save_candidates
         |
         ---> Airtable API

🛠️ Technology Stack
Backend
Python
FastAPI
Pydantic
Generative AI
LangChain
LangGraph
Google Gemini
Structured Output Generation
Observability
LangSmith
Database / Storage
Airtable REST API

📂 Project Structure

project-15-ai-candidate-screening-agent

│
├── app
│   │
│   ├── api.py
│   │
│   ├── graphs
│   │   └── graph_builder.py
│   │
│   ├── nodes
│   │   ├── screening_node.py
│   │   └── airtable_node.py
│   │
│   ├── states
│   │   ├── screening_state.py
│   │   ├── jd_criteria.py
│   │   └── candidate_score.py
│   │
│   ├── llm
│   │   └── google_gemini_llm.py
│   │
│   └── services
│       └── airtable_client.py
│
├── requirements.txt
├── README.md
├── .gitignore
└── tests

⚙️ Installation
Clone Repository
git clone <repository-url>

cd project-15-ai-candidate-screening-agent

Create Virtual Environment

python -m venv venv

source venv/bin/activate

Install Dependencies
pip install -r requirements.txt

🔐 Environment Variables

Create a .env file:

GOOGLE_API_KEY=<your-google-api-key>

AIRTABLE_PAT=<your-airtable-token>

AIRTABLE_BASE_ID=<your-airtable-base-id>

AIRTABLE_TABLE_NAME=Candidates

LANGCHAIN_API_KEY=<your-langsmith-api-key>

LANGCHAIN_TRACING_V2=true

LANGCHAIN_PROJECT=AI-Candidate-Screening-Agent

▶️ Run Application

Start FastAPI server:
uvicorn app.api:app --reload --port 8000
Application runs at:
http://localhost:8000

🔌 API Endpoints
Health Check
GET /health

Response:
{
  "status": "ok"
}

Screen Candidates
POST /screen

Request:

{
  "jd_text": "Senior React Native Engineer with AWS experience",
  "company": "ABC Technologies",
  "resumes": [
    {
      "name": "John Smith",
      "text": "8 years React Native experience..."
    }
  ]
}

🔄 Current Workflow

Current implementation:
Single Agent Workflow

                 LangGraph
                     |
                     |
        ------------------------------
        |             |              |
        v             v              v

    JD Analysis   Resume Score   Outreach

                     |
                     v

              Airtable Storage

This implementation uses:

One AI agent
Multiple LangGraph nodes
External tool integration
Database persistence
LLM observability

🚀 Future Enhancements
Multi-Agent Architecture

Future evolution:
                    Supervisor Agent
                           |
        --------------------------------------
        |                 |                  |
        v                 v                  v

   JD Analysis       Resume Agent      Outreach Agent


                           |
                           v

                    Airtable Agent

Planned Features
Resume PDF parsing
OCR support for scanned resumes
RAG-based candidate search
Vector database integration
Candidate ranking dashboard
Human approval workflow
Email automation
Recruiter feedback loop
RAGAS evaluation
Production monitoring

🎯 Learning Objectives

This project demonstrates:

✅ Building AI workflows using LangGraph
✅ LLM structured outputs
✅ Agent workflow orchestration
✅ Tool integration
✅ FastAPI AI backend development
✅ LangSmith observability
✅ Production AI application patterns

👨‍💻 Author

Ashok Mamidi

Senior AI Architect | Generative AI Engineer | Mobile Tech Lead

Skills:

Generative AI
LLM Applications
RAG Systems
AI Agents
LangChain
LangGraph
Swift / SwiftUI
React Native
Python


One small improvement I made: I changed the architecture wording from **"multi-agent"** to **"Single Agent Workflow"** because that is accurate for your current implementation. This will actually look stronger in interviews because you can explain the evolution path from workflow → multi-agent architecture.