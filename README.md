# A³ - AI Application Assistant

A³ (AI Application Assistant) is a Retrieval-Augmented Generation (RAG) career assistant designed to help job seekers analyze their resumes against job descriptions, identify skill gaps, improve application materials, prepare for interviews and generate cover letters.

The application combines OpenAI's language models, vector embeddings, and semantic search to provide context-aware career guidance grounded in the user's resume and the target job description.

---

## Features

### Resume & Job Description Input

Users can provide documents using either:

* PDF Upload
* Direct Text Input

This flexibility allows the application to work with resumes and job descriptions from multiple sources.

---

### RAG-Powered Job Match Analysis

The application:

1. Splits resume and job description into chunks.
2. Generates vector embeddings using OpenAI Embeddings.
3. Stores embeddings in ChromaDB.
4. Retrieves relevant context using semantic search.
5. Produces a grounded job match analysis.

Outputs include:

* Match Score
* Summary
* Strong Matches
* Potential Gaps
* Recommended Projects to Discuss
* Resume Improvement Suggestions
* Final Recommendation

---

### Ask AAA

Users can ask custom questions such as:

* What skills do I have that match this role?
* What projects should I discuss during interviews?
* What are my biggest skill gaps?
* How well does my experience align with the job requirements?

Responses are generated using retrieved resume and job description context rather than relying solely on the language model.

---

### Guided Career Tools

#### Matching and Missing Skills Analysis

Identifies:

* Existing strengths
* Missing qualifications
* Recommended learning priorities
* Relevant projects to emphasize\

#### Cover Letter Generation

Generates a tailored cover letter based on:

* Resume content
* Job requirements
* Relevant projects and experience


#### Interview Preparation

Provides:

* Technical Questions
* Behavioral Questions
* Project-Based Questions
* Gap-Based Questions

Along with suggested talking points.

#### Resume Bullet Improvement

Generates stronger, more targeted resume bullet points while remaining truthful to the candidate's experience.

#### Good Fit Summary

Creates a concise interview-ready response explaining why the candidate is a strong fit for the position.


---

## RAG Architecture

### Document Processing

Resume and job description text are:

1. Extracted from PDF or text input.
2. Cleaned and normalized.
3. Split into overlapping chunks.

### Embeddings

Each chunk is converted into a vector representation using:

* OpenAI Embeddings
* text-embedding-3-small

### Vector Database

Embeddings are stored in:

* ChromaDB

Metadata is attached to each chunk to distinguish between:

* Resume
* Job Description

### Retrieval

When a user asks a question:

1. Semantic search retrieves the most relevant chunks.
2. Resume and job description context are combined.
3. Duplicate chunks are removed.
4. Retrieved context is passed to the language model.

### Generation

OpenAI GPT-4o Mini generates grounded responses using only the retrieved context.

This helps reduce hallucinations and improves answer relevance.

---

## Technology Stack

### Frontend

* Streamlit

### AI

* OpenAI GPT-4o Mini
* OpenAI Embeddings

### RAG

* LangChain
* ChromaDB

### Document Processing

* PyPDF

### Environment Management

* Python Dotenv

---

## Project Structure

```text
A3_AI_application_assistant/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── .streamlit/
│   └── config.toml
│
└── chroma_db/
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/AmanuelTenaw/AmanuelTenaw-A3_AI_application_assistant.git
```

Move into the project directory:

```bash
cd AmanuelTenaw-A3_AI_application_assistant
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

macOS/Linux

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```text
OPENAI_API_KEY=your_openai_api_key_here
```

---

## Run the Application

```bash
streamlit run app.py
```

---

## Future Enhancements

* Persistent vector storage
* User authentication
* Job application tracking
* Multi-resume management
* Recruiter feedback analysis
* Resume version comparison
* LinkedIn profile integration
* ATS optimization scoring
* Cloud deployment and monitoring

---

## Author

Amanuel Tenaw

* B.S. Computer Science, UNC Charlotte
* Early Entry Master's Student in Artificial Intelligence
* Software Engineering & AI Enthusiast

A³ was developed as a portfolio project demonstrating Retrieval-Augmented Generation (RAG), vector databases, semantic search, prompt engineering, and AI-assisted career tooling.
