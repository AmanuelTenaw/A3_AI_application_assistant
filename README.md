# A³ - AI Application Assistant

A³ (AI Application Assistant) is a Retrieval-Augmented Generation (RAG) career assistant that helps job seekers evaluate their resumes against job descriptions using semantic search, vector embeddings, and Large Language Models (LLMs).
The application analyzes resume-job alignment, identifies skill gaps, generates tailored career guidance, prepares interview materials, and creates customized cover letters using retrieved context from both the resume and job description.
Rather than relying solely on an LLM’s general knowledge, A³ uses Retrieval-Augmented Generation (RAG) to ground responses in the user’s uploaded documents, improving relevance, transparency, and accuracy.

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

1.	Splits resume and job description into chunks.
2.	Generates vector embeddings using OpenAI Embeddings.
3.	Stores embeddings in ChromaDB.
4.	Retrieves relevant context using semantic search.
5.	Generates a grounded job fit analysis.


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

Responses are generated using retrieved resume and job description context through Retrieval-Augmented Generation (RAG), while maintaining limited conversational memory for follow-up questions.

---

### Guided Career Tools

#### Matching and Missing Skills Analysis

Identifies:

* Existing strengths
* Missing qualifications
* Recommended learning priorities
* Relevant projects to emphasize

#### Cover Letter Generation

Generates a tailored cover letter based on:

* Resume content
* Job requirements
* Relevant projects
* Candidate experience


#### Interview Preparation

Provides:

* Technical Questions
* Behavioral Questions
* Project-Based Questions
* Gap-Based Questions

Each question includes suggested talking points and answer strategies.

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

Mmetadata indicating whether each chunk originated from:

* Resume
* Job Description

### Retrieval

When a user asks a question:

1.	Semantic search retrieves relevant chunks.
2.	Resume and job description context are combined.
3.	Duplicate chunks are removed.
4.	Retrieved context is passed to the language model.


### Generation

Responses are generated using:
•	GPT-4o Mini
The model is instructed to use only the retrieved context whenever possible, reducing hallucinations and improving relevance.


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
   └── config.toml

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
*	User authentication
*	Multi-resume management
*	Job application tracking
*	ATS optimization scoring
*	Resume version comparison
*	Recruiter feedback analysis
*	LinkedIn profile integration
*	Cloud monitoring and analytics

---

## Author

Amanuel Tenaw

* B.S. Computer Science, UNC Charlotte
* Early Entry Master's Student in Artificial Intelligence
* Software Engineering & AI Enthusiast

A³ was developed as a portfolio project demonstrating Retrieval-Augmented Generation (RAG), vector databases, semantic search, prompt engineering, and AI-assisted career tooling.
