"""
A³ - AI Application Assistant

Author: Amanuel Tenaw

Description:
A³ is a RAG-powered career assistant that compares a candidate's resume
against a job description. It helps users analyze job fit, identify missing
skills, improve resume bullets, prepare for interviews, create a good-fit
summary, and generate a tailored cover letter.

Main Features:
- Resume input by PDF upload or pasted text
- Job description input by PDF upload or pasted text
- RAG-based job match analysis
- Resume/job description Q&A through Ask AAA
- Guided career tools 
- Retrieved source chunk visibility for transparency
- Downloadable results

Tech Stack:
- Streamlit
- OpenAI API
- LangChain
- ChromaDB
- OpenAI Embeddings
- pypdf
"""

import os
import re
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from openai import OpenAI

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


# =====================================================
# ENVIRONMENT SETUP
# Loads API keys and initializes the OpenAI client
# =====================================================

load_dotenv()

api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

client = OpenAI(api_key=api_key)
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =====================================================
# STREAMLIT PAGE CONFIGURATION
# Sets the browser title and page layout
# =====================================================

st.set_page_config(
    page_title="A³ - AI Application Assistant",
    layout="wide"
)

st.markdown("""
<style>

/* Main page */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Expanders */
div[data-testid="stExpander"] {
    border: 1px solid #E5E7EB;
    border-radius: 10px;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 8px;
    height: 3rem;
    font-weight: 500;
}

/* Upload boxes */
[data-testid="stFileUploader"] {
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 10px;
}

/* Cleaner spacing */
h1 {
    margin-bottom: 0.25rem;
}

h2, h3 {
    margin-top: 1rem;
}

</style>
""", unsafe_allow_html=True) 

st.title("A³ - AI Application Assistant")
st.caption("Analyze • Prepare • Succeed")
st.write("Upload or paste your resume and job description to begin.")


# =====================================================
# SESSION STATE SETUP
# Stores app results so they stay visible after reruns
# =====================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "job_match_result" not in st.session_state:
    st.session_state.job_match_result = None

if "job_match_docs" not in st.session_state:
    st.session_state.job_match_docs = []

if "guided_results" not in st.session_state:
    st.session_state.guided_results = {}

#if "last_resume_text" not in st.session_state:
#    st.session_state.last_resume_text = ""

#if "last_job_text" not in st.session_state:
#    st.session_state.last_job_text = ""

if "current_input_key" not in st.session_state:
    st.session_state.current_input_key = ""

# =====================================================
# INPUT PROCESSING FUNCTIONS
# Handles PDF extraction and pasted text cleaning
# =====================================================

def extract_pdf_text(uploaded_file):
    """
    Extract text from an uploaded PDF file.

    Args:
        uploaded_file:
            A PDF file uploaded through Streamlit's file uploader.

    Returns:
        str:
            Cleaned text extracted from all readable PDF pages.
    """

    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + " "

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_text(text):
    """
    Clean pasted or extracted text by removing extra whitespace.

    Args:
        text (str):
            Raw text from a PDF or text area.

    Returns:
        str:
            Cleaned text.
    """

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_resume_input():
    """
    Display resume input options in the Streamlit UI.

    The user can either upload a resume PDF or paste resume text.

    Returns:
        str:
            Cleaned resume text.
    """

    st.subheader("Resume Input")

    resume_input_type = st.radio(
        "Choose resume input type",
        ["Upload PDF", "Paste Text"],
        horizontal=True,
        key="resume_input_type"
    )

    resume_text = ""

    if resume_input_type == "Upload PDF":
        resume_file = st.file_uploader(
            "Upload Resume PDF",
            type=["pdf"],
            key="resume_pdf"
        )

        if resume_file:
            resume_text = extract_pdf_text(resume_file)

    else:
        resume_text = st.text_area(
            "Paste Resume Text",
            height=250,
            key="resume_text_area"
        )

    return clean_text(resume_text)


def get_job_input():
    """
    Display job description input options in the Streamlit UI.

    The user can either upload a job description PDF or paste job text.

    Returns:
        str:
            Cleaned job description text.
    """

    st.subheader("Job Description Input")

    job_input_type = st.radio(
        "Choose job description input type",
        ["Upload PDF", "Paste Text"],
        horizontal=True,
        key="job_input_type"
    )

    job_text = ""

    if job_input_type == "Upload PDF":
        job_file = st.file_uploader(
            "Upload Job Description PDF",
            type=["pdf"],
            key="job_pdf"
        )

        if job_file:
            job_text = extract_pdf_text(job_file)

    else:
        job_text = st.text_area(
            "Paste Job Description Text",
            height=250,
            key="job_text_area"
        )

    return clean_text(job_text)


# =====================================================
# RAG VECTORSTORE CREATION
# Splits documents, embeds chunks, and stores them in ChromaDB
# =====================================================

def create_rag_vectorstore(resume_text, job_text):
    """
    Create the RAG vector database from resume and job description text.

    Process:
    1. Split resume and job description into smaller chunks.
    2. Attach metadata to identify each chunk source.
    3. Convert chunks into embeddings using OpenAI embeddings.
    4. Store the embedded chunks in ChromaDB.

    Args:
        resume_text (str):
            Candidate resume text.

        job_text (str):
            Job description text.

    Returns:
        Chroma:
            Vector database containing resume and job description chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    resume_chunks = splitter.split_text(resume_text)
    job_chunks = splitter.split_text(job_text)

    texts = []
    metadatas = []

    # Store resume chunks with Resume metadata.
    for chunk in resume_chunks:
        if chunk.strip():
            texts.append(chunk)
            metadatas.append({"source": "Resume"})

    # Store job description chunks with Job Description metadata.
    for chunk in job_chunks:
        if chunk.strip():
            texts.append(chunk)
            metadatas.append({"source": "Job Description"})

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key
    )

    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )

    return vectorstore


# =====================================================
# RETRIEVAL FUNCTIONS
# Retrieves relevant chunks from the vector database
# =====================================================

def remove_duplicate_docs(docs):
    """
    Remove duplicate retrieved chunks.

    Args:
        docs (list):
            Retrieved LangChain document objects.

    Returns:
        list:
            Unique retrieved documents.
    """

    unique_docs = []
    seen_texts = set()

    for doc in docs:
        cleaned_text = doc.page_content.strip()

        if cleaned_text not in seen_texts:
            unique_docs.append(doc)
            seen_texts.add(cleaned_text)

    return unique_docs


def retrieve_docs(question, vectorstore):
    """
    Retrieve relevant resume and job description chunks for a question.

    This function searches the resume and job description separately.
    This helps ensure the final answer compares both the candidate's
    background and the job requirements instead of only retrieving from one side.

    Args:
        question (str):
            User question or guided tool prompt.

        vectorstore:
            Chroma vector database.

    Returns:
        list:
            Relevant resume and job description source chunks.
    """

    resume_query = question + " candidate resume skills experience projects education technical skills"
    job_query = question + " job description required skills qualifications responsibilities technologies"

    resume_docs = vectorstore.max_marginal_relevance_search(
        resume_query,
        k=3,
        fetch_k=10,
        filter={"source": "Resume"}
    )

    job_docs = vectorstore.max_marginal_relevance_search(
        job_query,
        k=3,
        fetch_k=10,
        filter={"source": "Job Description"}
    )

    return remove_duplicate_docs(resume_docs + job_docs)


def retrieve_match_context(vectorstore):
    """
    Retrieve a broad set of chunks for the job match analysis.

    Job match analysis needs more context than a normal question,
    so this function uses multiple targeted queries covering skills,
    projects, gaps, technologies, and responsibilities.

    Args:
        vectorstore:
            Chroma vector database.

    Returns:
        list:
            Retrieved source chunks for the full job match analysis.
    """

    queries = [
        "required skills qualifications responsibilities job requirements candidate matching skills experience",
        "technical skills programming languages frameworks cloud aws frontend backend database ai",
        "projects experience internships education certifications relevant work",
        "gaps missing skills weaknesses required technologies not shown in resume",
        "production support client facing debugging documentation"
    ]

    all_docs = []

    for query in queries:
        all_docs.extend(retrieve_docs(query, vectorstore))

    return remove_duplicate_docs(all_docs)


def docs_to_context(docs):
    """
    Convert retrieved document chunks into formatted context for the LLM.

    Args:
        docs (list):
            Retrieved LangChain documents.

    Returns:
        str:
            Formatted context with source labels.
    """

    return "\n\n".join(
        [
            f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
            for doc in docs
        ]
    )


# =====================================================
# LLM ANALYSIS FUNCTIONS
# Uses retrieved RAG context to generate grounded outputs
# =====================================================

def analyze_job_match(vectorstore):
    """
    Generate a RAG-grounded job match analysis.

    The model is instructed to use only retrieved resume and job
    description context. This prevents the app from inventing skills,
    projects, or experience that are not in the resume.

    Args:
        vectorstore:
            Chroma vector database.

    Returns:
        tuple:
            analysis text and source documents.
    """

    docs = retrieve_match_context(vectorstore)
    context = docs_to_context(docs)

    prompt = f"""
You are an AI career assistant using RAG.

Use ONLY the retrieved Resume and Job Description context below.
Do not invent experience that is not supported by the Resume context.

Do NOT include interview questions in this analysis.

Return the response in this exact format:

Match Score: [number]%

Summary:
[short paragraph]

Strong Matches:
- [skill/experience and why it matches]
- [skill/experience and why it matches]
- [skill/experience and why it matches]

Potential Gaps:
- [missing or weak skill and why it matters]
- [missing or weak skill and why it matters]
- [missing or weak skill and why it matters]

Recommended Projects to Discuss:
- [project or experience and what to emphasize]
- [project or experience and what to emphasize]

Resume Improvement Suggestions:
- [specific truthful suggestion]
- [specific truthful suggestion]

Final Recommendation:
[short recommendation]

Retrieved Context:
{context}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI career assistant that gives practical, honest, RAG-grounded job fit analysis."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content, docs


def generate_rag_answer(question, vectorstore, include_chat_history=True):
    """
    Generate a RAG-based answer for Ask AAA or guided career tools.

    Args:
        question (str):
            User question or guided feature prompt.

        vectorstore:
            Chroma vector database.

        include_chat_history (bool):
            Whether to include recent chat history in the prompt.

    Returns:
        tuple:
            generated answer and retrieved source documents.
    """

    docs = retrieve_docs(question, vectorstore)
    context = docs_to_context(docs)

    previous_chat = ""

    if include_chat_history:
        previous_chat = "\n".join(
            [
                f"User: {chat['question']}\nAAA: {chat['answer']}"
                for chat in st.session_state.chat_history[-3:]
            ]
        )

    prompt = f"""
You are AAA, an AI Application Assistant.

Use the Resume context and Job Description context to answer the user's question.

Important rules:
- If the user says "I", "me", or "my", treat that as the candidate in the uploaded resume.
- Compare the candidate's resume against the job description when the question asks about fit, skills, gaps, projects, or preparation.
- Do not only list job description requirements. Explain which resume skills match the job.
- If something is missing from the resume, clearly say it appears to be a gap.
- Mention whether information came from the Resume or Job Description when useful.
- Do not invent experience.

Previous Conversation:
{previous_chat}

Retrieved Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI career assistant that compares a resume with a job description using retrieved context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content, docs


def ask_aaa(question, vectorstore):
    """
    Run the main Ask AAA chat workflow.

    This function retrieves relevant RAG context, generates an answer,
    and saves the question/answer pair in chat history.

    Args:
        question (str):
            User's question.

        vectorstore:
            Chroma vector database.

    Returns:
        tuple:
            generated answer and source documents.
    """

    answer, docs = generate_rag_answer(
        question=question,
        vectorstore=vectorstore,
        include_chat_history=True
    )

    st.session_state.chat_history.append(
        {
            "question": question,
            "answer": answer
        }
    )

    return answer, docs


# =====================================================
# GUIDED CAREER TOOLS
# Prebuilt prompts for common job application tasks
# =====================================================

def run_guided_feature(feature_name, vectorstore):
    """
    Run one of the guided career tools.

    Supported tools:
    - Matching and Missing Skills Analysis
    - Resume Bullet Improvement
    - Interview Prep Questions
    - Good Fit Summary
    - Cover Letter

    Args:
        feature_name (str):
            Key identifying which guided feature to run.

        vectorstore:
            Chroma vector database.

    Returns:
        tuple:
            generated answer and retrieved source documents.
    """

    feature_prompts = {
        "missing_skills": """
Analyze the candidate's missing skills for this job.

Compare the Resume against the Job Description.

Return:
1. Strong matching skills
2. Missing or weak skills
3. Skills to learn first
4. Projects/experiences the candidate should highlight
""",
        "resume_bullets": """
Suggest improved resume bullets that better match this job description.

Use the candidate's existing experience from the Resume.

Return strong, truthful bullet points using action verbs, tools, and impact.
Do not invent experience.
""",
        "interview_questions": """
Generate interview prep questions with suggested answer strategies.

Use the Resume and Job Description.

Return:
1. Technical questions with suggested talking points
2. Behavioral questions with suggested talking points
3. Project-based questions with suggested talking points
4. Gap-based questions the candidate should be ready for

Do not invent experience.
Keep suggestions truthful and based on the retrieved Resume context.
""",
        "good_fit_summary": """
Write a concise 'Why I am a good fit for this role' answer.

Use the Resume and Job Description.

Make it sound natural, confident, and interview-ready.
""",
        "cover_letter": """
Write a tailored cover letter for this job.

Use the Resume and Job Description.

Requirements:
- Start directly with 'Dear Hiring Manager,'.
- Do NOT include:
    - Candidate name
    - Address
    - Email
    - Phone number
    - Date
    - Company address
    - Placeholder fields
- Make it professional, confident, and natural.
- Keep it truthful and based only on the retrieved Resume context.
- Do not invent experience.
- Mention the candidate's strongest matching projects and skills.
- Explain why the candidate is interested in the role.
- Keep it around 3–5 paragraphs.
- End with:
  'Thank you for your time and consideration. I look forward to the opportunity to discuss how my background and experiences can contribute to your team.'
- Finish with:
  'Sincerely,'
  'Amanuel Tenaw'
"""
    }

    question = feature_prompts[feature_name]

    answer, docs = generate_rag_answer(
        question=question,
        vectorstore=vectorstore,
        include_chat_history=False
    )

    return answer, docs


# =====================================================
# DOWNLOAD HELPER
# Formats generated results for text download
# =====================================================

def generate_download_text(title, content):
    """
    Format generated content for text file downloads.

    Args:
        title (str):
            Title of the generated output.

        content (str):
            Main generated content.

    Returns:
        str:
            Download-ready text.
    """

    return f"""
{title}

{content}
"""


# =====================================================
# STREAMLIT USER INTERFACE
# Main app layout and workflow
# =====================================================

left_col, right_col = st.columns(2)

with left_col:
    resume_text = get_resume_input()

with right_col:
    job_text = get_job_input()


# Show previews only after text exists.
if resume_text:
    with st.expander("View Resume Preview"):
        st.text_area("Resume Content Preview", resume_text, height=200)

if job_text:
    with st.expander("View Job Description Preview"):
        st.text_area("Job Description Content Preview", job_text, height=200)


# The RAG app only starts after both resume and job description are provided.
if resume_text and job_text:

    current_input_key = str(hash(resume_text + job_text))
    
    if current_input_key != st.session_state.current_input_key:
        st.session_state.chat_history = []
        st.session_state.job_match_result = None
        st.session_state.job_match_docs = []
        st.session_state.guided_results = {}
        st.session_state.current_input_key = current_input_key

    # Reset previous outputs when the user changes the resume or job description.
    #if (
    #    resume_text != st.session_state.last_resume_text
    #    or job_text != st.session_state.last_job_text
    #):
     #   st.session_state.chat_history = []
      #  st.session_state.job_match_result = None
       # st.session_state.job_match_docs = []
        #st.session_state.guided_results = {}

 #       st.session_state.last_resume_text = resume_text
#        st.session_state.last_job_text = job_text

    st.divider()

    # Build the RAG vectorstore from the latest resume/job text.
    with st.spinner("Building RAG system..."):
        vectorstore = create_rag_vectorstore(resume_text, job_text)

    st.success("RAG system ready.")


    # =====================================================
    # JOB MATCH ANALYSIS SECTION
    # =====================================================

    st.subheader("Job Match Analysis")

    if st.button("Analyze Job Match"):
        with st.spinner("Running RAG-powered job match analysis..."):
            result, source_docs = analyze_job_match(vectorstore)

        st.session_state.job_match_result = result
        st.session_state.job_match_docs = source_docs

    if st.session_state.job_match_result:
        with st.expander("View Job Match Analysis", expanded=True):
            st.write(st.session_state.job_match_result)

            st.download_button(
                label="Download Job Match Analysis",
                data=generate_download_text(
                    "Job Match Analysis",
                    st.session_state.job_match_result
                ),
                file_name="job_match_analysis.txt",
                mime="text/plain"
            )

        with st.expander("View Retrieved Source Chunks Used for Job Match"):
            for i, doc in enumerate(st.session_state.job_match_docs, start=1):
                source = doc.metadata.get("source", "Unknown")

                st.markdown(f"### Source Chunk {i}")
                st.write(f"**Source:** {source}")
                st.write(doc.page_content)


    # =====================================================
    # ASK AAA CHAT SECTION
    # =====================================================

    st.divider()

    st.subheader("Ask AAA")

    question = st.text_input(
        "Ask a question about your resume and this job description"
    )

    if question:
        with st.spinner("AAA is thinking..."):
            answer, source_docs = ask_aaa(question, vectorstore)

        st.write(answer)

        with st.expander("View Retrieved Source Chunks"):
            for i, doc in enumerate(source_docs, start=1):
                source = doc.metadata.get("source", "Unknown")

                st.markdown(f"### Source Chunk {i}")
                st.write(f"**Source:** {source}")
                st.write(doc.page_content)

    if st.session_state.chat_history:
        with st.expander("Chat History"):
            for i, chat in enumerate(st.session_state.chat_history, start=1):
                st.markdown(f"### Question {i}")
                st.write(chat["question"])
                st.markdown("**AAA Answer:**")
                st.write(chat["answer"])


    # =====================================================
    # GUIDED CAREER TOOLS SECTION
    # =====================================================

    st.divider()

    st.subheader("Guided Career Tools")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Matching and Missing Skills Analysis"):
            with st.spinner("Analyzing missing skills..."):
                answer, source_docs = run_guided_feature("missing_skills", vectorstore)

            st.session_state.guided_results["Missing Skills Analysis"] = {
                "answer": answer,
                "docs": source_docs,
                "file_name": "missing_skills_analysis.txt"
            }

        if st.button("Interview Prep Questions"):
            with st.spinner("Generating interview prep questions..."):
                answer, source_docs = run_guided_feature("interview_questions", vectorstore)

            st.session_state.guided_results["Interview Prep Questions"] = {
                "answer": answer,
                "docs": source_docs,
                "file_name": "interview_prep_questions.txt"
            }


        if st.button("Generate Cover Letter"):
            with st.spinner("Writing tailored cover letter..."):
                answer, source_docs = run_guided_feature("cover_letter", vectorstore)

            st.session_state.guided_results["Cover Letter"] = {
                "answer": answer,
                "docs": source_docs,
                "file_name": "cover_letter.txt"
            }

    with col2:
        if st.button("Improve Resume Bullets"):
            with st.spinner("Improving resume bullets..."):
                answer, source_docs = run_guided_feature("resume_bullets", vectorstore)

            st.session_state.guided_results["Improved Resume Bullets"] = {
                "answer": answer,
                "docs": source_docs,
                "file_name": "improved_resume_bullets.txt"
            }

        if st.button("Create Good Fit Summary"):
            with st.spinner("Creating good fit summary..."):
                answer, source_docs = run_guided_feature("good_fit_summary", vectorstore)

            st.session_state.guided_results["Good Fit Summary"] = {
                "answer": answer,
                "docs": source_docs,
                "file_name": "good_fit_summary.txt"
            }

        


    # =====================================================
    # GUIDED TOOL RESULTS SECTION
    # Keeps generated guided outputs visible in dropdowns
    # =====================================================

    if st.session_state.guided_results:
        st.divider()
        st.subheader("Guided Tool Results")

        for title, data in st.session_state.guided_results.items():
            with st.expander(f"View {title}", expanded=True):
                st.write(data["answer"])

                st.download_button(
                    label=f"Download {title}",
                    data=generate_download_text(title, data["answer"]),
                    file_name=data["file_name"],
                    mime="text/plain"
                )

                with st.expander(f"View Source Chunks for {title}"):
                    for i, doc in enumerate(data["docs"], start=1):
                        source = doc.metadata.get("source", "Unknown")

                        st.markdown(f"### Source Chunk {i}")
                        st.write(f"**Source:** {source}")
                        st.write(doc.page_content)

else:
    st.info("Please provide both a resume and a job description using either PDF upload or pasted text.") 
