"""
A³ - AI Application Assistant
Author: Amanuel Tenaw
"""

import os
import re
import uuid
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from openai import OpenAI

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


# =====================================================
# ENVIRONMENT SETUP
# =====================================================

load_dotenv()

api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
client = OpenAI(api_key=api_key)


# =====================================================
# STREAMLIT PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="A³ - AI Application Assistant",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

div[data-testid="stExpander"] {
    border: 1px solid #E5E7EB;
    border-radius: 10px;
}

.stButton > button {
    width: 100%;
    border-radius: 8px;
    height: 3rem;
    font-weight: 500;
}

[data-testid="stFileUploader"] {
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 10px;
}

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
# =====================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "job_match_result" not in st.session_state:
    st.session_state.job_match_result = None

if "job_match_docs" not in st.session_state:
    st.session_state.job_match_docs = []

if "guided_results" not in st.session_state:
    st.session_state.guided_results = {}

if "current_input_key" not in st.session_state:
    st.session_state.current_input_key = ""

if "active_resume_text" not in st.session_state:
    st.session_state.active_resume_text = ""

if "active_job_text" not in st.session_state:
    st.session_state.active_job_text = ""


# =====================================================
# INPUT PROCESSING FUNCTIONS
# =====================================================

def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_resume_input():
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
# =====================================================

def create_rag_vectorstore(resume_text, job_text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    resume_chunks = splitter.split_text(resume_text)
    job_chunks = splitter.split_text(job_text)

    texts = []
    metadatas = []

    for chunk in resume_chunks:
        if chunk.strip():
            texts.append(chunk)
            metadatas.append({"source": "Resume"})

    for chunk in job_chunks:
        if chunk.strip():
            texts.append(chunk)
            metadatas.append({"source": "Job Description"})

    if not texts:
        return None

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key
    )

    # IMPORTANT:
    # Unique collection name prevents Streamlit Cloud/Chroma from reusing old job descriptions.
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name=f"aaa_{uuid.uuid4().hex}"
    )

    return vectorstore


# =====================================================
# RETRIEVAL FUNCTIONS
# =====================================================

def remove_duplicate_docs(docs):
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
    ChromaDB retrieval fix for Streamlit Cloud.

    This keeps ChromaDB but avoids broken LangChain wrapper methods like:
    - vectorstore.similarity_search()
    - vectorstore.max_marginal_relevance_search()

    Instead, it embeds the query and searches the Chroma collection directly.
    """

    resume_query = question + " candidate resume skills experience projects education technical skills"
    job_query = question + " job description required skills qualifications responsibilities technologies"

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key
    )

    def chroma_direct_search(query, source_name, top_k=3):
        query_embedding = embeddings.embed_query(query)

        results = vectorstore._collection.query(
            query_embeddings=[query_embedding],
            n_results=20,
            include=["documents", "metadatas"]
        )

        docs = []

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for text, metadata in zip(documents, metadatas):
            if metadata and metadata.get("source") == source_name:
                docs.append(
                    Document(
                        page_content=text,
                        metadata=metadata
                    )
                )

            if len(docs) == top_k:
                break

        return docs

    resume_docs = chroma_direct_search(resume_query, "Resume", top_k=3)
    job_docs = chroma_direct_search(job_query, "Job Description", top_k=3)

    return remove_duplicate_docs(resume_docs + job_docs)


def retrieve_match_context(vectorstore):
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
    return "\n\n".join(
        [
            f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
            for doc in docs
        ]
    )


# =====================================================
# LLM ANALYSIS FUNCTIONS
# =====================================================

def analyze_job_match(vectorstore):
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
# =====================================================

def run_guided_feature(feature_name, vectorstore):
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
# =====================================================

def generate_download_text(title, content):
    return f"""
{title}

{content}
"""


# =====================================================
# STREAMLIT USER INTERFACE
# =====================================================

left_col, right_col = st.columns(2)

with left_col:
    new_resume_text = get_resume_input()

    if st.button("Add Resume"):
        if new_resume_text:
            st.session_state.active_resume_text = new_resume_text

            st.session_state.chat_history = []
            st.session_state.job_match_result = None
            st.session_state.job_match_docs = []
            st.session_state.guided_results = {}

            st.session_state.current_input_key = ""
            st.rerun()
        else:
            st.warning("Please upload or paste a resume before clicking Add Resume.")

with right_col:
    new_job_text = get_job_input()

    if st.button("Add Job Description"):
        if new_job_text:
            st.session_state.active_job_text = new_job_text

            st.session_state.chat_history = []
            st.session_state.job_match_result = None
            st.session_state.job_match_docs = []
            st.session_state.guided_results = {}

            st.session_state.current_input_key = ""
            st.rerun()
        else:
            st.warning("Please upload or paste a job description before clicking Add Job Description.")


resume_text = st.session_state.active_resume_text
job_text = st.session_state.active_job_text


if resume_text:
    with st.expander("View Resume Preview"):
        st.text_area("Resume Content Preview", resume_text, height=200)

if job_text:
    with st.expander("View Job Description Preview"):
        st.text_area("Job Description Content Preview", job_text, height=200)


if resume_text and job_text:

    current_input_key = str(hash(resume_text + job_text))

    if current_input_key != st.session_state.current_input_key:
        st.session_state.chat_history = []
        st.session_state.job_match_result = None
        st.session_state.job_match_docs = []
        st.session_state.guided_results = {}
        st.session_state.current_input_key = current_input_key

    st.divider()

    with st.spinner("Building RAG system..."):
        vectorstore = create_rag_vectorstore(resume_text, job_text)

    if vectorstore is None:
        st.error("Could not build RAG system because no text chunks were created.")
        st.stop()

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
