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

# Load environment variables from a local .env file when running locally.
load_dotenv()

# Get OpenAI API key from Streamlit Secrets first, then fall back to local .env.
api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

# Create the OpenAI client used for all chat completion calls.
client = OpenAI(api_key=api_key)


# =====================================================
# STREAMLIT PAGE CONFIGURATION
# =====================================================

# Configure the Streamlit browser tab title and page layout.
st.set_page_config(
    page_title="A³ - AI Application Assistant",
    layout="wide"
)

# Custom CSS to improve spacing, buttons, upload boxes, and expander styling.
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

# Main app title and description.
st.title("A³ - AI Application Assistant")
st.caption("Analyze • Prepare • Succeed")
st.write("Upload or paste your resume and job description to begin.")


# =====================================================
# SESSION STATE SETUP
# =====================================================

# Stores the Ask AAA chat history.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Stores the most recent Job Match Analysis result.
if "job_match_result" not in st.session_state:
    st.session_state.job_match_result = None

# Stores the source chunks used for the most recent Job Match Analysis.
if "job_match_docs" not in st.session_state:
    st.session_state.job_match_docs = []

# Stores outputs from Guided Career Tools.
if "guided_results" not in st.session_state:
    st.session_state.guided_results = {}

# Tracks the current resume + job description pair.
# This helps detect when the user changes either input.
if "current_input_key" not in st.session_state:
    st.session_state.current_input_key = ""

# Stores the resume that the user officially added by clicking Add Resume.
if "active_resume_text" not in st.session_state:
    st.session_state.active_resume_text = ""

# Stores the job description that the user officially added by clicking Add Job Description.
if "active_job_text" not in st.session_state:
    st.session_state.active_job_text = ""


# =====================================================
# INPUT PROCESSING FUNCTIONS
# =====================================================

def extract_pdf_text(uploaded_file):
    # Read the uploaded PDF file.
    reader = PdfReader(uploaded_file)

    # Store extracted text from all pages.
    text = ""

    # Extract text from each page in the PDF.
    for page in reader.pages:
        page_text = page.extract_text()

        # Only add text if the page contains readable text.
        if page_text:
            text += page_text + " "

    # Replace multiple spaces/newlines with a single space.
    text = re.sub(r"\s+", " ", text)

    # Return cleaned PDF text.
    return text.strip()


def clean_text(text):
    # Normalize spacing in pasted or extracted text.
    text = re.sub(r"\s+", " ", text)

    # Remove leading and trailing spaces.
    return text.strip()


def get_resume_input():
    # Display resume input section title.
    st.subheader("Resume Input")

    # Let the user choose between uploading a PDF or pasting text.
    resume_input_type = st.radio(
        "Choose resume input type",
        ["Upload PDF", "Paste Text"],
        horizontal=True,
        key="resume_input_type"
    )

    # Default empty resume text.
    resume_text = ""

    if resume_input_type == "Upload PDF":
        # Allow the user to upload a resume PDF.
        resume_file = st.file_uploader(
            "Upload Resume PDF",
            type=["pdf"],
            key="resume_pdf"
        )

        # Extract text if a PDF was uploaded.
        if resume_file:
            resume_text = extract_pdf_text(resume_file)

    else:
        # Allow the user to paste resume text manually.
        resume_text = st.text_area(
            "Paste Resume Text",
            height=250,
            key="resume_text_area"
        )

    # Return cleaned resume text.
    return clean_text(resume_text)


def get_job_input():
    # Display job description input section title.
    st.subheader("Job Description Input")

    # Let the user choose between uploading a PDF or pasting text.
    job_input_type = st.radio(
        "Choose job description input type",
        ["Upload PDF", "Paste Text"],
        horizontal=True,
        key="job_input_type"
    )

    # Default empty job description text.
    job_text = ""

    if job_input_type == "Upload PDF":
        # Allow the user to upload a job description PDF.
        job_file = st.file_uploader(
            "Upload Job Description PDF",
            type=["pdf"],
            key="job_pdf"
        )

        # Extract text if a PDF was uploaded.
        if job_file:
            job_text = extract_pdf_text(job_file)

    else:
        # Allow the user to paste job description text manually.
        job_text = st.text_area(
            "Paste Job Description Text",
            height=250,
            key="job_text_area"
        )

    # Return cleaned job description text.
    return clean_text(job_text)


# =====================================================
# RAG VECTORSTORE CREATION
# =====================================================

def create_rag_vectorstore(resume_text, job_text):
    # Split resume and job description into smaller chunks for retrieval.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    # Create chunks from resume text.
    resume_chunks = splitter.split_text(resume_text)

    # Create chunks from job description text.
    job_chunks = splitter.split_text(job_text)

    # Stores all text chunks that will be embedded.
    texts = []

    # Stores metadata showing whether each chunk came from Resume or Job Description.
    metadatas = []

    # Add resume chunks with Resume metadata.
    for chunk in resume_chunks:
        if chunk.strip():
            texts.append(chunk)
            metadatas.append({"source": "Resume"})

    # Add job description chunks with Job Description metadata.
    for chunk in job_chunks:
        if chunk.strip():
            texts.append(chunk)
            metadatas.append({"source": "Job Description"})

    # If no valid chunks exist, return None so the app can stop safely.
    if not texts:
        return None

    # Create OpenAI embeddings for storing and searching chunks.
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key
    )

    # IMPORTANT:
    # A unique collection name prevents ChromaDB from reusing old resume/job data.
    # This fixes the issue where deployed Streamlit apps may return old job requirements.
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name=f"aaa_{uuid.uuid4().hex}"
    )

    # Return the newly created vectorstore.
    return vectorstore


# =====================================================
# RETRIEVAL FUNCTIONS
# =====================================================

def remove_duplicate_docs(docs):
    # Stores unique documents only.
    unique_docs = []

    # Tracks text that has already been included.
    seen_texts = set()

    # Remove duplicate retrieved chunks.
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

    # Expand the resume-side query so retrieval focuses on candidate background.
    resume_query = question + " candidate resume skills experience projects education technical skills"

    # Expand the job-side query so retrieval focuses on job requirements.
    job_query = question + " job description required skills qualifications responsibilities technologies"

    # Create embeddings for the user's question.
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key
    )

    def chroma_direct_search(query, source_name, top_k=3):
        # Convert the search query into an embedding vector.
        query_embedding = embeddings.embed_query(query)

        # Search the Chroma collection directly.
        # n_results is set higher than top_k so we can filter by source afterward.
        results = vectorstore._collection.query(
            query_embeddings=[query_embedding],
            n_results=20,
            include=["documents", "metadatas"]
        )

        # Stores matching documents from the requested source.
        docs = []

        # Chroma returns nested lists, so we grab the first result list.
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        # Convert matching Chroma results into LangChain Document objects.
        for text, metadata in zip(documents, metadatas):
            # Only keep chunks from the requested source.
            if metadata and metadata.get("source") == source_name:
                docs.append(
                    Document(
                        page_content=text,
                        metadata=metadata
                    )
                )

            # Stop after collecting enough relevant chunks.
            if len(docs) == top_k:
                break

        return docs

    # Retrieve top resume-related chunks.
    resume_docs = chroma_direct_search(resume_query, "Resume", top_k=3)

    # Retrieve top job-description-related chunks.
    job_docs = chroma_direct_search(job_query, "Job Description", top_k=3)

    # Combine both sources and remove duplicates.
    return remove_duplicate_docs(resume_docs + job_docs)


def retrieve_match_context(vectorstore):
    # Multiple retrieval queries help gather a broader match-analysis context.
    queries = [
        "required skills qualifications responsibilities job requirements candidate matching skills experience",
        "technical skills programming languages frameworks cloud aws frontend backend database ai",
        "projects experience internships education certifications relevant work",
        "gaps missing skills weaknesses required technologies not shown in resume",
        "production support client facing debugging documentation"
    ]

    # Stores all retrieved chunks from all queries.
    all_docs = []

    # Retrieve relevant chunks for each match-analysis query.
    for query in queries:
        all_docs.extend(retrieve_docs(query, vectorstore))

    # Remove duplicate chunks before sending them to the LLM.
    return remove_duplicate_docs(all_docs)


def docs_to_context(docs):
    # Convert retrieved documents into readable text for the LLM prompt.
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
    # Retrieve resume and job description chunks for job match analysis.
    docs = retrieve_match_context(vectorstore)

    # Convert retrieved chunks into prompt context.
    context = docs_to_context(docs)

    # Prompt forces the model to use only retrieved context and follow a fixed format.
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

    # Send the RAG-grounded prompt to OpenAI.
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

    # Return the generated analysis and the source chunks used.
    return response.choices[0].message.content, docs


def generate_rag_answer(question, vectorstore, include_chat_history=True):
    # Retrieve context relevant to the user's question.
    docs = retrieve_docs(question, vectorstore)

    # Convert retrieved chunks into prompt-ready text.
    context = docs_to_context(docs)

    # Default to no previous chat unless enabled.
    previous_chat = ""

    if include_chat_history:
        # Include only the last 3 chat exchanges to keep the prompt focused.
        previous_chat = "\n".join(
            [
                f"User: {chat['question']}\nAAA: {chat['answer']}"
                for chat in st.session_state.chat_history[-3:]
            ]
        )

    # Main prompt for Ask AAA and Guided Career Tools.
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

    # Send prompt to OpenAI for a grounded answer.
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

    # Return answer and source chunks.
    return response.choices[0].message.content, docs


def ask_aaa(question, vectorstore):
    # Generate an answer using RAG and include chat history.
    answer, docs = generate_rag_answer(
        question=question,
        vectorstore=vectorstore,
        include_chat_history=True
    )

    # Save the question and answer in session state.
    st.session_state.chat_history.append(
        {
            "question": question,
            "answer": answer
        }
    )

    # Return answer and source chunks.
    return answer, docs


# =====================================================
# GUIDED CAREER TOOLS
# =====================================================

def run_guided_feature(feature_name, vectorstore):
    # Prompts for each guided career feature.
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

    # Select the correct prompt based on the clicked feature button.
    question = feature_prompts[feature_name]

    # Generate an answer without chat history so guided outputs stay independent.
    answer, docs = generate_rag_answer(
        question=question,
        vectorstore=vectorstore,
        include_chat_history=False
    )

    # Return guided feature output and retrieved chunks.
    return answer, docs


# =====================================================
# DOWNLOAD HELPER
# =====================================================

def generate_download_text(title, content):
    # Format downloaded text files with a title at the top.
    return f"""
{title}

{content}
"""


# =====================================================
# STREAMLIT USER INTERFACE
# =====================================================

# Create two columns: one for resume input and one for job description input.
left_col, right_col = st.columns(2)

with left_col:
    # Get the current resume input from upload or pasted text.
    new_resume_text = get_resume_input()

    # User must click Add Resume to make this resume the active resume.
    if st.button("Add Resume"):
        if new_resume_text:
            # Replace the previous active resume with the new one.
            st.session_state.active_resume_text = new_resume_text

            # Clear old outputs because the resume changed.
            st.session_state.chat_history = []
            st.session_state.job_match_result = None
            st.session_state.job_match_docs = []
            st.session_state.guided_results = {}

            # Reset input key so the app rebuilds context on rerun.
            st.session_state.current_input_key = ""

            # Rerun the app so the new resume becomes active immediately.
            st.rerun()
        else:
            # Warn user if they click Add Resume without providing text/PDF.
            st.warning("Please upload or paste a resume before clicking Add Resume.")

with right_col:
    # Get the current job description input from upload or pasted text.
    new_job_text = get_job_input()

    # User must click Add Job Description to make this job description active.
    if st.button("Add Job Description"):
        if new_job_text:
            # Replace the previous active job description with the new one.
            st.session_state.active_job_text = new_job_text

            # Clear old outputs because the job description changed.
            st.session_state.chat_history = []
            st.session_state.job_match_result = None
            st.session_state.job_match_docs = []
            st.session_state.guided_results = {}

            # Reset input key so the app rebuilds context on rerun.
            st.session_state.current_input_key = ""

            # Rerun the app so the new job description becomes active immediately.
            st.rerun()
        else:
            # Warn user if they click Add Job Description without providing text/PDF.
            st.warning("Please upload or paste a job description before clicking Add Job Description.")


# Use only the officially added resume/job description for analysis.
resume_text = st.session_state.active_resume_text
job_text = st.session_state.active_job_text


# Show resume preview if an active resume exists.
if resume_text:
    with st.expander("View Resume Preview"):
        st.text_area("Resume Content Preview", resume_text, height=200)

# Show job description preview if an active job description exists.
if job_text:
    with st.expander("View Job Description Preview"):
        st.text_area("Job Description Content Preview", job_text, height=200)


# Only build the RAG system when both resume and job description exist.
if resume_text and job_text:

    # Create a key representing the current resume + job description pair.
    current_input_key = str(hash(resume_text + job_text))

    # If the active inputs changed, clear old results and update the key.
    if current_input_key != st.session_state.current_input_key:
        st.session_state.chat_history = []
        st.session_state.job_match_result = None
        st.session_state.job_match_docs = []
        st.session_state.guided_results = {}
        st.session_state.current_input_key = current_input_key

    st.divider()

    # Build a fresh Chroma vectorstore for the current resume and job description.
    with st.spinner("Building RAG system..."):
        vectorstore = create_rag_vectorstore(resume_text, job_text)

    # Stop the app if no valid chunks were created.
    if vectorstore is None:
        st.error("Could not build RAG system because no text chunks were created.")
        st.stop()

    # Confirm that the RAG system is ready.
    st.success("RAG system ready.")


    # =====================================================
    # JOB MATCH ANALYSIS SECTION
    # =====================================================

    st.subheader("Job Match Analysis")

    # Run job match analysis only when the user clicks the button.
    if st.button("Analyze Job Match"):
        with st.spinner("Running RAG-powered job match analysis..."):
            result, source_docs = analyze_job_match(vectorstore)

        # Save result and source chunks in session state.
        st.session_state.job_match_result = result
        st.session_state.job_match_docs = source_docs

    # Display the saved job match analysis if it exists.
    if st.session_state.job_match_result:
        with st.expander("View Job Match Analysis", expanded=True):
            st.write(st.session_state.job_match_result)

            # Allow user to download the analysis as a text file.
            st.download_button(
                label="Download Job Match Analysis",
                data=generate_download_text(
                    "Job Match Analysis",
                    st.session_state.job_match_result
                ),
                file_name="job_match_analysis.txt",
                mime="text/plain"
            )

        # Show retrieved chunks used for transparency.
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

    # Input box for asking custom questions about the resume and job description.
    question = st.text_input(
        "Ask a question about your resume and this job description"
    )

    # Generate answer whenever a question is entered.
    if question:
        with st.spinner("AAA is thinking..."):
            answer, source_docs = ask_aaa(question, vectorstore)

        # Display the answer.
        st.write(answer)

        # Show retrieved chunks used to answer the question.
        with st.expander("View Retrieved Source Chunks"):
            for i, doc in enumerate(source_docs, start=1):
                source = doc.metadata.get("source", "Unknown")

                st.markdown(f"### Source Chunk {i}")
                st.write(f"**Source:** {source}")
                st.write(doc.page_content)

    # Display previous Ask AAA conversation history.
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

    # Split guided tools into two columns.
    col1, col2 = st.columns(2)

    with col1:
        # Run missing skills analysis.
        if st.button("Matching and Missing Skills Analysis"):
            with st.spinner("Analyzing missing skills..."):
                answer, source_docs = run_guided_feature("missing_skills", vectorstore)

            # Save result so it remains visible inside an expander.
            st.session_state.guided_results["Missing Skills Analysis"] = {
                "answer": answer,
                "docs": source_docs,
                "file_name": "missing_skills_analysis.txt"
            }

        # Generate interview preparation questions.
        if st.button("Interview Prep Questions"):
            with st.spinner("Generating interview prep questions..."):
                answer, source_docs = run_guided_feature("interview_questions", vectorstore)

            # Save result so it remains visible inside an expander.
            st.session_state.guided_results["Interview Prep Questions"] = {
                "answer": answer,
                "docs": source_docs,
                "file_name": "interview_prep_questions.txt"
            }

        # Generate tailored cover letter.
        if st.button("Generate Cover Letter"):
            with st.spinner("Writing tailored cover letter..."):
                answer, source_docs = run_guided_feature("cover_letter", vectorstore)

            # Save result so it remains visible inside an expander.
            st.session_state.guided_results["Cover Letter"] = {
                "answer": answer,
                "docs": source_docs,
                "file_name": "cover_letter.txt"
            }

    with col2:
        # Generate improved resume bullets.
        if st.button("Improve Resume Bullets"):
            with st.spinner("Improving resume bullets..."):
                answer, source_docs = run_guided_feature("resume_bullets", vectorstore)

            # Save result so it remains visible inside an expander.
            st.session_state.guided_results["Improved Resume Bullets"] = {
                "answer": answer,
                "docs": source_docs,
                "file_name": "improved_resume_bullets.txt"
            }

        # Generate good fit summary.
        if st.button("Create Good Fit Summary"):
            with st.spinner("Creating good fit summary..."):
                answer, source_docs = run_guided_feature("good_fit_summary", vectorstore)

            # Save result so it remains visible inside an expander.
            st.session_state.guided_results["Good Fit Summary"] = {
                "answer": answer,
                "docs": source_docs,
                "file_name": "good_fit_summary.txt"
            }


    # =====================================================
    # GUIDED TOOL RESULTS SECTION
    # =====================================================

    # Display all guided tool results that have been generated.
    if st.session_state.guided_results:
        st.divider()
        st.subheader("Guided Tool Results")

        # Loop through each saved guided result.
        for title, data in st.session_state.guided_results.items():
            with st.expander(f"View {title}", expanded=True):
                # Display generated result.
                st.write(data["answer"])

                # Allow user to download each guided tool result.
                st.download_button(
                    label=f"Download {title}",
                    data=generate_download_text(title, data["answer"]),
                    file_name=data["file_name"],
                    mime="text/plain"
                )

                # Show source chunks for transparency.
                with st.expander(f"View Source Chunks for {title}"):
                    for i, doc in enumerate(data["docs"], start=1):
                        source = doc.metadata.get("source", "Unknown")

                        st.markdown(f"### Source Chunk {i}")
                        st.write(f"**Source:** {source}")
                        st.write(doc.page_content)

else:
    # Show message until both resume and job description are added.
    st.info("Please provide both a resume and a job description using either PDF upload or pasted text.")
