import os
import streamlit as st
from pypdf import PdfReader
from google import genai
from google.genai import types

# Page setup
st.set_page_config(
    page_title="ATS Resume Checker & Analyzer",
    page_icon="📄",
    layout="wide"
)

# Initialize Gemini Client
def get_gemini_client(api_key: str):
    return genai.Client(api_key=api_key)

# Function to extract text from uploaded PDF
def extract_pdf_text(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    extracted_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            extracted_text += text + "\n"
    return extracted_text.strip()

# Function to analyze resume with Gemini Flash
def analyze_resume(client, resume_text: str, job_description: str) -> str:
    system_instruction = """
    You are an expert HR Specialist and Application Tracking System (ATS) auditor.
    Analyze the provided resume against the provided target job description.
    
    Structure your response strictly as follows using Markdown:
    
    ## 🎯 ATS Compatibility Score
    - **Score**: [Provide a percentage score, e.g., 75%]
    - **Verdict**: [Brief 1-sentence evaluation]
    
    ## 🔍 Key Missing Keywords
    - List top missing technical/soft skills or keywords from the job description.
    
    ## 💡 Strengths
    - Key strong points in the resume that match the role.
    
    ## 🛠️ Required Improvements
    - Bullet points detailing actionable suggestions to fix formatting, phrasing, or content gaps.
    
    ## 📝 Tailored Executive Summary Suggestion
    - Provide a short 2-3 sentence tailored resume summary the candidate can use.
    """

    prompt = f"""
    TARGET JOB DESCRIPTION:
    {job_description}

    CANDIDATE RESUME:
    {resume_text}
    """

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.2, # Lower temperature for analytical evaluation
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=config
    )
    return response.text

# --- UI Interface ---
st.title("📄 AI ATS Resume Checker & Analyzer")
st.write("Upload your resume and paste a job description to get an instant ATS score and tailored improvements.")

# API Key handling via Sidebar or Streamlit Secrets
with st.sidebar:
    st.header("🔑 Configuration")
    api_key_input = st.text_input(
        "Gemini API Key", 
        type="password",
        help="Get your key at https://aistudio.google.com/"
    )
    
    # Check if API key is set via secrets or manual input
    api_key = api_key_input or st.secrets.get("GEMINI_API_KEY", "")
    
    if not api_key:
        st.warning("Please enter your Gemini API Key to proceed.")
    else:
        st.success("API Key detected!")

# Main Columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Upload Resume")
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
    
    st.subheader("2. Target Job Description")
    job_description = st.text_area(
        "Paste the Job Description here", 
        height=250,
        placeholder="Paste requirements, skills, and qualifications..."
    )

with col2:
    st.subheader("3. ATS Evaluation Output")
    analyze_btn = st.button("🚀 Analyze Resume", type="primary", use_container_width=True)

    if analyze_btn:
        if not api_key:
            st.error("Missing API key. Please add it in the sidebar or Streamlit Secrets.")
        elif not uploaded_file:
            st.error("Please upload a PDF resume.")
        elif not job_description.strip():
            st.error("Please provide a target job description.")
        else:
            with st.spinner("Extracting resume text and running ATS analysis..."):
                try:
                    resume_text = extract_pdf_text(uploaded_file)
                    
                    if not resume_text:
                        st.error("Could not read text from the uploaded PDF. Make sure it is not a scanned image PDF.")
                    else:
                        client = get_gemini_client(api_key)
                        analysis_result = analyze_resume(client, resume_text, job_description)
                        st.markdown(analysis_result)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
