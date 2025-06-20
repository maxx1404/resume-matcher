import streamlit as st
from matcher import extract_text_from_resume, compute_similarity, missing_keywords

st.set_page_config(page_title = "Resume Matcher", layout = "centered")
st.title("Resume Matcher")

uploaded_file = st.file_uploader("Upload your resume", type="pdf")
jd_input = st.text_area("Paste your Job Description here")
jd_text = st.text_area("Paste the job description here")

if uploaded_file and jd_input:
    with st.spinner("Analysizing your resume..."):
        resume_text = extract_text_from_resume(uploaded_file)
        score = compute_similarity(resume_text, jd_input)
        missing = missing_keywords(resume_text, jd_input)
    
    st.success("Match Completed")
    st.markdown(f' Match SCORE {score:.2f}')

    st.markdown("Top 5 missing keywrds from youre resume")
    st.code(", ".join(missing), language='markdown')
