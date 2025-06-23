import streamlit as st
import pandas as pd
from matcher import extract_text_from_resume, compute_similarity, missing_keywords

st.set_page_config(page_title = "Resume Matcher", layout = "centered")
st.title("Resume Matcher")

uploaded_files = st.file_uploader("Upload multiple resumes(PDF)", type="pdf", accept_multiple_files = True)
jd_input = st.text_area("Paste your Job Description here")


if uploaded_files and jd_input:
    if st.button("Analyze All Resumes"):
        results = []

        with st.spinner("Analyzing your resume..."):
            for uploaded_file in uploaded_files:
                resume_text = extract_text_from_resume(uploaded_file)
                score = compute_similarity(resume_text, jd_input)
                missing = missing_keywords(resume_text, jd_input)

                results.append({
                    "Name": uploaded_file.name,
                    "Score": round(score,3),
                    "Top Missing keywords": ', '.join(missing[:5])
                })
        
        df = pd.DataFrame(results).sort_values(by="Score", ascending=False)

        st.success("Analysis Completed")
        st.dataframe(df, use_container_width = True)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label = "Download Results as csv",
            data = csv,
            file_name = "recruiter_match_results.csv",
            mime= "text/csv"
        )