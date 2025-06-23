import streamlit as st
import pandas as pd
from matcher import extract_text_from_resume, compute_similarity, missing_keywords

st.set_page_config(page_title = "Resume Matcher", layout = "centered")
st.title("Resume Matcher")

mode = st.radio("Choose Mode", ["Student", "Recruiter"], horizontal = True)

if mode == "Student":
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")
    jd_input = st.text_area("Paste the Job Description")

    if uploaded_file and jd_input:
        if st.button("Analyze Resume"):
            with st.spinner("Analyzing your resume..."):
                resume_text = extract_text_from_resume(uploaded_file)
                score = compute_similarity(resume_text, jd_input)
                missing = missing_keywords(resume_text, jd_input)

            st.success("Match complete!")
            st.markdown(f"** Match Score:** `{score:.2f}`")

            if score >= 0.75:
                st.success("Excellent match!")
            elif score >= 0.4:
                st.warning("Partial match")
            else:
                st.error("Poor match.")

            st.markdown("**Top 5 Missing Keywords:**")
            st.code(", ".join(missing[:5]))

elif mode == "Recruiter":
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