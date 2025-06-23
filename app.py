import streamlit as st
from matcher import extract_text_from_resume, compute_similarity, missing_keywords

st.set_page_config(page_title = "Resume Matcher", layout = "centered")
st.title("Resume Matcher")

uploaded_file = st.file_uploader("Upload your resume(PDF)", type="pdf")
jd_input = st.text_area("Paste your Job Description here")


if uploaded_file and jd_input:
    if st.button("Analyze Resume"):
        with st.spinner("Analysizing your resume..."):
            resume_text = extract_text_from_resume(uploaded_file)
            score = compute_similarity(resume_text, jd_input)
            missing = missing_keywords(resume_text, jd_input)
        
        st.success("Match Completed")
        st.markdown(f' Match SCORE {score:.2f}')
        if score >= 0.7:
            st.success("Strong Match")
        elif score >=0.4:
            st.success("Decent match")
        else:
            st.success("Poor Match")

        st.markdown("Top 5 missing keywrds from youre resume")
        st.code(", ".join(missing), language='markdown')

        