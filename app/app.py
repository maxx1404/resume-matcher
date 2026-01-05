# app/streamlit_app.py
import streamlit as st
import json
from pathlib import Path
from resume_utils import extract_text_from_pdf, preprocess_text
from semantic_kernel_setup import build_kernel, register_skills
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
import asyncio
import os
import openai
from dotenv import load_dotenv
load_dotenv()

# -----------------------------
# Streamlit page setup
# -----------------------------
st.set_page_config(page_title="Resume Analyser", layout="wide")
st.title("Resume Analyzer — Semantic Kernel Resume Analyzer (MVP)")

st.sidebar.header("Settings")
st.sidebar.markdown("Fill `.env` before running. Use small dev models for low cost.")

uploaded = st.file_uploader("Upload resume (PDF)", type=["pdf"])
jd_text = st.text_area("Paste Job Description (or upload JD text file)", height=250)

# -----------------------------
# Helper: OpenAI Chat (no Azure needed)
# -----------------------------
def chat_complete(system_prompt, user_prompt, temperature=0.2, max_tokens=400):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    res = openai.ChatCompletion.create(
        model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return res.choices[0].message.content

# -----------------------------
# Main app logic
# -----------------------------
if uploaded is not None and jd_text.strip():
    try:
        # Extract resume text
        resume_bytes = uploaded.read()
        resume_text = extract_text_from_pdf(resume_bytes)
        st.subheader("Resume preview (first 1200 chars)")
        st.code(resume_text[:1200] + ("\n..." if len(resume_text) > 1200 else ""))

        with st.spinner("Building Semantic Kernel and analyzing..."):
            # Build kernel and register skills
            kernel = build_kernel()
            extract_fn, match_fn, rewrite_fn = register_skills(kernel)
            
            # -----------------------------
            # 1. Extract skills
            # -----------------------------
            args = KernelArguments(resume=resume_text)
            ext_res = asyncio.run(kernel.invoke(extract_fn, args))
            skills_out = str(ext_res)

            st.markdown("**Extracted Skills:**")
            st.write(skills_out)
            
            # -----------------------------
            # 2. Compute match score
            # -----------------------------
            match_args = KernelArguments(resume=resume_text, job=jd_text)
            match_res = asyncio.run(kernel.invoke(match_fn, match_args))
            match_raw = str(match_res)
            
            try:
                match_json = json.loads(match_raw)
            except Exception:
                match_json = {"match_score": "N/A", "missing_skills": [], "recommended_learning": []}
                st.warning("Match skill returned non-JSON response; showing raw output below.")
                st.text(match_raw)

            st.metric("Fit Score", match_json.get("match_score", "N/A"))
            st.subheader("Missing Skills (top)")
            st.write(match_json.get("missing_skills", []))
            st.subheader("Recommended Learning")
            st.write(match_json.get("recommended_learning", []))

            # -----------------------------
            # 3. Rewrite sample bullets
            # -----------------------------
            sample_bullets = "Built an ML model. Created APIs. Wrote scripts to clean data."
            rewrite_args = KernelArguments(bullets=sample_bullets, job=jd_text)
            rewrite_res = asyncio.run(kernel.invoke(rewrite_fn, rewrite_args))
            rewrite_raw = str(rewrite_res)
            
            try:
                rewjson = json.loads(rewrite_raw)
                rewrites = rewjson.get("rewrites", [])
            except Exception:
                rewrites = []
                st.warning("Rewrite skill returned non-JSON output; raw output shown below.")
                st.text(rewrite_raw)

            st.subheader("Example Rewritten Bullets")
            if rewrites:
                for r in rewrites:
                    st.write("- " + r)
            else:
                st.write("No rewrites produced. You can paste sample bullets above and re-run.")

            # -----------------------------
            # 4. Generate final ATS summary via chat
            # -----------------------------
            sys_prompt = "You are an experienced hiring manager and resume advisor. Produce a concise ATS-optimized summary (4 lines) and 3 actionable suggestions for improvement."
            user_prompt = f"Resume:\n{resume_text[:4000]}\n\nJob:\n{jd_text[:2000]}"
            summary = chat_complete(sys_prompt, user_prompt)
            st.subheader("AI Summary & Suggestions")
            st.write(summary)

    except Exception as e:
        st.error(f"Error during analysis: {e}")
        import traceback
        st.code(traceback.format_exc())
else:
    st.info("Upload a resume (PDF) and paste a job description to analyze.")