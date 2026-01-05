# app/semantic_kernel_setup.py
import os
from dotenv import load_dotenv
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions.kernel_arguments import KernelArguments

load_dotenv()

def build_kernel():
    kernel = sk.Kernel()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")

    chat_service = OpenAIChatCompletion(
        ai_model_id="gpt-4o-mini",
        service_id="chat",
        api_key=api_key
    )

    kernel.add_service(chat_service)
    return kernel


def register_skills(kernel: sk.Kernel):
    extract_prompt = """
    Analyze the resume text and return a comma-separated list of skills.

    Resume:
    {{$resume}}
    """

    match_prompt = """
    Return valid JSON only.

    {
      "match_score": "<0-100>",
      "missing_skills": [],
      "recommended_learning": []
    }

    Resume:
    {{$resume}}

    Job:
    {{$job}}
    """

    rewrite_prompt = """
    Return JSON only.

    { "rewrites": [] }

    Rewrite bullets to be ATS optimized.

    Bullets:
    {{$bullets}}

    Job:
    {{$job}}
    """

    extract_fn = kernel.add_function(
        function_name="extract_fn",
        plugin_name="resume",
        prompt=extract_prompt
    )

    match_fn = kernel.add_function(
        function_name="match_fn",
        plugin_name="resume",
        prompt=match_prompt
    )

    rewrite_fn = kernel.add_function(
        function_name="rewrite_fn",
        plugin_name="resume",
        prompt=rewrite_prompt
    )

    return extract_fn, match_fn, rewrite_fn