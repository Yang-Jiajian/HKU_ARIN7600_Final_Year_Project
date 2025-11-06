import os
from flask import current_app
from typing import Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage



def _get_llm_config():
    """Read LLM config from Flask config or environment variables."""
    app_config = getattr(current_app, "config", {})
    api_base = app_config.get("LLM_API_BASE") or os.getenv("LLM_API_BASE")
    api_key = app_config.get("LLM_API_KEY") or os.getenv("LLM_API_KEY")
    model = app_config.get("LLM_MODEL") or os.getenv("LLM_MODEL")
    return api_base, api_key, model


def _build_chat_model(api_base: str, api_key: str, model: str) -> Tuple[object, str]:
    try:
        llm = ChatOpenAI(
            api_key=api_key,
            model=model,
            base_url=api_base,  # supports OpenAI-compatible providers
            temperature=0.9,
            max_tokens=200,
            timeout=30,
        )
        return llm, ""
    except Exception as e:  # initialization error
        return None, str(e)


def generate_ielts_prompt():
    """Use LangChain ChatOpenAI to generate a random IELTS Writing Task 2 topic."""
    api_base, api_key, model = _get_llm_config()

    if not api_base or not api_key or not model:
        return {
            "error": "LLM configuration missing",
            "detail": "Please set LLM_API_BASE, LLM_API_KEY, and LLM_MODEL",
        }, 500

    llm, init_error = _build_chat_model(api_base, api_key, model)
    if llm is None:
        return {"error": "Failed to initialize LangChain ChatOpenAI", "detail": init_error}, 500

    system_prompt = "You are a helpful assistant that creates IELTS Writing Task."
    user_prompt = (
        "Generate ONE realistic IELTS Writing Task to essay question. "
        "Vary topic randomly (e.g., education, technology, environment, health, culture, work). "
        "Return only the question text in English, without extra commentary."
    )

    try:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        response = llm.invoke(messages)
        content = (getattr(response, "content", "") or "").strip()
        if not content:
            return {"error": "Empty response from LLM"}, 502
        return {"assistant": content}
    except Exception as e:
        return {
            "error": "Failed to call LLM via LangChain",
            "detail": str(e),
        }, 502


def evaluate_ielts_essay(topic: str, essay: str):
    """Score and review an IELTS Task 2 essay using IELTS official criteria.
    Returns a structured JSON with overall band, breakdown, and actionable advice.
    """
    api_base, api_key, model = _get_llm_config()
    if not api_base or not api_key or not model:
        return {
            "error": "LLM configuration missing",
            "detail": "Please set LLM_API_BASE, LLM_API_KEY, and LLM_MODEL",
        }, 500

    llm, init_error = _build_chat_model(api_base, api_key, model)
    if llm is None:
        return {"error": "Failed to initialize LangChain ChatOpenAI", "detail": init_error}, 500

    system = (
        "You are an IELTS Writing Task 2 examiner. Evaluate essays using official band descriptors: "
        "Task Response, Coherence and Cohesion, Lexical Resource, Grammatical Range and Accuracy. "
        "Provide a fair score 0-9 (increments of 0.5 allowed), a breakdown per criterion, concise strengths, "
        "weaknesses, and prioritized suggestions. Return STRICT JSON with keys: "
        "overall_band (number), breakdown (object with task_response, coherence_cohesion, lexical_resource, grammatical_range_accuracy), "
        "strengths (array of strings), weaknesses (array of strings), suggestions (array of strings)."
    )
    user = (
        f"Prompt: {topic}\n\nEssay:\n{essay}\n\nReturn JSON only."
    )

    try:
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        response = llm.invoke(messages)
        content = (getattr(response, "content", "") or "").strip()
        if not content:
            return {"error": "Empty response from LLM"}, 502

        # Try to parse JSON; if fails, return raw assistant text
        import json
        try:
            data = json.loads(content)
            return data
        except Exception:
            return {"assistant": content}
    except Exception as e:
        return {"error": "Failed to evaluate essay", "detail": str(e)}, 502


def continue_ielts_conversation(conversation: list):
    """Continue the IELTS feedback conversation with provided context messages.
    conversation: list of {role: 'system'|'user'|'assistant', content: str}
    """
    api_base, api_key, model = _get_llm_config()
    if not api_base or not api_key or not model:
        return {
            "error": "LLM configuration missing",
            "detail": "Please set LLM_API_BASE, LLM_API_KEY, and LLM_MODEL",
        }, 500

    llm, init_error = _build_chat_model(api_base, api_key, model)
    if llm is None:
        return {"error": "Failed to initialize LangChain ChatOpenAI", "detail": init_error}, 500

    # Ensure there is a guiding system prompt to keep the assistant as an IELTS coach
    default_system = SystemMessage(content=(
        "You are an IELTS Writing Task 2 coach. Answer follow-up questions succinctly, "
        "reference previous feedback when relevant, and provide concrete examples."
    ))

    lc_messages = [default_system]
    for m in conversation or []:
        role = (m or {}).get("role", "").strip()
        content = (m or {}).get("content", "")
        if not content:
            continue
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "assistant":
            # LangChain's AIMessage would be ideal, but ChatOpenAI accepts plain strings via invoke as well
            from langchain_core.messages import AIMessage
            lc_messages.append(AIMessage(content=content))
        else:
            lc_messages.append(HumanMessage(content=content))

    try:
        response = llm.invoke(lc_messages)
        content = (getattr(response, "content", "") or "").strip()
        if not content:
            return {"error": "Empty response from LLM"}, 502
        return {"assistant": content}
    except Exception as e:
        return {"error": "Failed to continue conversation", "detail": str(e)}, 502
