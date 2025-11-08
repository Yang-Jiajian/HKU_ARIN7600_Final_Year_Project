import os
import uuid
import json
import re
import csv
from typing import Tuple, Optional, List, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from openai.types.responses.response_output_message import Content

# 全局 chat model 对象
_chat_model: Optional[ChatOpenAI] = None
_initialization_error: Optional[str] = None

class ListResponse(TypedDict):
    scores: List[float]

def _build_chat_model(api_base: str, api_key: str, model: str) -> Tuple[object, str]:
    try:
        llm = ChatOpenAI(
            api_key=api_key,
            model=model,
            base_url=api_base,  # supports OpenAI-compatible providers
        )
        return llm, ""
    except Exception as e:  # initialization error
        return None, str(e)


def initialize_chat_model(app):
    """在应用启动时初始化全局 chat model。
    
    Args:
        app: Flask 应用实例
    """
    global _chat_model, _initialization_error
    
    api_base = app.config.get("LLM_API_BASE") or os.getenv("LLM_API_BASE")
    api_key = app.config.get("LLM_API_KEY") or os.getenv("LLM_API_KEY")
    model = app.config.get("LLM_MODEL") or os.getenv("LLM_MODEL")
    
    if not api_base or not api_key or not model:
        _initialization_error = "LLM configuration missing: Please set LLM_API_BASE, LLM_API_KEY, and LLM_MODEL"
        _chat_model = None
        return
    
    _chat_model, _initialization_error = _build_chat_model(api_base, api_key, model)


def get_chat_model():
    """获取全局 chat model 对象。
    
    Returns:
        Tuple[Optional[ChatOpenAI], Optional[str]]: (chat_model, error_message)
    """
    global _chat_model, _initialization_error
    return _chat_model, _initialization_error


def generate_ielts_topic(conversation_id: str, user_id: str):
    """Use LangChain ChatOpenAI to generate a random IELTS Writing Task 2 topic."""
    
    llm, init_error = get_chat_model()
    if llm is None:
        error_msg = init_error or "Chat model not initialized"
        return {"error": "Failed to initialize LangChain ChatOpenAI", "detail": error_msg}, 500

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
        # save the conversation to the history file
        message_id = str(uuid.uuid4())
        record = [
            {
                "message_id": message_id,
                "role": "bot",
                "content": content
            }
        ]
        print(record)
        save_conversation_to_history(conversation_id=conversation_id, user_id=user_id, record=record)
        return record
    except Exception as e:
        return {
            "error": "Failed to call LLM via LangChain",
            "detail": str(e),
        }, 502


def evaluate_ielts_essay(conversation:list, essay:str, conversation_id: str, user_id: str):
    """Score and review an IELTS Task 2 essay using IELTS official criteria.
    Returns a structured JSON with overall band, breakdown, and actionable advice.
    """
    llm, init_error = get_chat_model()
    if llm is None:
        error_msg = init_error or "Chat model not initialized"
        return {"error": "Failed to initialize LangChain ChatOpenAI", "detail": error_msg}, 500
    topic = conversation[0]["content"]
    system = (
        '''
        You are an IELTS Writing Task 2 examiner. Evaluate essays using the official IELTS Writing Task 2 band descriptors:
Task Response, Coherence and Cohesion, Lexical Resource, and Grammatical Range and Accuracy.

Provide:

A fair overall band score from 0 to 9 (increments of 0.5 allowed)
A detailed numerical breakdown for each criterion
Concise strengths, weaknesses, and prioritized suggestions
Return your answer STRICTLY in the following plain-text format:

**Overall Score**: [number] out of 9.0\n\n
**Breakdown**:\n\n
**Task Achievement**: [number]\n\n
**Coherence & cohesion**: [number]\n\n
**Lexical Resource**: [number]\n\n
**Grammar Range & Accuracy**: [number]\n\n

**Strengths**: 
...(list of strengths)

**Weaknesses**: 
...(list of weaknesses)

**Suggestions**:
...(list of suggestions)
Do not include extra commentary or explanatory text — only the formatted result shown above.
        '''
    )
    user = (
        f"Prompt: {topic}\n\nEssay:\n{essay}\n\n."
    )

    try:
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        content = (getattr(response, "content", "") or "").strip()
        
        if not content:
            return {"error": "Empty response from LLM"}, 503
        
        match = re.search(
            r"\*\*Breakdown\*\*:?([\s\S]*?)(?:\*\*Strengths\*\*|\*\*Weaknesses\*\*|\*\*Suggestions\*\*|$)",
            content,
            re.IGNORECASE,
        )

        breakdown_text = match.group(1).strip() if match else ""

        # Step 2️⃣ — Remove the Markdown bold formatting (**bold** → plain)
        cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", breakdown_text)

        # Step 3️⃣ — Extract float (or integer) scores after the 4 target categories
        pattern = (
            r"(?:Task Achievement|Coherence\s*&\s*cohesion|Lexical Resource|Grammar Range\s*&\s*Accuracy)"
            r"\s*:\s*([0-9]+(?:\.[0-9]+)?)"
        )

        scores = [float(x) for x in re.findall(pattern, cleaned, flags=re.IGNORECASE)]
        
        print(scores)
        with open(f"./app/data/{user_id}/writing_dashboard.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(scores + [scores[-1]])

        

        # save the conversation to the history file
        record = [
            {
                "message_id": str(uuid.uuid4()),
                "role": "user",
                "content": essay
            },
            {
                "message_id": str(uuid.uuid4()),
                "role": "bot",
                "content": content
            }
        ]
        conversation.extend(record)
        save_conversation_to_history(conversation_id=conversation_id, user_id=user_id, record=record)
        return conversation
    except Exception as e:
        return {"error": "Failed to evaluate essay", "detail": str(e)}, 502


def continue_ielts_conversation(conversation: list, query: str,conversation_id:str, user_id:str):
    """Continue the IELTS feedback conversation with provided context messages.
    conversation: list of {role: 'system'|'user'|'assistant', content: str}
    """
    llm, init_error = get_chat_model()
    if llm is None:
        error_msg = init_error or "Chat model not initialized"
        return {"error": "Failed to initialize LangChain ChatOpenAI", "detail": error_msg}, 500

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
        elif role == "bot":
            # LangChain's AIMessage would be ideal, but ChatOpenAI accepts plain strings via invoke as well
            from langchain_core.messages import AIMessage
            lc_messages.append(AIMessage(content=content))
        else:
            lc_messages.append(HumanMessage(content=content))
    lc_messages.append(HumanMessage(content=query))
    try:
        response = llm.invoke(lc_messages)
        content = (getattr(response, "content", "") or "").strip()
        if not content:
            return {"error": "Empty response from LLM"}, 502
        record = [
            {
                "message_id": str(uuid.uuid4()),
                "role": "user",
                "content": query
            },
            {
                "message_id": str(uuid.uuid4()),
                "role": "bot",
                "content": content
            }
        ]
        save_conversation_to_history(conversation_id=conversation_id, user_id=user_id, record=record)
        conversation.extend(record)
        return conversation
    except Exception as e:
        return {"error": "Failed to continue conversation", "detail": str(e)}, 502


def save_conversation_to_history(conversation_id: str, user_id: str, record: list) -> None:
    """将生成的机器人内容保存/追加到用户历史文件。
    
    - 路径：app/data/{user_id}/history.json
    - 若会话存在：向 conversation 追加一条 role=bot 的消息
    - 若会话不存在：创建新会话并写入首条消息
    - 若文件不存在：创建新文件
    - 发生异常时静默失败（打印错误），不抛出
    """

    try:
        app_dir = os.path.dirname(os.path.dirname(__file__))
        user_data_dir = os.path.join(app_dir, "data", user_id)
        os.makedirs(user_data_dir, exist_ok=True)
        history_path = os.path.join(user_data_dir, "history.json")
        print(history_path)
        # ===== 读取历史文件 =====
        history: list = []
        if os.path.exists(history_path):
            with open(history_path, "r", encoding="utf-8") as rf:
                try:
                    history = json.load(rf) or []
                except json.JSONDecodeError:
                    print(f"[WARN] history.json for user {user_id} is corrupted, resetting file.")
                    history = []
        

        if len(history) == 0:
            print(f"history length = 0")
            history.append({
                "conversation_id": conversation_id,
                "title": f"Practice {conversation_id}",
                "conversation": record
            })
        else :
            conv_is_exist = False
            for conv in history:
                if conv["conversation_id"] == conversation_id:
                    conv["conversation"].extend(record)
                    conv_is_exist = True
            if not conv_is_exist:
                history.append({
                    "conversation_id": len(history) + 1,
                    "title": f"Practice {len(history) + 1}",
                    "conversation": record
                })
        

        # ===== 写入文件 =====
        with open(history_path, "w", encoding="utf-8") as wf:
            json.dump(history, wf, ensure_ascii=False, indent=4)

    except Exception as e:
        print(f"[ERROR] Failed to save conversation history for user {user_id}: {e}")