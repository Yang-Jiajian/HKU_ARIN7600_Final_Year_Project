import os
import uuid
import json
import re
import csv
import base64
from typing import Tuple, Optional, List, TypedDict, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from openai.types.responses.response_output_message import Content
from openai import OpenAI

# 全局 chat model 对象
_chat_model: Optional[ChatOpenAI] = None
_initialization_error: Optional[str] = None

# 全局多模态客户端对象
_multimodal_client: Optional[OpenAI] = None
_multimodal_initialization_error: Optional[str] = None

class ListResponse(TypedDict):
    scores: List[float]

def _build_chat_model(api_base: str, api_key: str, model: str) -> Tuple[object, str]:
    try:
        llm = ChatOpenAI(
            api_key=api_key,
            model=model,
            base_url=api_base,  # supports OpenAI-compatible providers
            temperature=1.0
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
    """Use LangChain ChatOpenAI to generate a random IELTS Writing Task 2 topic and concise tips."""
    
    llm, init_error = get_chat_model()
    if llm is None:
        error_msg = init_error or "Chat model not initialized"
        return {"error": "Failed to initialize LangChain ChatOpenAI", "detail": error_msg}, 500

    system_prompt = "You are a helpful assistant that creates IELTS Writing Task."
    user_prompt_question = (
        "Generate ONE realistic IELTS Writing Task 2 essay question. "
        "Vary topic randomly (e.g., education, technology, environment, health, culture, work). "
        "Return only the question text in English, without extra commentary."
    )

    # Ask for concise, actionable tips for the generated question
    system_prompt_tips = (
        "You are an IELTS Writing coach. Provide concise, actionable tips for Planning and Writing Task 2."
    )
    user_prompt_tips = (
        "Given the IELTS Writing Task 2 question above, provide a short guidance including:\n"
        "- A possible thesis (one-sentence position)\n"
        "- 2-3 key arguments with a brief justification each\n"
        "- Suggested structure (Intro, Body 1/2, Conclusion)\n"
        "Keep it concise and practical. Output in English using clear bullet points."
    )

    try:
        # First call: generate the question
        response_q = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt_question)])
        question = (getattr(response_q, "content", "") or "").strip()
        if not question:
            return {"error": "Empty response from LLM"}, 502

        # Second call: generate tips for the question (include the question for context)
        response_tips = llm.invoke([
            SystemMessage(content=system_prompt_tips),
            HumanMessage(content=f"Question: {question}\n\n{user_prompt_tips}")
        ])
        tips = (getattr(response_tips, "content", "") or "").strip()

        # save the conversation to the history file
        message_id = str(uuid.uuid4())
        record = [
            {
                "message_id": message_id,
                "role": "bot",
                "content": question,
                "tips": tips
            }
        ]
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
If the essay is not written in English, give 0 score.
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
            writer.writerow(scores)

        

        # save the conversation to the history file
        record = {
            "message_id": str(uuid.uuid4()),
            "role": "bot",
            "content": content
        }
        save_conversation_to_history(conversation_id=conversation_id, user_id=user_id, record=[conversation[-1],record])
        conversation.append(record)
        print(conversation)
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
        record = {
            "message_id": str(uuid.uuid4()),
            "role": "bot",
            "content": content
        }
    
        save_conversation_to_history(conversation_id=conversation_id, user_id=user_id, record=[conversation[-1],record])
        conversation.append(record)
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
        history_path = os.path.join(user_data_dir, "writing_history.json")
        print(f"history path: {history_path}")
        # ===== 读取历史文件 =====
        history: list = []
        if os.path.exists(history_path):
            with open(history_path, "r", encoding="utf-8") as rf:
                try:
                    history = json.load(rf) or []
                except json.JSONDecodeError:
                    print(f"[WARN] writing_history.json for user {user_id} is corrupted, resetting file.")
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


def initialize_multimodal_client(app):
    """在应用启动时初始化全局多模态客户端（阿里云百炼）。
    
    Args:
        app: Flask 应用实例
    """
    global _multimodal_client, _multimodal_initialization_error
    
    api_key = app.config.get("DASHSCOPE_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    api_base = app.config.get("DASHSCOPE_API_BASE") or os.getenv("DASHSCOPE_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    if not api_key:
        _multimodal_initialization_error = "DASHSCOPE_API_KEY configuration missing"
        _multimodal_client = None
        return
    
    try:
        _multimodal_client = OpenAI(
            api_key=api_key,
            base_url=api_base,
        )
        _multimodal_initialization_error = None
    except Exception as e:
        _multimodal_initialization_error = str(e)
        _multimodal_client = None


def get_multimodal_client():
    """获取全局多模态客户端对象。
    
    Returns:
        Tuple[Optional[OpenAI], Optional[str]]: (multimodal_client, error_message)
    """
    global _multimodal_client, _multimodal_initialization_error
    return _multimodal_client, _multimodal_initialization_error


def process_ielts_speaking_task(
    audio_base64: str,
    task_number: int,
    conversation_id: str,
    user_id: str,
    conversation: Optional[List[Dict[str, Any]]] = None,
    return_audio: bool = True,
    audio_format: str = "wav"
) -> Tuple[Dict[str, Any], int]:
    """处理雅思口语任务（Task 1, 2, 或 3）。
    
    Args:
        audio_base64: Base64编码的音频数据（不包含data:;base64,前缀）
        task_number: 任务编号（1, 2, 或 3）
        conversation_id: 会话ID
        user_id: 用户ID
        conversation: 已有的对话历史
        return_audio: 是否返回音频响应
        audio_format: 音频格式（wav, mp3, webm 等）
    
    Returns:
        Tuple[Dict[str, Any], int]: (响应数据, HTTP状态码)
    """
    client, init_error = get_multimodal_client()
    if client is None:
        error_msg = init_error or "Multimodal client not initialized"
        return {"error": "Failed to initialize multimodal client", "detail": error_msg}, 500
    
    # 根据任务编号设置不同的提示词
    task_prompts = {
        1: """You are an official IELTS Speaking Examiner conducting Part 1 of the IELTS Speaking test. 
Part 1 typically lasts 4-5 minutes and involves general questions about familiar topics (e.g., home, family, work, studies, interests).

Your task:
After the candidate's response, provide your evaluation based on the four official IELTS criteria:
- Fluency and Coherence
- Lexical Resource
- Grammatical Range and Accuracy
- Pronunciation
If the audio is not in English, give 0 score.
Your standards ALWAYS MUST be very STRICT.
Return your evaluation in the following format:

**Overall Score**: [number] out of 9.0

**Breakdown**:

**Fluency & Coherence**: [number]

**Lexical Resource**: [number]

**Grammatical Range & Accuracy**: [number]

**Pronunciation**: [number]

**Strengths**: 
...(list of strengths)

**Weaknesses**: 
...(list of weaknesses)

**Suggestions**:
...(list of suggestions)""",
        
        2: """You are an official IELTS Speaking Examiner conducting Part 2 of the IELTS Speaking test. 
Part 2 is the "Long Turn" where the candidate speaks for 1-2 minutes on a given topic after 1 minute of preparation.

Your task:
1. Provide a topic card with a task description
2. Listen to the candidate's 1-2 minute speech
3. Evaluate their performance based on the four official IELTS criteria
4. Provide detailed feedback

Return your evaluation in the following format:

**Overall Score**: [number] out of 9.0

**Breakdown**:

**Fluency & Coherence**: [number]

**Lexical Resource**: [number]

**Grammatical Range & Accuracy**: [number]

**Pronunciation**: [number]

**Strengths**: 
...(list of strengths)

**Weaknesses**: 
...(list of weaknesses)

**Suggestions**:
...(list of suggestions)""",
        
        3: """You are an official IELTS Speaking Examiner conducting Part 3 of the IELTS Speaking test. 
Part 3 is a two-way discussion (4-5 minutes) that explores abstract ideas and issues related to the topic in Part 2.

Your task:
1. Ask more abstract and analytical questions
2. Listen to the candidate's responses
3. Engage in a discussion, asking follow-up questions
4. Evaluate their ability to express and justify opinions, analyze, discuss and speculate about issues

Return your evaluation in the following format:

**Overall Score**: [number] out of 9.0

**Breakdown**:

**Fluency & Coherence**: [number]

**Lexical Resource**: [number]

**Grammatical Range & Accuracy**: [number]

**Pronunciation**: [number]

**Strengths**: 
...(list of strengths)

**Weaknesses**: 
...(list of weaknesses)

**Suggestions**:
...(list of suggestions)"""
    }
    
    system_prompt = task_prompts.get(task_number)
    if not system_prompt:
        return {"error": f"Invalid task number: {task_number}. Must be 1, 2, or 3"}, 400
    
    # 准备音频数据（使用与test.py相同的格式）
    # 如果已经包含data:前缀，直接使用；否则提取base64部分并添加前缀
    if audio_base64.startswith("data:"):
        # 如果已经包含完整的数据URL，直接使用
        audio_data_url = audio_base64
    else:
        # 如果是纯base64字符串，添加前缀
        audio_data_url = f"data:;base64,{audio_base64}"
    
    # 构建消息内容
    content_items = [
        {
            "type": "input_audio",
            "input_audio": {
                "data": audio_data_url,
                "format": audio_format,  # 支持 wav, mp3, webm 等格式
            },
        },
        {
            "type": "text",
            "text": system_prompt
        }
    ]
    
    # 如果有对话历史，添加上下文
    if conversation:
        context_text = "\n\nPrevious conversation:\n"
        for msg in conversation:  # 只保留最近3条消息作为上下文
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                context_text += f"Candidate: {content}\n"
            elif role == "bot":
                context_text += f"Examiner: {content}\n"
        content_items.append({
            "type": "text",
            "text": context_text
        })
    
    try:
        # 调用多模态API
        completion = client.chat.completions.create(
            model=os.getenv("DASHSCOPE_MODEL", "qwen3-omni-flash"),
            messages=[
                {
                    "role": "user",
                    "content": content_items,
                },
            ],
            modalities=["text", "audio"] if return_audio else ["text"],
            audio={"voice": "Cherry", "format": "wav"} if return_audio else None,
            stream=True,
            stream_options={"include_usage": True},
        )
        
        # 处理流式响应
        text_response = ""
        audio_string = ""
        transcript = ""
        
        for chunk in completion:
            if chunk.choices:
                if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                    text_response += chunk.choices[0].delta.content
                
                if hasattr(chunk.choices[0].delta, "audio") and chunk.choices[0].delta.audio:
                    try:
                        if "data" in chunk.choices[0].delta.audio:
                            audio_string += chunk.choices[0].delta.audio["data"]
                        if "transcript" in chunk.choices[0].delta.audio:
                            transcript += chunk.choices[0].delta.audio.get("transcript", "")
                    except Exception as e:
                        print(f"Error processing audio chunk: {e}")
        
        if not text_response:
            return {"error": "Empty response from multimodal API"}, 502
        
        # 解析评分
        scores = extract_speaking_scores(text_response)
        
        # 保存评分到dashboard
        if scores and len(scores) >= 4:
            save_speaking_scores(user_id, scores)
        
        # 准备响应数据
        response_data = {
            "text": text_response,
            "scores": scores,
            "transcript": transcript if transcript else None
        }
        
        # 如果有音频响应，使用base64编码的字符串
        if return_audio and audio_string:
            response_data["audio"] = audio_string
        
        # 保存对话历史
        user_message = {
            "message_id": str(uuid.uuid4()),
            "role": "user",
            "content": "🎤 Your recording",
            "conversation_id": int(conversation_id) if str(conversation_id).isdigit() else conversation_id,
            "part": task_number
        }
        bot_message = {
            "message_id": str(uuid.uuid4()),
            "role": "bot",
            "content": text_response,
            "conversation_id": int(conversation_id) if str(conversation_id).isdigit() else conversation_id,
            "part": task_number
        }

        # 如果有音频数据，使用base64编码保存为data URL格式
        if return_audio and audio_string:
            bot_message["audio"] = f"data:audio/wav;base64,{audio_string}"
        
        # 将用户上传的原始音频也保存到历史（使用 data URL，保留格式）
        try:
            if audio_base64:
                # 若前端已经传来带 data: 前缀的数据，此处直接复用，否则补上前缀
                if audio_base64.startswith("data:"):
                    user_message["audio"] = audio_base64
                else:
                    # 默认使用传入的 audio_format
                    fmt = audio_format or "wav"
                    user_message["audio"] = f"data:audio/{fmt};base64,{audio_base64}"
        except Exception as _:
            # 静默失败，不影响主流程
            pass
        
        save_oral_conversation_to_history(
            conversation_id=conversation_id,
            user_id=user_id,
            record=[user_message, bot_message]
        )
        
        return response_data, 200
        
    except Exception as e:
        return {
            "error": "Failed to process speaking task",
            "detail": str(e)
        }, 502


def extract_speaking_scores(text: str) -> Optional[List[float]]:
    """从评估文本中提取口语评分。
    
    Args:
        text: 包含评分的文本
    
    Returns:
        List[float]: [Fluency & Coherence, Lexical Resource, Grammatical Range & Accuracy, Pronunciation]
    """
    try:
        # 提取Breakdown部分的文本
        match = re.search(
            r"\*\*Breakdown\*\*:?([\s\S]*?)(?:\*\*Strengths\*\*|\*\*Weaknesses\*\*|\*\*Suggestions\*\*|$)",
            text,
            re.IGNORECASE,
        )
        
        breakdown_text = match.group(1).strip() if match else ""
        
        # 移除Markdown格式
        cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", breakdown_text)
        
        # 提取四个维度的分数
        pattern = (
            r"(?:Fluency\s*&\s*Coherence|Lexical Resource|Grammatical Range\s*&\s*Accuracy|Pronunciation)"
            r"\s*:\s*([0-9]+(?:\.[0-9]+)?)"
        )
        
        scores = [float(x) for x in re.findall(pattern, cleaned, flags=re.IGNORECASE)]
        
        # 确保有4个分数，按顺序：Fluency & Coherence, Lexical Resource, Grammatical Range & Accuracy, Pronunciation
        if len(scores) >= 4:
            return scores[:4]
        elif len(scores) > 0:
            # 如果分数不足，用0填充
            while len(scores) < 4:
                scores.append(0.0)
            return scores
        
        return None
    except Exception as e:
        print(f"Error extracting speaking scores: {e}")
        return None


def save_speaking_scores(user_id: str, scores: List[float]) -> None:
    """保存口语评分到dashboard CSV文件。
    
    Args:
        user_id: 用户ID
        scores: 评分列表 [Fluency & Coherence, Lexical Resource, Grammatical Range & Accuracy, Pronunciation]
    """
    try:
        app_dir = os.path.dirname(os.path.dirname(__file__))
        user_data_dir = os.path.join(app_dir, "data", user_id)
        os.makedirs(user_data_dir, exist_ok=True)
        
        dashboard_path = os.path.join(user_data_dir, "oral_dashboard.csv")
        
        # 检查文件是否存在，如果不存在则创建并写入表头
        if not os.path.exists(dashboard_path):
            with open(dashboard_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Fluency & Coherence",
                    "Lexical Resource",
                    "Grammatical Range & Accuracy",
                    "Pronunciation"
                ])
        
        # 追加评分数据
        with open(dashboard_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(scores)
            
    except Exception as e:
        print(f"[ERROR] Failed to save speaking scores for user {user_id}: {e}")


def generate_ielts_speaking_topic(conversation_id: str, user_id: str, part: int) -> Tuple[Dict[str, Any], int]:
    """生成 IELTS 口语题目（Part 1, 2, 或 3）。
    
    Args:
        conversation_id: 会话ID
        user_id: 用户ID
        part: 口语部分（1, 2, 或 3）
    
    Returns:
        Tuple[Dict[str, Any], int]: (响应数据, HTTP状态码)
    """
    llm, init_error = get_chat_model()
    if llm is None:
        error_msg = init_error or "Chat model not initialized"
        return {"error": "Failed to initialize LangChain ChatOpenAI", "detail": error_msg}, 500
    
    # 根据不同的 part 设置不同的提示词
    part_prompts = {
        1: {
            "system": "You are an IELTS Speaking Part 1 examiner. Your task is to generate ONE authentic, natural-sounding question — but **avoid overused clichés** (e.g., *Do you enjoy spending time outdoors?*, *Do you like reading?*).  ",
            "user": (
                """
                Draw from a broad range of Part 1 topics: work/study, hometown, accommodation, daily routine, technology, food, weather, hobbies (beyond sports/music), childhood, shopping, transport, pets, etc.  
→ Before generating, mentally 'roll a die' to pick a less common subtopic.  
→ Keep the question simple, direct, and answerable in 1–2 sentences.  
→ Output **only the question**, in English, no punctuation extras, no numbering, no quotation marks.
                """
            )
        },
        2: {
            "system": "You are an IELTS Speaking test question generator. Generate realistic Part 2 topic cards.",
            "user": (
                "Generate ONE realistic IELTS Speaking Part 2 topic card. "
                "Part 2 is a 'Long Turn' where candidates speak for 1-2 minutes. "
                "The topic card should include: "
                "1. A main topic (e.g., 'Describe a memorable journey', 'Describe a person you admire') "
                "2. A task description with 2-3 bullet points guiding what to cover. "
                "Format it as a clear topic card with the main topic as a heading and bullet points below. "
                "Return only the topic card content in English, without extra commentary."
            )
        },
        3: {
            "system": "You are an IELTS Speaking test question generator. Generate realistic Part 3 questions.",
            "user": (
                "Generate ONE realistic IELTS Speaking Part 3 question. "
                "Part 3 questions are abstract and analytical, exploring deeper issues related to topics from Part 2. "
                "These questions require candidates to express opinions, analyze, discuss, and speculate. "
                "Examples: 'What are the benefits and drawbacks of...?', 'How do you think... will change in the future?', "
                "'Do you think... is more important than...? Why?' "
                "Return only the question text in English, without extra commentary."
            )
        }
    }
    
    if part not in part_prompts:
        return {"error": f"Invalid part: {part}. Must be 1, 2, or 3"}, 400
    
    prompt = part_prompts[part]
    
    try:
        messages = [
            SystemMessage(content=prompt["system"]),
            HumanMessage(content=prompt["user"])
        ]
        response = llm.invoke(messages)
        content = (getattr(response, "content", "") or "").strip()
        
        if not content:
            return {"error": "Empty response from LLM"}, 502

        # Generate concise answering tips for the given part and question
        tips_system = "You are an IELTS Speaking coach. Provide concise, actionable tips."
        tips_user = ""
        if part == 1:
            tips_user = (
                "Given this Part 1 question, provide a brief guidance on how to answer naturally:\n"
                "- How to structure a 2-3 sentence answer (past/present/examples)\n"
                "- 2-3 helpful phrases or collocations\n"
                "- One common pitfall to avoid\n"
                "Keep it short and practical. Use bullet points. Question: "
                f"{content}"
            )
        elif part == 2:
            tips_user = (
                "Given this Part 2 topic card, provide a brief guidance on how to answer for 1-2 minutes:\n"
                "- A simple outline (opening, 2-3 points, closing)\n"
                "- 3-4 prompts to cover details (who/what/when/where/why/how)\n"
                "- 2-3 helpful linking phrases\n"
                "Keep it short and practical. Use bullet points. Topic:\n"
                f"{content}"
            )
        else:
            tips_user = (
                "Given this Part 3 question, provide a brief guidance on answering analytically:\n"
                "- A structure (position, reasons, examples, mini-conclusion)\n"
                "- 2-3 ideas or angles to consider\n"
                "- 2-3 academic phrases/connectors\n"
                "Keep it short and practical. Use bullet points. Question: "
                f"{content}"
            )

        tips_resp = llm.invoke([SystemMessage(content=tips_system), HumanMessage(content=tips_user)])
        tips = (getattr(tips_resp, "content", "") or "").strip()

        # 保存题目到历史记录
        message_id = str(uuid.uuid4())
        record = [{
            "message_id": message_id,
            "role": "bot",
            "content": content,
            "tips": tips,
            "conversation_id": int(conversation_id) if conversation_id.isdigit() else conversation_id,
            "part": part
        }]
        
        save_oral_conversation_to_history(
            conversation_id=conversation_id,
            user_id=user_id,
            record=record
        )
        
        return {"question": content, "tips": tips}, 200
        
    except Exception as e:
        return {
            "error": "Failed to generate speaking topic",
            "detail": str(e)
        }, 502


def save_oral_conversation_to_history(conversation_id: str, user_id: str, record: list) -> None:
    """将口语对话内容保存/追加到用户历史文件。
    
    - 路径：app/data/{user_id}/history.json
    - 若会话存在：向 conversation 追加消息
    - 若会话不存在：创建新会话并写入首条消息
    - 若文件不存在：创建新文件
    - 发生异常时静默失败（打印错误），不抛出
    """
    try:
        app_dir = os.path.dirname(os.path.dirname(__file__))
        user_data_dir = os.path.join(app_dir, "data", user_id)
        os.makedirs(user_data_dir, exist_ok=True)
        history_path = os.path.join(user_data_dir, "oral_history.json")
        
        # ===== 读取历史文件 =====
        history: list = []
        if os.path.exists(history_path):
            with open(history_path, "r", encoding="utf-8") as rf:
                try:
                    history = json.load(rf) or []
                except json.JSONDecodeError:
                    print(f"[WARN] history.json for user {user_id} is corrupted, resetting file.")
                    history = []
        
        # 转换 conversation_id 为整数（如果可能）
        try:
            conv_id_int = int(conversation_id)
        except (ValueError, TypeError):
            conv_id_int = conversation_id
        
        # ===== 查找或创建会话 =====
        conv_exists = False
        for conv in history:
            # 支持整数和字符串类型的 conversation_id 比较
            conv_id = conv.get("conversation_id")
            if conv_id == conv_id_int or str(conv_id) == str(conversation_id):
                conv["conversation"].extend(record)
                # 更新 selected_part（使用最新消息的 part）
                if record and isinstance(record[0], dict) and "part" in record[0]:
                    conv["selected_part"] = record[0].get("part")
                conv_exists = True
                break
        
        if not conv_exists:
            # 创建新会话
            new_conv = {
                "conversation_id": conv_id_int,
                "title": f"Practice {conv_id_int}",
                "conversation": record,
                "selected_part": record[0].get("part") if record and isinstance(record[0], dict) and "part" in record[0] else None
            }
            history.append(new_conv)
        
        # ===== 写入文件 =====
        with open(history_path, "w", encoding="utf-8") as wf:
            json.dump(history, wf, ensure_ascii=False, indent=4)
            
    except Exception as e:
        print(f"[ERROR] Failed to save oral conversation history for user {user_id}: {e}")