import os
import json
import base64
import pandas as pd
from flask import Blueprint, jsonify, request
from app.utils.llm import process_ielts_speaking_task, generate_ielts_speaking_topic

oral_bp = Blueprint("oral", __name__)


@oral_bp.route("/oral/get_topic", methods=["POST"])
def get_topic():
    """获取口语题目（Part 1, 2, 或 3）"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body is required"}), 400
        
        user_id = data.get("user_id")
        conversation_id = data.get("conversation_id")
        part = data.get("part")
        
        if not user_id:
            return jsonify({"error": "user_id is required"}), 400
        if not conversation_id:
            return jsonify({"error": "conversation_id is required"}), 400
        if part is None:
            return jsonify({"error": "part is required"}), 400
        
        # 验证 part 是否为有效值（1, 2, 或 3）
        try:
            part = int(part)
            if part not in [1, 2, 3]:
                return jsonify({"error": "part must be 1, 2, or 3"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "part must be an integer (1, 2, or 3)"}), 400
        
        # 调用 LLM 生成题目
        result = generate_ielts_speaking_topic(
            conversation_id=str(conversation_id),
            user_id=user_id,
            part=part
        )
        
        if isinstance(result, tuple):
            body, status = result
            return jsonify(body), status
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": "Failed to get topic", "detail": str(e)}), 500


@oral_bp.route("/oral/evaluate_audio", methods=["POST"])
def evaluate_audio():
    """评估口语音频"""
    try:
        # 检查是否有音频文件
        if 'audio' not in request.files:
            return jsonify({"error": "audio file is required"}), 400
        
        audio_file = request.files['audio']
        
        # 读取音频数据并检查是否为空
        audio_file.seek(0)  # 确保从文件开头读取
        audio_data = audio_file.read()
        if not audio_data or len(audio_data) == 0:
            return jsonify({"error": "audio file is empty"}), 400
        
        # 重置文件指针，以便后续使用
        audio_file.seek(0)
        
        # 获取其他参数
        user_id = request.form.get("user_id")
        part = request.form.get("part")
        conversation_id = request.form.get("conversation_id")
        
        if not user_id:
            return jsonify({"error": "user_id is required"}), 400
        if not part:
            return jsonify({"error": "part is required"}), 400
        if not conversation_id:
            return jsonify({"error": "conversation_id is required"}), 400
        
        # 验证 part 是否为有效值（1, 2, 或 3）
        try:
            part = int(part)
            if part not in [1, 2, 3]:
                return jsonify({"error": "part must be 1, 2, or 3"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "part must be an integer (1, 2, or 3)"}), 400
        
        # 将音频文件转换为 base64（audio_data 已在上面读取）
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        # 获取音频格式（从文件名或 Content-Type 推断）
        audio_format = "webm"  # 默认格式，前端发送的是 webm
        filename = audio_file.filename.lower() if audio_file.filename else ""
        content_type = audio_file.content_type or ""
        
        # 优先从文件名判断格式
        if filename:
            if filename.endswith('.wav'):
                audio_format = "wav"
            elif filename.endswith('.mp3'):
                audio_format = "mp3"
            elif filename.endswith('.webm'):
                audio_format = "webm"
            elif filename.endswith('.ogg'):
                audio_format = "ogg"
        # 如果文件名不可用，从 Content-Type 判断
        elif content_type:
            if 'webm' in content_type:
                audio_format = "webm"
            elif 'wav' in content_type or 'wave' in content_type:
                audio_format = "wav"
            elif 'mp3' in content_type or 'mpeg' in content_type:
                audio_format = "mp3"
            elif 'ogg' in content_type:
                audio_format = "ogg"
        
        # 获取对话历史
        conversation = None
        try:
            history_result = get_history_by_userId(user_id=user_id)
            if not isinstance(history_result, tuple) and "history" in history_result:
                history_data = history_result["history"]
                # 转换 conversation_id 为整数（如果可能）以便比较
                try:
                    conv_id_int = int(conversation_id)
                except (ValueError, TypeError):
                    conv_id_int = conversation_id
                
                # 查找对应的会话
                for conv in history_data:
                    conv_id = conv.get("conversation_id")
                    # 支持整数和字符串类型的比较
                    if conv_id == conv_id_int or str(conv_id) == str(conversation_id):
                        conversation = conv.get("conversation", [])
                        break
        except Exception as e:
            print(f"[WARN] Failed to load conversation history: {e}")
            # 如果获取历史失败，继续处理，不使用历史记录
        
        # 调用处理函数
        result = process_ielts_speaking_task(
            audio_base64=audio_base64,
            task_number=part,
            conversation_id=str(conversation_id),
            user_id=user_id,
            conversation=conversation,
            return_audio=True,
            audio_format=audio_format
        )
        
        if isinstance(result, tuple):
            body, status = result
            # 前端期望的响应格式包含 assistant 字段
            if status == 200 and "text" in body:
                body["assistant"] = body["text"]
            return jsonify(body), status
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": "Failed to evaluate audio", "detail": str(e)}), 500

@oral_bp.route("/oral/get_dashboard", methods=["GET"])
def get_oral_dashboard():
    """获取口语dashboard数据"""
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    
    result = get_oral_dashboard_by_userId(user_id=user_id)
    if isinstance(result, tuple):
        body, status = result
        return jsonify(body), status
    return jsonify(result)


def round_to_half(value):
    """
    按照规则舍入：
    - [0, 0.25) -> 向下取整
    - [0.25, 0.75) -> 0.5
    - [0.75, 1) -> 向上取整
    """
    if pd.isna(value):
        return 0.0
    
    integer_part = int(value)
    decimal_part = value - integer_part
    
    if decimal_part < 0.25:
        return float(integer_part)
    elif decimal_part < 0.75:
        return float(integer_part) + 0.5
    else:
        return float(integer_part + 1)


def get_oral_dashboard_by_userId(user_id: str):
    """获取用户的口语dashboard数据"""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    dashboard_path = os.path.join(base_dir, "data", user_id, "oral_dashboard.csv")
    
    if not os.path.exists(dashboard_path):
        return {"error": f"oral_dashboard.csv not found for user {user_id}"}, 404
    
    try:
        # 读取 CSV 文件
        df = pd.read_csv(dashboard_path, encoding='utf-8')
        
        # 检查文件是否为空
        if df.empty:
            return {"radar_chart": {}, "line_chart": []}
        
        # 计算各字段的平均值（用于 radar_chart）
        averages = {}
        for column in df.columns:
            # 确保列是数值类型
            numeric_values = pd.to_numeric(df[column], errors='coerce')
            avg = numeric_values.mean()
            # 如果平均值是 NaN，返回 None 或 0，并保留1位小数
            averages[column] = round(float(avg), 1) if not pd.isna(avg) else 0.0
        
        # 计算每一行的平均值（用于 line_chart）
        line_chart = []
        for index, row in df.iterrows():
            # 将每一行的所有值转换为数值类型
            numeric_row = pd.to_numeric(row, errors='coerce')
            # 计算该行的平均值
            row_avg = numeric_row.mean()
            # 按照规则舍入
            rounded_avg = round_to_half(row_avg)
            line_chart.append(rounded_avg)
        
        return {"radar_chart": averages, "line_chart": line_chart}
        
    except pd.errors.EmptyDataError:
        return {"error": "oral_dashboard.csv is empty"}, 400
    except Exception as exc:
        return {"error": "failed to read or process oral_dashboard.csv", "detail": str(exc)}, 500


@oral_bp.route("/oral/get_history", methods=["GET"])
def get_oral_history():
    """获取口语历史记录"""
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    result = get_history_by_userId(user_id=user_id)
    if isinstance(result, tuple):
        body, status = result
        return jsonify(body), status
    return jsonify(result)


def get_history_by_userId(user_id: str):
    """获取用户的口语历史记录"""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    history_path = os.path.join(base_dir, "data", user_id, "oral_history.json")
    print(history_path)
    if not os.path.exists(history_path):
        return {"error": f"history file not found for user {user_id}"}, 404

    try:
        with open(history_path, "r", encoding="utf-8") as file:
            history_data = json.load(file)
    except json.JSONDecodeError:
        return {"error": "history file is not valid JSON"}, 500
    except Exception as exc:
        return {"error": "failed to read history file", "detail": str(exc)}, 500

    return {"history": history_data}

