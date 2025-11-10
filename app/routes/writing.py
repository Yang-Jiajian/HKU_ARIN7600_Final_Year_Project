import os
import json
import pandas as pd
from flask import Blueprint, jsonify, request
from app.utils.llm import generate_ielts_topic, evaluate_ielts_essay, continue_ielts_conversation


writing_bp = Blueprint("writing", __name__)

# http://localhost:5000/writing/get_topic
@writing_bp.route("/writing/get_topic", methods=["POST"])
def get_ielts_prompt():
    data = request.get_json(silent=True) or {}
    conversation_id = data.get("conversation_id", "")
    user_id = data.get("user_id", "")
    print(f"conversation id:{conversation_id}")
    result = generate_ielts_topic(conversation_id=conversation_id, user_id=user_id)
    # result may be (dict, status) or dict
    if isinstance(result, tuple):
        body, status = result
        return jsonify(body), status
    return jsonify(result)


@writing_bp.route("/writing/evaluate", methods=["POST"])
def evaluate_essay():
    data = request.get_json(silent=True) or {}
    conversation = data.get("conversation",[])
    print("="*10)
    print(conversation)
    print("="*10)
    conversation_id = data.get("conversation_id", "")
    user_id = data.get("user_id", "")
    essay = data.get("essay","")
    
    result = evaluate_ielts_essay(conversation=conversation, essay=essay, conversation_id=conversation_id, user_id=user_id)
    if isinstance(result, tuple):
        body, status = result
        return jsonify(body), status
    return jsonify(result)


@writing_bp.route("/writing/continue", methods=["POST"])
def continue_conversation():
    data = request.get_json(silent=True) or {}
    query = data.get("query","")
    conversation = data.get("conversation", [])
    conversation_id = data.get("conversation_id", "")
    user_id = data.get("user_id", "")
    print("="*10)
    print(conversation)
    print("="*10)
    if not isinstance(conversation, list):
        return jsonify({"error": "conversation must be a list of messages"}), 400
    result = continue_ielts_conversation(conversation=conversation, query=query,conversation_id=conversation_id, user_id=user_id)
    if isinstance(result, tuple):
        body, status = result
        return jsonify(body), status
    return jsonify(result)

@writing_bp.route("/writing/get_history", methods=["GET"])
def get_history():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    result = get_history_by_userId(user_id=user_id)
    if isinstance(result, tuple):
        body, status = result
        return jsonify(body), status
    return jsonify(result)


def get_history_by_userId(user_id: str):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    history_path = os.path.join(base_dir, "data", user_id, "writing_history.json")
    print(history_path)
    if not os.path.exists(history_path):
        return {"error": f"writing_history file not found for user {user_id}"}, 404

    try:
        with open(history_path, "r", encoding="utf-8") as file:
            history_data = json.load(file)
    except json.JSONDecodeError:
        return {"error": "writing_history file is not valid JSON"}, 500
    except Exception as exc:
        return {"error": "failed to read writing_history file", "detail": str(exc)}, 500

    return {"history": history_data}

@writing_bp.route("/writing/get_dashboard", methods=["GET"])
def get_dashboard():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    
    result = get_dashboard_by_userId(user_id=user_id)
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


def get_dashboard_by_userId(user_id: str):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    dashboard_path = os.path.join(base_dir, "data", user_id, "writing_dashboard.csv")
    
    if not os.path.exists(dashboard_path):
        return {"error": f"writing_dashboard.csv not found for user {user_id}"}, 404
    
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
        return {"error": "writing_dashboard.csv is empty"}, 400
    except Exception as exc:
        return {"error": "failed to read or process writing_dashboard.csv", "detail": str(exc)}, 500