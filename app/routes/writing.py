import os
import json
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
    history_path = os.path.join(base_dir, "data", user_id, "history.json")

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