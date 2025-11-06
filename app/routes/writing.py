from flask import Blueprint, jsonify, request
from app.utils.llm import generate_ielts_prompt, evaluate_ielts_essay, continue_ielts_conversation


writing_bp = Blueprint("writing", __name__)


@writing_bp.route("/writing/get_topic", methods=["GET"])
def get_ielts_prompt():
    result = generate_ielts_prompt()
    # result may be (dict, status) or dict
    if isinstance(result, tuple):
        body, status = result
        return jsonify(body), status
    return jsonify(result)

@writing_bp.route("/writing/evaluate", methods=["POST"])
def evaluate_essay():
    data = request.get_json(silent=True) or {}
    topic = data.get("topic", "")
    essay = data.get("essay", "")
    if not topic or not essay:
        return jsonify({"error": "Missing topic or essay"}), 400
    result = evaluate_ielts_essay(topic=topic, essay=essay)
    if isinstance(result, tuple):
        body, status = result
        return jsonify(body), status
    return jsonify(result)

@writing_bp.route("/writing/continue", methods=["POST"])
def continue_conversation():
    data = request.get_json(silent=True) or {}
    conversation = data.get("conversation", [])
    if not isinstance(conversation, list):
        return jsonify({"error": "conversation must be a list of messages"}), 400
    result = continue_ielts_conversation(conversation=conversation)
    if isinstance(result, tuple):
        body, status = result
        return jsonify(body), status
    return jsonify(result)
