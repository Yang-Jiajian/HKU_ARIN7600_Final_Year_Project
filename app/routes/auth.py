from flask import Blueprint, jsonify, request

auth_bp = Blueprint("auth", __name__)

@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    # 简化示例：实际应验证数据库用户
    if username == "admin" and password == "123456":
        return jsonify({"token": "fake-jwt-token", "user": username})
    return jsonify({"error": "Invalid credentials"}), 401