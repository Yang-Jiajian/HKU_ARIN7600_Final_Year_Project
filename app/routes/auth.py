import pandas as pd
import jwt
from datetime import datetime, timedelta
from flask import Blueprint, jsonify, request
from config import Config

auth_bp = Blueprint("auth", __name__)

@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    username = data.get("username","")
    password = data.get("password","")

    print(username, password)
    # 简化示例：实际应验证数据库用户
    df = pd.read_csv('./app/data/user_info.csv', encoding='utf-8')
    matches = df[df['username'] == username]
    if len(matches) == 0:
        return jsonify({"state": 123, "message":"User Not Exist"})
    stored_password = matches.iloc[0]['password']
    if str(stored_password) == password:
        # # 生成 JWT token
        # payload = {
        #     'username': username,
        #     'exp': datetime.utcnow() + timedelta(days=7),  # token 7天后过期
        #     'iat': datetime.utcnow()  # 签发时间
        # }
        # token = jwt.encode(payload, Config.SECRET_KEY, algorithm='HS256')
        return jsonify({"state": 200,"message":"ok"})
    else:
        return jsonify({"state": 456, "message":"Password Incorrect"})
    

@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
