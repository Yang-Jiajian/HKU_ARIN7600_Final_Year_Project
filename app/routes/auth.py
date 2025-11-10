import pandas as pd
import jwt
import os
import json
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
    username = data.get("username","")
    password = data.get("password","")
    
    # 检查用户名和密码是否为空
    if not username or not password:
        return jsonify({"state": 400, "message": "Username and password are required"})
    
    # 读取用户信息CSV文件
    csv_path = './app/data/user_info.csv'
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    # 检查用户名是否已存在
    matches = df[df['username'] == username]
    if len(matches) > 0:
        return jsonify({"state": 409, "message": "Username already exists"})
    
    # 添加新用户到CSV文件
    new_user = pd.DataFrame({'username': [username], 'password': [password]})
    df = pd.concat([df, new_user], ignore_index=True)
    df.to_csv(csv_path, index=False, encoding='utf-8')
    
    # 创建用户文件夹
    user_folder_path = f'./app/data/{username}'
    os.makedirs(user_folder_path, exist_ok=True)
    
    # 创建初始的 history.json 文件
    writing_history_file_path = os.path.join(user_folder_path, 'writing_history.json')
    oral_history_file_path = os.path.join(user_folder_path, 'oral_history.json')
    with open(writing_history_file_path, 'w', encoding='utf-8') as f:
        json.dump([], f, ensure_ascii=False, indent=4)
    with open(oral_history_file_path, 'w', encoding='utf-8') as f:
        json.dump([], f, ensure_ascii=False, indent=4)
    
    # 创建初始的 writing_dashboard.csv 文件（只包含表头）
    writing_dashboard_columns = ['Task Achievement', 'Coherence', 'Lexical', 'Grammar']
    writing_dashboard_df = pd.DataFrame(columns=writing_dashboard_columns)
    writing_dashboard_path = os.path.join(user_folder_path, 'writing_dashboard.csv')
    writing_dashboard_df.to_csv(writing_dashboard_path, index=False, encoding='utf-8')
    
    # 创建初始的 oral_dashboard.csv 文件（只包含表头）
    oral_dashboard_columns = ['Fluency','Gramma','Lexical','Pronunciation']
    oral_dashboard_df = pd.DataFrame(columns=oral_dashboard_columns)
    oral_dashboard_path = os.path.join(user_folder_path, 'oral_dashboard.csv')
    oral_dashboard_df.to_csv(oral_dashboard_path, index=False, encoding='utf-8')
    
    return jsonify({"state": 200, "message": "User registered successfully"})