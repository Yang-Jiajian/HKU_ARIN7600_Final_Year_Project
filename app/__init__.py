from flask import Flask
from flask_cors import CORS  # 跨域支持
from .routes import register_blueprints
from .utils.llm import initialize_chat_model, initialize_multimodal_client

def create_app():
    app = Flask(__name__)
    # 加载配置（可根据环境变量切换不同配置类，如 DevelopmentConfig / ProductionConfig）
    app.config.from_object("config.DevelopmentConfig")

    CORS(app)  # ⚙️ 开启跨域支持，让前端(如http://localhost:5173)能访问API

    # 初始化全局 chat model
    initialize_chat_model(app)
    
    # 初始化多模态客户端
    initialize_multimodal_client(app)

    # 注册蓝图
    register_blueprints(app)

    return app