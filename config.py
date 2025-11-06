import os
from dotenv import load_dotenv

# 从 .env 加载环境变量（若文件存在）
load_dotenv()


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "sqlite:///app.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # LLM 配置（通过 .env / 环境变量提供）
    LLM_API_BASE = os.getenv("LLM_API_BASE")  # 例如：https://api.openai.com
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL")  # 例如：gpt-4o-mini / gpt-4.1 / deepseek-chat 等


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False