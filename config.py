class Config:
    SECRET_KEY = "supersecret"
    SQLALCHEMY_DATABASE_URI = "sqlite:///app.db"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # LLM 配置（通过环境变量覆盖）
    LLM_API_BASE = "https://api.deepseek.com/v1"  # 例如：https://api.openai.com
    LLM_API_KEY = "sk-3094e607e1a749a888d2bd49298754ec"
    LLM_MODEL = "deepseek-chat"     # 例如：gpt-4o-mini / gpt-4.1 / deepseek-chat 等


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False