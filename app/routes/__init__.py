from .main import main_bp
from .auth import auth_bp
from .writing import writing_bp

def register_blueprints(app):
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(writing_bp)