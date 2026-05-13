"""
IntelliCloud - Face Detecting Device for Attendance
Main Flask Application Entry Point
"""

from flask import Flask
from flask_cors import CORS
from database.db import db
from server.routes import register_routes
from dotenv import load_dotenv
import os

load_dotenv()

def create_app():
    app = Flask(__name__, template_folder='templates', static_folder='static')

    # Config
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'intellicloud-secret-key')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///attendance.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['UPLOAD_FOLDER'] = 'TrainingImage'
    app.config['MODEL_FOLDER'] = 'TrainingImageLabel'

    CORS(app)
    db.init_app(app)

    with app.app_context():
        db.create_all()

    register_routes(app)

    return app


if __name__ == '__main__':
    app = create_app()
    print("=" * 50)
    print("  IntelliCloud Attendance System")
    print("  Running at http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
