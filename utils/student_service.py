"""
Student Service - Registration and management
"""

from database.db import db, Student
from models.face_recognition import FaceRecognitionEngine
import os


engine = FaceRecognitionEngine()


class StudentService:

    @staticmethod
    def register_student(roll_number: str, name: str, email: str = None,
                         department: str = None, year: int = None) -> dict:
        """Register a new student in the database."""
        existing = Student.query.filter_by(roll_number=roll_number).first()
        if existing:
            return {'success': False, 'message': f'Roll number {roll_number} already registered.'}

        student = Student(
            roll_number=roll_number,
            name=name,
            email=email,
            department=department,
            year=year
        )
        db.session.add(student)
        db.session.commit()
        return {'success': True, 'message': 'Student registered successfully.', 'student': student.to_dict()}

    @staticmethod
    def update_image_count(roll_number: str, count: int) -> bool:
        student = Student.query.filter_by(roll_number=roll_number).first()
        if student:
            student.image_count += count
            db.session.commit()
            return True
        return False

    @staticmethod
    def get_all_students():
        students = Student.query.filter_by(is_active=True).all()
        return [s.to_dict() for s in students]

    @staticmethod
    def get_student(roll_number: str):
        student = Student.query.filter_by(roll_number=roll_number).first()
        return student.to_dict() if student else None

    @staticmethod
    def delete_student(roll_number: str) -> dict:
        student = Student.query.filter_by(roll_number=roll_number).first()
        if not student:
            return {'success': False, 'message': 'Student not found.'}
        student.is_active = False
        db.session.commit()
        return {'success': True, 'message': f'Student {roll_number} deactivated.'}

    @staticmethod
    def train_model() -> dict:
        """Retrain the face recognition model with all registered students."""
        success, message = engine.train_lbph()
        return {'success': success, 'message': message}
