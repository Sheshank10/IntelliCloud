"""
Database Models for IntelliCloud Attendance System
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class Student(db.Model):
    __tablename__ = 'students'

    id = db.Column(db.Integer, primary_key=True)
    roll_number = db.Column(db.String(20), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)
    department = db.Column(db.String(100), nullable=True)
    year = db.Column(db.Integer, nullable=True)
    image_count = db.Column(db.Integer, default=0)
    registered_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)

    attendances = db.relationship('Attendance', backref='student', lazy=True)

    def to_dict(self):
        return {
            'id': self.id,
            'roll_number': self.roll_number,
            'name': self.name,
            'email': self.email,
            'department': self.department,
            'year': self.year,
            'image_count': self.image_count,
            'registered_at': self.registered_at.strftime('%Y-%m-%d %H:%M:%S'),
            'is_active': self.is_active
        }

    def __repr__(self):
        return f'<Student {self.roll_number} - {self.name}>'


class Attendance(db.Model):
    __tablename__ = 'attendance'

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=False)
    date = db.Column(db.Date, default=datetime.utcnow().date)
    time_in = db.Column(db.Time, nullable=True)
    status = db.Column(db.String(20), default='Present')  # Present / Absent / Late
    confidence = db.Column(db.Float, nullable=True)
    marked_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'student_id': self.student_id,
            'student_name': self.student.name if self.student else 'Unknown',
            'roll_number': self.student.roll_number if self.student else 'N/A',
            'date': self.date.strftime('%Y-%m-%d'),
            'time_in': self.time_in.strftime('%H:%M:%S') if self.time_in else None,
            'status': self.status,
            'confidence': self.confidence,
            'marked_at': self.marked_at.strftime('%Y-%m-%d %H:%M:%S')
        }

    def __repr__(self):
        return f'<Attendance {self.student_id} on {self.date}>'
