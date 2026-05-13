"""
Attendance Service - Business logic for marking & querying attendance
"""

from datetime import datetime, date, timedelta
from database.db import db, Student, Attendance
import pandas as pd
import io


class AttendanceService:

    @staticmethod
    def mark_attendance(roll_number: str, confidence: float = None) -> dict:
        """
        Mark attendance for a student identified by roll_number.
        Prevents duplicate entries for the same day.
        """
        student = Student.query.filter_by(roll_number=roll_number, is_active=True).first()
        if not student:
            return {'success': False, 'message': f'Student {roll_number} not found.'}

        today = date.today()
        existing = Attendance.query.filter_by(
            student_id=student.id, date=today
        ).first()

        if existing:
            return {
                'success': False,
                'already_marked': True,
                'message': f'Attendance already marked for {student.name} today.',
                'student': student.to_dict()
            }

        now = datetime.now()
        record = Attendance(
            student_id=student.id,
            date=today,
            time_in=now.time(),
            status='Present',
            confidence=confidence,
            marked_at=now
        )
        db.session.add(record)
        db.session.commit()

        return {
            'success': True,
            'message': f'Attendance marked for {student.name}.',
            'student': student.to_dict(),
            'attendance': record.to_dict()
        }

    @staticmethod
    def get_today_attendance():
        """Return all attendance records for today."""
        today = date.today()
        records = Attendance.query.filter_by(date=today).all()
        return [r.to_dict() for r in records]

    @staticmethod
    def get_attendance_by_date(target_date: str):
        """Return attendance for a specific date (YYYY-MM-DD)."""
        d = datetime.strptime(target_date, '%Y-%m-%d').date()
        records = Attendance.query.filter_by(date=d).all()
        return [r.to_dict() for r in records]

    @staticmethod
    def get_student_attendance(roll_number: str, days: int = 30):
        """Return last N days attendance for a specific student."""
        student = Student.query.filter_by(roll_number=roll_number).first()
        if not student:
            return []
        since = date.today() - timedelta(days=days)
        records = Attendance.query.filter(
            Attendance.student_id == student.id,
            Attendance.date >= since
        ).order_by(Attendance.date.desc()).all()
        return [r.to_dict() for r in records]

    @staticmethod
    def get_attendance_summary():
        """Summary stats: total students, present today, absent today."""
        today = date.today()
        total_students = Student.query.filter_by(is_active=True).count()
        present_today = Attendance.query.filter_by(date=today, status='Present').count()
        absent_today = total_students - present_today
        return {
            'total_students': total_students,
            'present_today': present_today,
            'absent_today': absent_today,
            'date': today.strftime('%Y-%m-%d')
        }

    @staticmethod
    def export_attendance_csv(target_date: str = None) -> bytes:
        """Export attendance as CSV bytes for download."""
        if target_date:
            records = AttendanceService.get_attendance_by_date(target_date)
        else:
            records = AttendanceService.get_today_attendance()

        df = pd.DataFrame(records)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return buf.getvalue().encode('utf-8')
