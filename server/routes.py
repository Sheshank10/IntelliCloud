"""
Flask Route Definitions - REST API for IntelliCloud
"""

from flask import Blueprint, request, jsonify, send_file, render_template, Response
from utils.attendance_service import AttendanceService
from utils.student_service import StudentService
from models.face_recognition import FaceRecognitionEngine
import cv2
import io
import base64
import numpy as np

engine = FaceRecognitionEngine()
main_bp = Blueprint('main', __name__)


# ─── Pages ──────────────────────────────────────────────────────────────────

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/register')
def register_page():
    return render_template('register.html')

@main_bp.route('/attendance')
def attendance_page():
    return render_template('attendance.html')

@main_bp.route('/dashboard')
def dashboard_page():
    return render_template('dashboard.html')


# ─── Student API ─────────────────────────────────────────────────────────────

@main_bp.route('/api/students', methods=['GET'])
def get_students():
    students = StudentService.get_all_students()
    return jsonify({'success': True, 'students': students})


@main_bp.route('/api/students/<roll_number>', methods=['GET'])
def get_student(roll_number):
    student = StudentService.get_student(roll_number)
    if student:
        return jsonify({'success': True, 'student': student})
    return jsonify({'success': False, 'message': 'Student not found'}), 404


@main_bp.route('/api/students/register', methods=['POST'])
def register_student():
    data = request.get_json()
    result = StudentService.register_student(
        roll_number=data.get('roll_number'),
        name=data.get('name'),
        email=data.get('email'),
        department=data.get('department'),
        year=data.get('year')
    )
    return jsonify(result), 200 if result['success'] else 400


@main_bp.route('/api/students/<roll_number>', methods=['DELETE'])
def delete_student(roll_number):
    result = StudentService.delete_student(roll_number)
    return jsonify(result)


# ─── Face Capture & Training ──────────────────────────────────────────────────

@main_bp.route('/api/capture', methods=['POST'])
def capture_faces():
    """
    Accepts: { roll_number, images: [base64, ...] }
    Saves face crops and returns count saved.
    """
    data = request.get_json()
    roll_number = data.get('roll_number')
    images_b64 = data.get('images', [])

    frames = []
    for b64 in images_b64:
        img_bytes = base64.b64decode(b64.split(',')[-1])
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is not None:
            frames.append(frame)

    saved = engine.save_student_images(roll_number, frames, count=30)
    StudentService.update_image_count(roll_number, saved)
    return jsonify({'success': True, 'saved': saved, 'message': f'{saved} face images saved.'})


@main_bp.route('/api/train', methods=['POST'])
def train_model():
    result = StudentService.train_model()
    return jsonify(result)


# ─── Recognition & Attendance ─────────────────────────────────────────────────

@main_bp.route('/api/recognize', methods=['POST'])
def recognize_face():
    """
    Accepts: { image: base64 }
    Returns identified student + marks attendance.
    """
    data = request.get_json()
    b64 = data.get('image', '')

    img_bytes = base64.b64decode(b64.split(',')[-1])
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({'success': False, 'message': 'Invalid image'}), 400

    # Load model if needed
    if not hasattr(engine, 'label_map'):
        loaded = engine.load_lbph_model()
        if not loaded:
            return jsonify({'success': False, 'message': 'Model not trained yet. Please train first.'}), 400

    faces = engine.detect_faces(frame)
    if len(faces) == 0:
        return jsonify({'success': False, 'message': 'No face detected in image.'})

    x, y, w, h = faces[0]
    face_roi = engine.crop_face(frame, faces[0])
    face_resized = cv2.resize(face_roi, (200, 200))

    roll_number, confidence = engine.predict_lbph(face_resized)

    if roll_number is None:
        return jsonify({'success': False, 'message': 'Face not recognized. Confidence too low.'})

    # Mark attendance
    result = AttendanceService.mark_attendance(roll_number, confidence)
    result['face_rect'] = {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
    result['confidence'] = float(confidence)
    return jsonify(result)


# ─── Attendance API ───────────────────────────────────────────────────────────

@main_bp.route('/api/attendance/today', methods=['GET'])
def today_attendance():
    records = AttendanceService.get_today_attendance()
    return jsonify({'success': True, 'records': records})


@main_bp.route('/api/attendance/summary', methods=['GET'])
def attendance_summary():
    summary = AttendanceService.get_attendance_summary()
    return jsonify({'success': True, 'summary': summary})


@main_bp.route('/api/attendance/<roll_number>', methods=['GET'])
def student_attendance(roll_number):
    days = request.args.get('days', 30, type=int)
    records = AttendanceService.get_student_attendance(roll_number, days)
    return jsonify({'success': True, 'records': records})


@main_bp.route('/api/attendance/export', methods=['GET'])
def export_attendance():
    target_date = request.args.get('date')
    csv_bytes = AttendanceService.export_attendance_csv(target_date)
    return send_file(
        io.BytesIO(csv_bytes),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'attendance_{target_date or "today"}.csv'
    )


def register_routes(app):
    app.register_blueprint(main_bp)
