"""
Face Recognition Engine
Uses FaceNet for feature extraction + SVM classifier
Falls back to LBPH (OpenCV) for lightweight mode
"""

import cv2
import numpy as np
import os
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class FaceRecognitionEngine:
    """
    Multi-algorithm Face Recognition Engine:
    - Primary  : FaceNet (deep features) + SVM classifier
    - Fallback : LBPH (Local Binary Pattern Histograms) via OpenCV
    """

    def __init__(self, model_dir='TrainingImageLabel', image_dir='TrainingImage'):
        self.model_dir = model_dir
        self.image_dir = image_dir
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.lbph_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.svm_model = None
        self.label_encoder = LabelEncoder()
        self.embeddings = []
        self.labels = []
        self.mode = 'lbph'   # 'lbph' or 'facenet'
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Face Detection
    # ------------------------------------------------------------------
    def detect_faces(self, frame):
        """Detect faces in a BGR frame. Returns list of (x,y,w,h) rects."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
        )
        return faces if len(faces) > 0 else []

    def crop_face(self, frame, rect):
        """Crop and return a gray face ROI given (x,y,w,h)."""
        x, y, w, h = rect
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray[y:y + h, x:x + w]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_lbph(self):
        """Train the LBPH recognizer from images in TrainingImage/."""
        faces, labels = [], []
        label_map = {}
        current_label = 0

        for folder_name in os.listdir(self.image_dir):
            folder_path = os.path.join(self.image_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            if folder_name not in label_map:
                label_map[folder_name] = current_label
                current_label += 1

            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)
                if not (img_path.endswith('.jpg') or img_path.endswith('.png')):
                    continue
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                detected = self.face_cascade.detectMultiScale(img, 1.3, 5)
                for (x, y, w, h) in detected:
                    faces.append(img[y:y + h, x:x + w])
                    labels.append(label_map[folder_name])

        if not faces:
            return False, "No faces found for training."

        self.lbph_recognizer.train(faces, np.array(labels))
        self.lbph_recognizer.save(os.path.join(self.model_dir, 'lbph_model.yml'))

        # Save label map
        with open(os.path.join(self.model_dir, 'label_map.pkl'), 'wb') as f:
            pickle.dump(label_map, f)

        self.mode = 'lbph'
        return True, f"LBPH model trained on {len(faces)} face samples from {len(label_map)} students."

    def train_svm(self, embeddings, labels):
        """Train SVM classifier on pre-computed face embeddings."""
        if len(set(labels)) < 2:
            return False, "Need at least 2 different students to train SVM."

        encoded_labels = self.label_encoder.fit_transform(labels)
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, encoded_labels, test_size=0.2, random_state=42
        )
        self.svm_model = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale')
        self.svm_model.fit(X_train, y_train)
        acc = accuracy_score(y_test, self.svm_model.predict(X_test))

        # Save
        with open(os.path.join(self.model_dir, 'svm_model.pkl'), 'wb') as f:
            pickle.dump(self.svm_model, f)
        with open(os.path.join(self.model_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)

        self.mode = 'svm'
        return True, f"SVM model trained. Accuracy: {acc * 100:.2f}%"

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def load_lbph_model(self):
        model_path = os.path.join(self.model_dir, 'lbph_model.yml')
        label_path = os.path.join(self.model_dir, 'label_map.pkl')
        if not os.path.exists(model_path):
            return False
        self.lbph_recognizer.read(model_path)
        with open(label_path, 'rb') as f:
            self.label_map = pickle.load(f)
        self.reverse_map = {v: k for k, v in self.label_map.items()}
        return True

    def predict_lbph(self, face_roi):
        """
        Predict identity using LBPH.
        Returns (roll_number, confidence) or (None, None).
        Lower confidence value = better match in LBPH.
        """
        label, confidence = self.lbph_recognizer.predict(face_roi)
        if confidence < 70:   # threshold – tune as needed
            roll = self.reverse_map.get(label, None)
            return roll, confidence
        return None, confidence

    def predict_svm(self, embedding):
        """Predict identity using SVM + FaceNet embedding."""
        proba = self.svm_model.predict_proba([embedding])[0]
        best_idx = np.argmax(proba)
        confidence = proba[best_idx]
        if confidence > 0.6:
            label = self.label_encoder.inverse_transform([best_idx])[0]
            return label, float(confidence)
        return None, float(confidence)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def save_student_images(self, roll_number, frames, count=30):
        """
        Save face crops from `frames` list for a given student.
        Returns number of images saved.
        """
        student_dir = os.path.join(self.image_dir, roll_number)
        os.makedirs(student_dir, exist_ok=True)
        saved = 0
        for i, frame in enumerate(frames):
            rects = self.detect_faces(frame)
            if len(rects) == 0:
                continue
            face = self.crop_face(frame, rects[0])
            face_resized = cv2.resize(face, (200, 200))
            existing = len(os.listdir(student_dir))
            filename = os.path.join(student_dir, f'{roll_number}_{existing + 1}.jpg')
            cv2.imwrite(filename, face_resized)
            saved += 1
            if saved >= count:
                break
        return saved

    def draw_face_box(self, frame, rect, label=None, confidence=None, color=(0, 255, 0)):
        """Draw bounding box and label on frame."""
        x, y, w, h = rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        if label:
            text = label
            if confidence is not None:
                text += f' ({confidence:.1f}%)'
            cv2.putText(frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame
