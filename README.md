# 🧠 IntelliCloud - Face Detecting Device for Attendance

**Bachelor of Technology – Artificial Intelligence and Machine Learning**  
Malla Reddy University | Batch 15 | Department of AIML

---

## 👥 Team Members

| Name | Roll Number |
|------|------------|
| Allenki Harshini | 2111CS020162 |
| Harshini Pedapudi | 2111CS020163 |
| A Harshini | 2111CS020164 |
| K Harshith | 2111CS020165 |
| T Harshitha | 2111CS020166 |

**Project Guide:** Prof. A Kalyani

---

## 📌 Overview

IntelliCloud is an AI-powered attendance system that uses **facial recognition** to automatically mark student attendance. It combines:

- **FaceNet** – Deep learning model for face feature extraction
- **SVM** – Classification of recognized identities
- **LBPH (OpenCV)** – Lightweight fallback recognizer
- **Flask** – Backend REST API
- **SQLite / PostgreSQL** – Attendance database

---

## 🏗️ Architecture

```
Client (Webcam) ──► Flask Server ──► Face Recognition Engine
                         │                    │
                         ▼                    ▼
                    SQLite DB           LBPH / FaceNet + SVM
                         │
                         ▼
                   Web Dashboard (HTML/CSS/JS)
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/IntelliCloud.git
cd IntelliCloud
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
```bash
cp .env.example .env
# Edit .env with your settings
```

### 5. Run the server
```bash
python app.py
```

Open browser at: **http://localhost:5000**

---

## 🚀 Usage

### Web Interface (Recommended)
1. Go to `/register` → Enter student details → Capture 30 face images
2. Go to `/attendance` → Start camera → System auto-recognizes and marks attendance
3. Go to `/dashboard` → View records, export CSV

### Client Script (Standalone)
```bash
python client/client.py
# Choose: 1 (Attendance Mode) or 2 (Register Student Mode)
```

---

## 🔬 Algorithms Used

| Algorithm | Purpose |
|-----------|---------|
| **Haar Cascade** | Fast face detection |
| **LBPH** | Local Binary Pattern Histograms – lightweight recognition |
| **FaceNet (CNN)** | Deep face embeddings for high accuracy |
| **SVM** | Classifies face embeddings into student identities |

---

## 📁 Project Structure

```
IntelliCloud/
├── app.py                  # Flask app entry point
├── requirements.txt
├── .env.example
├── database/
│   └── db.py               # SQLAlchemy models (Student, Attendance)
├── models/
│   └── face_recognition.py # Recognition engine (LBPH + SVM)
├── server/
│   └── routes.py           # REST API routes
├── utils/
│   ├── attendance_service.py
│   └── student_service.py
├── client/
│   └── client.py           # Standalone webcam client
├── templates/              # HTML pages
│   ├── base.html
│   ├── index.html
│   ├── register.html
│   ├── attendance.html
│   └── dashboard.html
├── static/
│   ├── css/style.css
│   └── js/main.js
├── TrainingImage/          # Stored face images (per student)
└── TrainingImageLabel/     # Trained model files
```

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/students/register` | Register new student |
| GET | `/api/students` | List all students |
| POST | `/api/capture` | Save face images |
| POST | `/api/train` | Retrain recognition model |
| POST | `/api/recognize` | Recognize face & mark attendance |
| GET | `/api/attendance/today` | Today's attendance |
| GET | `/api/attendance/summary` | Summary stats |
| GET | `/api/attendance/export` | Download CSV |

---

## 📄 License

Academic project – Malla Reddy University, 2024.
