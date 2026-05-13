"""
Client Script - Standalone webcam-based attendance marker
Run this on the device with the webcam (client machine).
It connects to the Flask server for recognition.
"""

import cv2
import base64
import requests
import json
import time
import sys

SERVER_URL = "http://localhost:5000"  # Change to server IP if remote


def encode_frame(frame) -> str:
    """Encode OpenCV frame to base64 JPEG."""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer).decode('utf-8')


def draw_status(frame, message: str, color=(0, 255, 0)):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, message, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def run_attendance_mode():
    """Continuously capture and recognize faces, marking attendance."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        sys.exit(1)

    print("IntelliCloud Client - Attendance Mode")
    print(f"Connected to server: {SERVER_URL}")
    print("Press 'q' to quit.")

    last_request = 0
    request_interval = 3  # seconds between recognition attempts

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()

        # Draw top bar
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (26, 35, 126), -1)
        cv2.putText(frame, "IntelliCloud Attendance System", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if current_time - last_request > request_interval:
            try:
                b64 = encode_frame(frame)
                response = requests.post(
                    f"{SERVER_URL}/api/recognize",
                    json={"image": f"data:image/jpeg;base64,{b64}"},
                    timeout=5
                )
                data = response.json()

                if data.get('success'):
                    msg = f"PRESENT: {data['student']['name']} ({data['student']['roll_number']})"
                    draw_status(frame, msg, (0, 200, 0))
                elif data.get('already_marked'):
                    msg = f"Already marked: {data['student']['name']}"
                    draw_status(frame, msg, (255, 165, 0))
                else:
                    draw_status(frame, data.get('message', 'Unknown'), (0, 0, 200))

                last_request = current_time

            except requests.exceptions.ConnectionError:
                draw_status(frame, "Server not reachable!", (0, 0, 255))
            except Exception as e:
                draw_status(frame, f"Error: {str(e)[:50]}", (0, 0, 255))

        cv2.imshow("IntelliCloud - Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def register_student_mode():
    """Capture 30 images for a new student and send to server."""
    roll = input("Enter Roll Number: ").strip()
    name = input("Enter Student Name: ").strip()
    dept = input("Enter Department (optional): ").strip()
    year = input("Enter Year (optional): ").strip()

    # Register in DB
    payload = {"roll_number": roll, "name": name, "department": dept or None,
               "year": int(year) if year else None}
    res = requests.post(f"{SERVER_URL}/api/students/register", json=payload)
    print(res.json().get('message'))

    print("Starting face capture... Press 'c' to capture, 'q' to quit.")
    cap = cv2.VideoCapture(0)
    frames = []

    while len(frames) < 30:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, f"Captured: {len(frames)}/30 - Press 'c'",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            frames.append(frame.copy())
            print(f"Captured {len(frames)}/30")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not frames:
        print("No images captured.")
        return

    # Upload
    images_b64 = [f"data:image/jpeg;base64,{encode_frame(f)}" for f in frames]
    res = requests.post(f"{SERVER_URL}/api/capture",
                        json={"roll_number": roll, "images": images_b64})
    print(res.json().get('message'))

    # Train
    print("Training model...")
    res = requests.post(f"{SERVER_URL}/api/train")
    print(res.json().get('message'))


if __name__ == '__main__':
    print("IntelliCloud Client")
    print("1. Attendance Mode")
    print("2. Register Student Mode")
    choice = input("Select mode (1/2): ").strip()
    if choice == '1':
        run_attendance_mode()
    elif choice == '2':
        register_student_mode()
    else:
        print("Invalid choice.")
