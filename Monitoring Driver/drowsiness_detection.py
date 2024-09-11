import cv2
import dlib
import numpy as np
from playsound import playsound
import threading
from scipy.spatial import distance as dist
from flask import Flask, render_template, Response

# Initialize Flask app
app = Flask(__name__)

# Paths to the facial landmark predictor and Haar cascade files
predictor_path = 'shape_predictor_68_face_landmarks.dat'
detector_path = 'haarcascade_frontalface_default.xml'

# Initialize the dlib face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Initialize video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[3], mouth[9])
    B = dist.euclidean(mouth[2], mouth[10])
    C = dist.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

# Define constants with adjusted thresholds and frame count
EYE_AR_THRESH = 0.23  # Adjusted threshold for drowsiness detection
MOUTH_AR_THRESH = 0.55  # Adjusted threshold for yawning detection
CONSEC_FRAMES = 20  # Reduced number of consecutive frames for drowsiness detection

frame_count = 0

def play_alarm():
    playsound('alarm.mp3')

def generate_frames():
    global frame_count
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = np.array([(p.x, p.y) for p in shape.parts()])

            left_eye = shape[36:42]
            right_eye = shape[42:48]
            mouth = shape[48:60]

            leftEAR = eye_aspect_ratio(left_eye)
            rightEAR = eye_aspect_ratio(right_eye)
            ear = (leftEAR + rightEAR) / 2.0

            mar = mouth_aspect_ratio(mouth)

            # Debugging info
            print(f"Frame Count: {frame_count}, Left EAR: {leftEAR}, Right EAR: {rightEAR}, Average EAR: {ear}")
            print(f"Mouth MAR: {mar}")

            # Visualize EAR and MAR values
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Drowsiness Detection
            if ear < EYE_AR_THRESH:
                frame_count += 1
                if frame_count >= CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSINESS DETECTED!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if threading.active_count() == 1:  # Avoid multiple alarms
                        threading.Thread(target=play_alarm).start()
                    frame_count = 0  # Reset count after triggering the alarm
            else:
                frame_count = 0  # Reset count if EAR is above the threshold

            # Yawn Detection
            if mar > MOUTH_AR_THRESH:
                cv2.putText(frame, "YAWNING DETECTED!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Encode the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
 