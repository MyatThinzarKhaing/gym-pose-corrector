# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import math
import time

app = Flask(__name__)
CORS(app)

# =========================
# MEDIAPIPE SETUP
# =========================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# WORKOUT SESSION
# =========================
session = {
    "running": False,
    "exercise": "pushup",
    "target_reps": 10,
    "counter": 0,
    "stage": "down",
    "phase": "idle",  # idle | active | rest | finished
    "rest_start": None
}

REST_SECONDS = 10


# =========================
# ANGLE CALCULATION
# =========================
def calculate_angle(a, b, c):
    x1, y1 = a[:2]
    x2, y2 = b[:2]
    x3, y3 = c[:2]

    angle = math.degrees(
        math.atan2(y3 - y2, x3 - x2) -
        math.atan2(y1 - y2, x1 - x2)
    )

    if angle < 0:
        angle += 360

    return angle


# =========================
# PUSHUP DETECTION
# =========================
def process_pushup(frame, landmarks, w, h):
    shoulder = (
        int(landmarks[12].x * w),
        int(landmarks[12].y * h)
    )

    elbow = (
        int(landmarks[14].x * w),
        int(landmarks[14].y * h)
    )

    wrist = (
        int(landmarks[16].x * w),
        int(landmarks[16].y * h)
    )

    angle = calculate_angle(shoulder, elbow, wrist)

    feedback = []

    # DOWN
    if angle < 90:
        session["stage"] = "down"

    # UP + COUNT REP
    if angle > 160 and session["stage"] == "down":
        session["stage"] = "up"
        session["counter"] += 1
        feedback.append("Good Push-up!")

    # FORM CHECKS
    if angle > 170 and session["stage"] == "down":
        feedback.append("Go Lower")

    if angle < 50:
        feedback.append("Do not go too low")

    return feedback, angle


# =========================
# BICEP CURL DETECTION
# =========================
def process_bicep(frame, landmarks, w, h):
    shoulder = (
        int(landmarks[12].x * w),
        int(landmarks[12].y * h)
    )

    elbow = (
        int(landmarks[14].x * w),
        int(landmarks[14].y * h)
    )

    wrist = (
        int(landmarks[16].x * w),
        int(landmarks[16].y * h)
    )

    hip = (
        int(landmarks[24].x * w),
        int(landmarks[24].y * h)
    )

    angle = calculate_angle(shoulder, elbow, wrist)

    feedback = []

    # CURL UP
    if angle < 50:
        session["stage"] = "up"

    # LOWER DOWN + COUNT
    if angle > 150 and session["stage"] == "up":
        session["stage"] = "down"
        session["counter"] += 1
        feedback.append("Good Curl!")

    # FORM CHECKS
    if abs(elbow[0] - shoulder[0]) > 60:
        feedback.append("Keep elbow close")

    if abs(shoulder[0] - hip[0]) > 40:
        feedback.append("Do not swing body")

    if angle > 350 and session["stage"] == "up":
        feedback.append("Lift higher")

    return feedback, angle


# =========================
# FRAME PROCESSING
# =========================
def process_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb)

    if not results.pose_landmarks:
        return {
            "feedback": ["No body detected"],
            "angle": 0
        }

    h, w, _ = frame.shape

    landmarks = results.pose_landmarks.landmark

    # =========================
    # PUSHUP
    # =========================
    if session["exercise"] == "pushup":
        feedback, angle = process_pushup(
            frame,
            landmarks,
            w,
            h
        )

    # =========================
    # BICEP CURL
    # =========================
    else:
        feedback, angle = process_bicep(
            frame,
            landmarks,
            w,
            h
        )

    return {
        "feedback": feedback,
        "angle": angle
    }


# =========================
# START WORKOUT
# =========================
@app.route("/start", methods=["POST"])
def start_workout():
    session["running"] = True
    session["exercise"] = "pushup"
    session["counter"] = 0
    session["stage"] = "down"
    session["phase"] = "active"

    return jsonify({
        "message": "Workout Started"
    })


# =========================
# PAUSE WORKOUT
# =========================
@app.route("/pause", methods=["POST"])
def pause_workout():
    session["running"] = False

    return jsonify({
        "message": "Workout Paused"
    })


# =========================
# RESUME WORKOUT
# =========================
@app.route("/resume", methods=["POST"])
def resume_workout():
    session["running"] = True

    return jsonify({
        "message": "Workout Resumed"
    })


# =========================
# ANALYZE FRAME
# =========================
@app.route("/analyze", methods=["POST"])
def analyze():

    # =========================
    # REST TIMER
    # =========================
    if session["phase"] == "rest":

        elapsed = time.time() - session["rest_start"]

        if elapsed >= REST_SECONDS:

            session["exercise"] = "bicep"
            session["counter"] = 0
            session["stage"] = "down"
            session["phase"] = "active"

        return jsonify({
            "feedback": ["Rest Time"],
            "angle": 0,
            "counter": session["counter"],
            "exercise": session["exercise"],
            "phase": session["phase"],
            "rest_remaining": max(
                0,
                int(REST_SECONDS - elapsed)
            )
        })

    file = request.files["frame"]

    npimg = np.frombuffer(file.read(), np.uint8)

    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    result = process_frame(frame)

    # =========================
    # PUSHUP COMPLETE
    # =========================
    if (
        session["exercise"] == "pushup"
        and session["counter"] >= session["target_reps"]
    ):

        session["phase"] = "rest"
        session["rest_start"] = time.time()

    # =========================
    # BICEP COMPLETE
    # =========================
    elif (
        session["exercise"] == "bicep"
        and session["counter"] >= session["target_reps"]
    ):

        session["phase"] = "finished"

    return jsonify({
        "feedback": result["feedback"],
        "angle": int(result["angle"]),
        "counter": session["counter"],
        "exercise": session["exercise"],
        "phase": session["phase"]
    })


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )