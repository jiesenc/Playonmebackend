import cv2
import mediapipe as mp
import numpy as np
import os
import uuid
import json
import subprocess

STANDARD_SEQUENCE_PATH = "standard_forehand_sequence.json"
REPORT_LOG_PATH = "report_log.json"

mp_pose = mp.solutions.pose
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

ANGLE_PAIRS = [
    (11, 13, 15), (12, 14, 16), (23, 25, 27),
    (24, 26, 28), (11, 23, 25), (12, 24, 26)
]

STANDARD_ANGLES = [160, 140, 170, 170, 150, 150]

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def extract_pose_angles_and_keypoints(frame, pose_model):
    results = pose_model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return None, None
    landmarks = results.pose_landmarks.landmark
    h, w = frame.shape[:2]
    keypoints = [(int(l.x * w), int(l.y * h)) for l in landmarks]
    angles = [calculate_angle(keypoints[a], keypoints[b], keypoints[c]) for a, b, c in ANGLE_PAIRS]
    return angles, keypoints

def draw_keypoints(image, keypoints, color=(0, 255, 0)):
    for (x, y) in keypoints:
        if x > 0 and y > 0:
            cv2.circle(image, (int(x), int(y)), 4, color, -1)

def draw_connections(image, keypoints, connections, color=(0, 255, 0)):
    for start_idx, end_idx in connections:
        if 0 <= start_idx < len(keypoints) and 0 <= end_idx < len(keypoints):
            x1, y1 = keypoints[start_idx]
            x2, y2 = keypoints[end_idx]
            if all(i > 0 for i in (x1, y1, x2, y2)):
                cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

def convert_to_web_compatible(input_path, output_path):
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-vf", "setpts=2.0*PTS",
            "-vcodec", "libx264", "-acodec", "aac",
            "-movflags", "faststart",
            output_path
        ], check=True)
    except Exception as e:
        print("FFmpeg conversion error:", e)

def analyze_video(video_path, upload_folder, model_name="default"):
    os.makedirs(upload_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, "Video error", {"pro": "N/A", "match": "0%"}, "error.mp4", None, "error.mp4"

    try:
        with open(STANDARD_SEQUENCE_PATH, "r") as f:
            standard_sequence = json.load(f)
    except:
        return 0, "Missing standard sequence file", {"pro": "N/A", "match": "0%"}, "error.mp4", None, "error.mp4"

    if model_name == "federer":
        custom_angles = [158, 135, 168, 165, 148, 152]
        pro_name = "Roger Federer"
    elif model_name == "nadal":
        custom_angles = [165, 145, 160, 175, 149, 149]
        pro_name = "Rafael Nadal"
    else:
        custom_angles = STANDARD_ANGLES
        pro_name = "Standard Model"

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    ret, frame = cap.read()
    if not ret:
        return 0, "Read frame failed", {"pro": pro_name, "match": "0%"}, "error.mp4", None, "error.mp4"

    h, w = frame.shape[:2]
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(upload_folder, f"{base}_overlay_{uuid.uuid4().hex}.mp4")
    err_path = os.path.join(upload_folder, f"{base}_error_{uuid.uuid4().hex}.mp4")
    web_path = out_path.replace(".mp4", "_web.mp4")
    web_err = err_path.replace(".mp4", "_web.mp4")

    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    err_out = cv2.VideoWriter(err_path, cv2.VideoWriter_fourcc(*'mp4v'), fps // 2, (w, h))
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    angle_diffs = []
    report_log = []
    frame_idx = 0
    total_std_frames = len(standard_sequence)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        overlay = frame.copy()
        angles, keypoints = extract_pose_angles_and_keypoints(frame, pose)
        hints = []

        std_idx = frame_idx % total_std_frames
        std_kps = [(int(p[0] * w), int(p[1] * h)) for p in standard_sequence[std_idx]]

        if angles:
            diff = np.mean([abs(a - s) for a, s in zip(angles, custom_angles)])
            angle_diffs.append(diff)
            if abs(angles[0] - custom_angles[0]) > 30:
                hints.append("Elbow too closed")
            if abs(angles[2] - custom_angles[2]) > 20:
                hints.append("Shoulder not rotated")
            if abs(angles[4] - custom_angles[4]) > 20:
                hints.append("Hip not engaged")

        if keypoints:
            draw_keypoints(overlay, keypoints)
            draw_connections(overlay, keypoints, POSE_CONNECTIONS)
        for x, y in std_kps:
            if x > 0 and y > 0:
                cv2.circle(overlay, (x, y), 5, (200, 200, 255), -1)

        out.write(cv2.addWeighted(frame, 0.6, overlay, 0.4, 0))

        if hints:
            report_log.append({"frame": frame_idx, "hints": hints})
            for _ in range(5):
                err_out.write(overlay)

        frame_idx += 1

    cap.release()
    out.release()
    err_out.release()
    pose.close()

    with open(REPORT_LOG_PATH, "w") as f:
        json.dump(report_log, f, indent=2)

    avg_diff = np.mean(angle_diffs) if angle_diffs else 100
    score = max(0, 100 - avg_diff)
    feedback = "Good job!" if score > 85 else "Try to improve angle match"
    pro_result = {"pro": pro_name, "match": f"{round(100 - avg_diff, 1)}%"}

    convert_to_web_compatible(out_path, web_path)
    convert_to_web_compatible(err_path, web_err)

    return score, feedback, pro_result, os.path.basename(web_path), None, os.path.basename(web_err)
