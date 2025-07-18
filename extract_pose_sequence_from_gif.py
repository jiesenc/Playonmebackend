# ✅ extract_pose_sequence_from_gif.py 

import cv2
import mediapipe as mp
import json
import numpy as np
import os
import sys

GIF_PATH = "andy-murray-bent-arm-forehand-slow-motion.gif"
OUTPUT_JSON = "standard_forehand_sequence.json"
VIS_OUTPUT_FOLDER = "standard_pose_frames"

if not os.path.exists(GIF_PATH):
    print(f"❌ ERROR: GIF file not found at path: {GIF_PATH}")
    sys.exit()
print(f"✅ GIF file found: {GIF_PATH}")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
os.makedirs(VIS_OUTPUT_FOLDER, exist_ok=True)

gif = cv2.VideoCapture(GIF_PATH)
if not gif.isOpened():
    print(f"❌ ERROR: Could not open GIF file: {GIF_PATH}")
    sys.exit()

frames, frame_count = [], 0
while True:
    ret, frame = gif.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame_rgb)
    frame_count += 1
gif.release()
print(f"✅ Total frames extracted: {frame_count}")

pose_sequence = []
for idx, frame in enumerate(frames):
    results = pose.process(frame)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        keypoints = [[l.x, l.y] for l in landmarks]
        pose_sequence.append(keypoints)

        # visualization
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        for l in landmarks:
            cv2.circle(annotated, (int(l.x * w), int(l.y * h)), 3, (0, 255, 0), -1)
        out_path = os.path.join(VIS_OUTPUT_FOLDER, f"frame_{idx:03d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        print(f"🟢 Frame {idx+1}: Pose saved + Visual saved.")
    else:
        pose_sequence.append([[0.0, 0.0]] * 33)
        print(f"⚠️ Frame {idx+1}: No pose detected.")

pose.close()

with open(OUTPUT_JSON, "w") as f:
    json.dump(pose_sequence, f)

print(f"\n✅ Pose sequence saved to: {OUTPUT_JSON}")
print(f"🔗 Total frames in JSON: {len(pose_sequence)}")
