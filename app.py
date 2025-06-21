from flask import Flask, request, jsonify, send_from_directory
from pose_analysis import analyze_video
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    video = request.files["video"]
    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)

    score, feedback, pro_match, overlay_video, _, error_clip = analyze_video(video_path, UPLOAD_FOLDER)

    return jsonify({
        "score": score,
        "feedback": feedback,
        "pro_match": pro_match,
        "overlay_url": f"/static/{overlay_video}",
        "error_clip_url": f"/static/{error_clip}"
    })

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run()