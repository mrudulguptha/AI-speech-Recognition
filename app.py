import base64

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, render_template, request

from model.model import predict_lip_reading


app = Flask(__name__)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)


def extract_lips(frame: np.ndarray) -> np.ndarray:
    """Extract and return the lip region resized to 224x224.

    Falls back to the original resized frame when no face/lips are detected.
    """
    # MediaPipe expects RGB input.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if not results.multi_face_landmarks:
        return cv2.resize(frame, (224, 224))

    lip_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
    landmarks = results.multi_face_landmarks[0].landmark

    height, width = frame.shape[:2]
    points = []

    for idx in lip_indices:
        landmark = landmarks[idx]
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))

    if not points:
        return cv2.resize(frame, (224, 224))

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    padding = 15
    x_min = max(min(xs) - padding, 0)
    x_max = min(max(xs) + padding, width - 1)
    y_min = max(min(ys) - padding, 0)
    y_max = min(max(ys) + padding, height - 1)

    if x_min >= x_max or y_min >= y_max:
        return cv2.resize(frame, (224, 224))

    lip_crop = frame[y_min:y_max, x_min:x_max]
    if lip_crop.size == 0:
        return cv2.resize(frame, (224, 224))

    return cv2.resize(lip_crop, (224, 224))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    encoded_frames = payload.get("frames", [])

    if not isinstance(encoded_frames, list):
        return jsonify({"error": "'frames' must be a list."}), 400

    if len(encoded_frames) < 20:
        return jsonify({"prediction": "Collecting frames..."})

    processed_frames = []

    # Only process the latest 20 frames for a fixed-size sequence input.
    for encoded_frame in encoded_frames[-20:]:
        try:
            base64_data = encoded_frame.split(",", 1)[1]
            image_bytes = base64.b64decode(base64_data)
        except (IndexError, ValueError, TypeError):
            continue

        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        frame = cv2.resize(frame, (224, 224))
        lip_frame = extract_lips(frame)
        processed_frames.append(lip_frame)

    if len(processed_frames) < 20:
        return jsonify({"prediction": "Collecting frames..."})

    predicted_word = predict_lip_reading(processed_frames)
    return jsonify({"prediction": predicted_word})


if __name__ == "__main__":
    app.run(debug=True)
