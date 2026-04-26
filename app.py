import atexit
from collections import deque
import threading

import cv2
from flask import Flask, Response, jsonify, render_template

from model.model import predict_lip_reading


app = Flask(__name__)

# Keep the latest 20 frames for prediction requests.
frame_buffer = deque(maxlen=20)
buffer_lock = threading.Lock()

# Single webcam capture shared by stream and prediction routes.
camera = cv2.VideoCapture(0)


if not camera.isOpened():
    raise RuntimeError("Could not open webcam. Please ensure a camera is connected.")


def _release_camera_on_exit():
    if camera.isOpened():
        camera.release()


atexit.register(_release_camera_on_exit)


def generate_video_stream():
    """Yield webcam frames as MJPEG for the /video endpoint."""
    while True:
        success, frame = camera.read()
        if not success:
            continue

        with buffer_lock:
            # Save a copy so prediction is isolated from streaming mutations.
            frame_buffer.append(frame.copy())

        success, encoded = cv2.imencode(".jpg", frame)
        if not success:
            continue

        frame_bytes = encoded.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video_feed():
    return Response(
        generate_video_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/predict", methods=["GET"])
def predict():
    with buffer_lock:
        frames_for_prediction = list(frame_buffer)

    if not frames_for_prediction:
        return jsonify({"error": "No frames available yet. Please wait for the video feed."}), 400

    predicted_word = predict_lip_reading(frames_for_prediction)
    return jsonify({"prediction": predicted_word, "frames_used": len(frames_for_prediction)})


if __name__ == "__main__":
    app.run(debug=True)
