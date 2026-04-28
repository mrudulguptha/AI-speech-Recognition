import cv2
import mediapipe as mp
import numpy as np
import os

# Setup MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False)

# Lip landmark indices
LIP_LANDMARKS = [
    61,146,91,181,84,17,314,405,321,375,
    291,308,324,318,402,317,14,87,178,88,
    95,185,40,39,37,0,267,269,270,409,
    415,310,311,312,13,82,81,42,183,78
]

DATASET_PATH = "dataset"
OUTPUT_PATH = "processed"

os.makedirs(OUTPUT_PATH, exist_ok=True)

def extract_from_video(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for better detection (important)
        frame = cv2.resize(frame, (640, 480))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            lips = []
            for idx in LIP_LANDMARKS:
                point = face_landmarks.landmark[idx]
                lips.append([point.x, point.y])

            sequence.append(lips)

    cap.release()

    print("Frames extracted:", len(sequence))

    if len(sequence) > 0:
        print("Saving:", save_path)
        np.save(save_path, np.array(sequence))


# Loop through dataset
for speaker in os.listdir(DATASET_PATH):
    speaker_path = os.path.join(DATASET_PATH, speaker)

    if not os.path.isdir(speaker_path):
        continue

    for file in os.listdir(speaker_path):
        #  UPDATED LINE (supports .mov also)
        if file.endswith((".mpg", ".mp4", ".mov")):
            print("FOUND VIDEO:", file)

            video_path = os.path.join(speaker_path, file)

            save_name = f"{speaker}_{file.split('.')[0]}.npy"
            save_path = os.path.join(OUTPUT_PATH, save_name)

            print(f"Processing {video_path}")
            extract_from_video(video_path, save_path)

print("Done extracting landmarks!")