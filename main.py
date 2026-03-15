# main.py
import cv2
from detect_balls import BallDetector
import pandas as pd
import os

# Create output folder if it doesn't exist
os.makedirs("output", exist_ok=True)

# Initialize ball detector (will train model later)
detector = BallDetector(model_path="runs/train/frc2026_balls/weights/best.pt")

# Replace with your match video file
video_path = "match.mp4"
cap = cv2.VideoCapture(video_path)

frame_num = 0
data = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    balls_detected = detector.detect(frame)
    data.append({"frame": frame_num, "balls_detected": balls_detected})
    frame_num += 1

cap.release()

# Export results to CSV
df = pd.DataFrame(data)
df.to_csv("output/scouting.csv", index=False)

print("Scouting CSV saved to output/scouting.csv")
