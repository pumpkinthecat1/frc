# main.py
import cv2
import yt_dlp
from detect_balls import BallDetector
import pandas as pd

# 1. Setup YouTube Stream Downloader
# Replace this URL with any FRC Livestream or Match Video
youtube_url = "https://www.youtube.com/watch?v=EXAMPLE_LINK"

ydl_opts = {"format": "best", "quiet": True}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(youtube_url, download=False)
    stream_url = info['url']

# 2. Initialize the AI
detector = BallDetector(model_path="runs/detect/frc2026_balls/weights/best.pt")

# 3. Open the Livestream
cap = cv2.VideoCapture(stream_url)

data = []
frame_num = 0

print("Starting analysis... Press Ctrl+C to stop.")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Every 30 frames (roughly 1 second), check for balls
        if frame_num % 30 == 0:
            balls = detector.detect(frame)
            data.append({"timestamp_seconds": frame_num // 30, "balls_scored": balls})
            print(f"Time: {frame_num // 30}s | Balls: {balls}")

        frame_num += 1
except KeyboardInterrupt:
    print("Stopped by user.")

# 4. Save to Scouting Sheet
cap.release()
df = pd.DataFrame(data)
df.to_csv("frc_2026_scouting_report.csv", index=False)
print("Scouting report saved!")
