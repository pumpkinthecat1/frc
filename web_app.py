# web_app.py
import streamlit as st
import cv2
import yt_dlp
from detect_balls import BallDetector
import pandas as pd

st.set_page_config(page_title="FRC 2026 AI Scout", layout="wide")

st.title("🤖 FRC 2026 Video Scout AI")
st.write("Paste a YouTube link below to start scouting for balls.")

# Input for the YouTube URL
youtube_url = st.text_input("YouTube Link:", "https://www.youtube.com/watch?v=...")

if st.button("Start Scouting"):
    st.info("Connecting to YouTube stream...")
    
    # 1. Setup YouTube Stream
    ydl_opts = {"format": "best", "quiet": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        stream_url = info['url']

    # 2. Initialize AI
    # Make sure your 'best.pt' file is in your GitHub repo!
    detector = BallDetector(model_path="best.pt")

    # 3. Process Video
    cap = cv2.VideoCapture(stream_url)
    st_frame = st.empty() # Placeholder for video
    data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Only check every 30 frames to save power
        balls = detector.detect(frame)
        data.append(balls)
        
        # Show the video feed on the website
        st_frame.image(frame, channels="BGR", use_column_width=True)
        
        # Stop if user clicks a button (handled by Streamlit)
        if len(data) > 500: break 

    cap.release()
    st.success("Match Finished!")
    st.write(f"Total Balls Detected: {sum(data)}")
