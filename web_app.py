import streamlit as st
import cv2
import yt_dlp
from ultralytics import YOLO
import os

# This prevents the "Monitor" error on servers
os.environ["QT_QPA_PLATFORM"] = "offscreen"

st.title("🤖 FRC AI Video Scout (Test Mode)")

# 1. Load a pre-trained model (No file upload needed!)
# This model knows "person", "sports ball", "bottle", etc.
model = YOLO("yolov8n.pt") 

url = st.text_input("Paste YouTube Link (Match or Livestream):", "")

if url:
    st.write("⏳ Connecting to stream... please wait.")
    try:
        ydl_opts = {"format": "best[ext=mp4]/best", "quiet": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            video_url = info["url"]

        cap = cv2.VideoCapture(video_url)
        frame_placeholder = st.empty()
        stop_button = st.button("Stop Scouting")

        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                break

            # Run AI (looking for 'person' or 'sports ball')
            results = model(frame, conf=0.3)
            
            # Plot the boxes
            annotated_frame = results[0].plot()

            # Show in Streamlit
            frame_placeholder.image(annotated_frame, channels="BGR")

        cap.release()
    except Exception as e:
        st.error(f"Error: {e}")
