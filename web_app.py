import streamlit as st
import cv2
import yt_dlp
from ultralytics import YOLO

st.title("🤖 FRC 2026 AI Video Scout")

# 1. Load the Model (Make sure best.pt is in your GitHub!)
model = YOLO("best.pt")

url = st.text_input("Paste YouTube Match Link:", "")

if url:
    st.write("Processing stream...")
    
    # Use yt-dlp to get the actual video stream URL
    ydl_opts = {"format": "best"}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        video_url = info["url"]

    # Open the video stream
    cap = cv2.VideoCapture(video_url)
    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run AI Detection
        results = model(frame, conf=0.5)
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display in the web app
        frame_placeholder.image(annotated_frame, channels="BGR")

    cap.release()
