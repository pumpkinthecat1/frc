# app.py
from flask import Flask, request, jsonify
import yt_dlp
from detect_balls import BallDetector

app = Flask(__name__)
detector = BallDetector(model_path="weights/best.pt")

@app.route('/scout', methods=['POST'])
def scout_video():
    data = request.json
    url = data.get('url')
    
    # Logic to pull stream and run detector
    # (Similar to your main.py logic)
    
    return jsonify({"status": "success", "message": "Analysis started"})

if __name__ == '__main__':
    app.run(debug=True)
Step 2: The Easiest "Web" Way (Streamlit)
If you want a website without learning HTML/CSS, use Streamlit. It’s a tool specifically for AI people to make web apps in pure Python.

Create a file called web_app.py:

Python
import streamlit as st
from main import run_analysis # You'd wrap your main logic in a function

st.title("FRC 2026 AI Scout")
url = st.text_input("Paste YouTube Match Link:")

if st.button("Analyze Match"):
    st.write("Watching stream... please wait.")
    results = run_analysis(url)
    st.table(results)
