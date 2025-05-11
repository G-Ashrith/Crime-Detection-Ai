from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import threading
import cv2
import yt_dlp
import time

app = Flask(__name__)
model = load_model("violence_detection_model_epoch8.h5")

# Global status text to be updated by prediction thread
status_text = "Initializing..."

# Settings
IMG_SIZE = 112
FRAMES_PER_CLIP = 32
violence_threshold = 0.7


def stream_and_predict(youtube_url):
    global status_text

    try:
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'quiet': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=False)
            video_url = info_dict['url']

        cap = cv2.VideoCapture(video_url)
        buffer = []

        while True:
            ret, frame = cap.read()
            if not ret:
                status_text = "🔴 Lost video stream"
                break

            frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            buffer.append(frame_resized)

            if len(buffer) == FRAMES_PER_CLIP:
                clip = np.array(buffer)
                clip = np.expand_dims(clip, axis=0)
                clip = clip / 255.0

                prediction = model.predict(clip)[0][0]

                if prediction > violence_threshold:
                    status_text = f"🚨 Violence Detected ({prediction:.2f})"
                else:
                    status_text = f"✅ No Violence ({prediction:.2f})"

                buffer = buffer[8:]  # Slide window by few frames

            time.sleep(1)  # Adjust for latency control

        cap.release()

    except Exception as e:
        status_text = f"Error: {str(e)}"


@app.route('/', methods=['GET', 'POST'])
def index():
    global status_text
    if request.method == 'POST':
        youtube_url = request.form['youtube_url']
        threading.Thread(target=stream_and_predict, args=(youtube_url,), daemon=True).start()
        return render_template('index.html', video_url=youtube_url, status="Started")
    return render_template('index.html')


@app.route('/status')
def get_status():
    return jsonify({"status": status_text})


if __name__ == "__main__":
    app.run(debug=True)
