import os
import cv2
import numpy as np
from flask import Flask, render_template, request, Response
from tensorflow.keras.models import load_model

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
model = load_model('violence_detection_model_epoch8.h5')

IMG_SIZE = 112
FRAMES_PER_CLIP = 32
VIOLENCE_THRESHOLD = 0.7

def detect_violence(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    violence_detected = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frames.append(resized)

    cap.release()

    for i in range(0, len(frames) - FRAMES_PER_CLIP + 1, FRAMES_PER_CLIP):
        clip = frames[i:i + FRAMES_PER_CLIP]
        if len(clip) == FRAMES_PER_CLIP:
            input_clip = np.expand_dims(np.array(clip) / 255.0, axis=0)
            prediction = model.predict(input_clip, verbose=0)[0][0]
            if prediction > VIOLENCE_THRESHOLD:
                violence_detected = True
                break

    return violence_detected

def stream_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        success, frame = cap.read()
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['video']
    filename = file.filename
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(video_path)

    violence = detect_violence(video_path)
    label = "🚨 Violence Detected" if violence else "✅ No Violence Detected"
    label_class = "alert" if violence else "safe"

    return render_template('index.html', filename=filename, label=label, label_class=label_class)

@app.route('/video_feed/<filename>')
def video_feed(filename):
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    return Response(stream_video(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
