
🚨 Crime Detection AI – Violence Detection in Videos & Live Streams
==================================================================

A Flask-based AI web application that detects **violence in uploaded videos** and **live YouTube streams** using a pre-trained deep learning model (`.h5`). The app processes frames in real time and flags clips based on a set threshold.

Project Overview
----------------
**Crime Detection AI** is an intelligent system that:
- Accepts local video uploads for violence detection.
- Streams and analyzes YouTube Live content for real-time violence identification.
- Utilizes a TensorFlow `.h5` model trained to classify violent vs. non-violent behavior.
- Provides instant visual feedback on detection results.

Features
--------
- 🎥 Upload Video: Users can upload `.mp4` or similar formats to detect violence.
- 📡 Live YouTube Stream Analysis: Paste a YouTube live URL and get real-time predictions.
- ⚠️ Threshold-based Alerts: If violence probability > `0.7`, the system raises an alert.
- 🖼️ Streamed Video Playback: Play video with detection results shown alongside.
- 🧠 Deep Learning Powered: Uses a trained ConvLSTM model on frame sequences.

Tech Stack
----------
| Tool / Library        | Purpose                          |
|-----------------------|----------------------------------|
| Flask                 | Web framework                    |
| TensorFlow/Keras      | Model loading & inference        |
| OpenCV                | Video frame extraction           |
| NumPy                 | Clip array handling              |
| yt_dlp                | YouTube live video streaming     |
| HTML + CSS            | Frontend                         |

Project Structure
-----------------
Crime-Detection-Ai/
│
├── static/
│   └── uploads/
│       └── styles.css
│
├── templates/
│   └── index.html
│
├── violence_detection_model_epoch8.h5
├── app.py (video upload interface)
├── livestream.py (YouTube Live detection)
├── README.txt
└── Execution video.mp4 (local demo - not hosted on GitHub)

How to Run Locally
------------------
1. Clone the repo:
   git clone https://github.com/your-username/Crime-Detection-Ai.git
   cd Crime-Detection-Ai

2. Install dependencies:
   pip install -r requirements.txt

   Create `requirements.txt` with:
   flask
   opencv-python
   numpy
   tensorflow
   yt-dlp

3. Run Video Upload Detection:
   python app.py

4. Run YouTube Live Stream Detection:
   python livestream.py

Model Info
----------
- File: `violence_detection_model_epoch8.h5`
- Input: 32-frame clips of size `112x112`
- Threshold: `0.7` for violence detection
- Inference: Normalized frame sequence → prediction → real-time status
- **Dataset Used:** [Real-Life Violence Situations Dataset (Kaggle)](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)


Demo
----
🖥️ Execution Video (local path):  
['Execution video.mp4'](https://github.com/G-Ashrith/Crime-Detection-Ai/blob/main/Execution%20video.mp4)


Sample Output
-------------
- ✅ No Violence Detected (0.23)
- 🚨 Violence Detected (0.89)

Limitations
-----------
- Doesn’t support high-latency live streams.
- Frame buffer may lag with poor bandwidth.
- Model accuracy depends on training dataset variety.

Future Improvements
-------------------
- Add real-time bounding box visualizations.
- Extend detection to other crimes (e.g., theft, abuse).
- Replace YouTube streaming with RTSP/CCTV input.
- Host model with TensorFlow Serving for faster inference.

License
-------
MIT License — Feel free to use, modify, and share.

Author
------
G.Ashrith — Passionate about AI, safety tech, and real-world solutions.
