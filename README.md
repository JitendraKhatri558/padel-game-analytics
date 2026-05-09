This project demonstrates:
- **Computer Vision:** Object detection, tracking, pose estimation
- **Machine Learning:** YOLOv8, rule-based classification, motion detection
- **Data Engineering:** Structured JSON/CSV output, analytics pipeline
- **Software Engineering:** Modular code, clean architecture, documentation
- **Problem-Solving:** Handling real-world challenges (ball detection on blue court, player ID stability)

**Results:** 152 shots detected, 4 players tracked, 70% ball accuracy on 5.4-min video
**Demo video:** https://drive.google.com/drive/u/1/folders/1Nnk6OKAiVMbFzti3viRxrU2rlw_x9Nb4
**Tech Stack:** Python, OpenCV, PyTorch, Ultralytics, Pandas, Matplotlib

# padel-game-analytics
## Approach & Methodology

### Pipeline Overview
1. **Frame extraction** — OpenCV reads video frame by frame at 25fps
2. **Player detection** — YOLOv8n-pose detects players and extracts 17 body keypoints per person
3. **Ball detection** — Two-stage approach:
   - Primary: Background subtraction (MOG2) detects moving circular objects
   - Fallback: HSV color filtering for white/bright objects
4. **Player tracking** — Court divided into 4 quadrants; players assigned stable IDs 0-3 by position
5. **Shot classification** — Rule-based system using wrist keypoint position:
   - Forehand: racket wrist on dominant side of body center
   - Backhand: racket wrist crosses to non-dominant side
   - Serve/Smash: wrist raised above shoulder level
   - Volley: wrist near body center at net position
6. **Hit detection** — Ball velocity change >1.8x or direction change >45 degrees triggers shot event
7. **Output** — Structured JSON/CSV with frame, timestamp, player ID, shot type, ball position

### Challenges Faced
- Ball detection is the hardest problem — padel ball is only 4-8 pixels wide at camera distance
- White ball on blue court blends with court lines — solved using morphological line removal
- Player IDs were unstable across rallies — solved using court quadrant zone assignment
- MediaPipe incompatible with Python 3.13 — replaced with YOLOv8 built-in pose keypoints
- Camera angle (top-down fisheye) distorts keypoint positions affecting classification accuracy

### Improvements I Would Make
1. Train a dedicated YOLOv8 model on padel ball dataset (Roboflow has labeled datasets)
2. Use TrackNet — a deep learning model specifically designed for sports ball tracking
3. Replace rule-based classifier with LSTM trained on labeled keypoint sequences
4. Add racket detection as a separate YOLO class for more accurate shot timing
5. Use multiple camera angles to eliminate fisheye distortion issues
6. Add rally segmentation to separate individual rallies for per-rally statistics
