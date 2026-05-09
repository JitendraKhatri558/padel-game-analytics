import sys
print(f"Python: {sys.version}")

import cv2
print(f"OpenCV: {cv2.__version__}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

import ultralytics
print(f"Ultralytics: {ultralytics.__version__}")

import mediapipe
print(f"MediaPipe: {mediapipe.__version__}")

import numpy as np
print(f"NumPy: {np.__version__}")

import pandas as pd
print(f"Pandas: {pd.__version__}")

print("\nAll packages installed successfully!")