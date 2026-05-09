import cv2
from src.detection import PadelDetector

VIDEO_PATH = "data/raw/input_sample_video.mp4"

def test_on_frames():
    detector = PadelDetector()
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("ERROR: Could not open video file!")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video loaded: {total} frames at {fps:.1f} fps")
    print("Processing frames 0-500...\n")

    ball_detections = 0
    # Save these specific frames as images
    save_frames = {100, 200, 300, 400, 500}

    for frame_num in range(501):
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.detect_frame(frame)

        if results["ball"]:
            ball_detections += 1

        if frame_num in save_frames:
            print(f"--- Frame {frame_num} ({frame_num/fps:.1f}s) ---")
            print(f"  Players : {len(results['players'])}")
            print(f"  Ball    : {results['ball']}")

            annotated = detector.draw_detections(frame, results)
            cv2.imwrite(f"frame_{frame_num}_annotated.jpg", annotated)
            print(f"  Saved   : frame_{frame_num}_annotated.jpg")

    cap.release()
    detection_rate = (ball_detections / 500) * 100
    print(f"\nBall detected in {ball_detections}/500 frames ({detection_rate:.1f}%)")
    print("Open the saved jpg files to check accuracy!")

if __name__ == "__main__":
    test_on_frames()