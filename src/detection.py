import cv2
import numpy as np
from ultralytics import YOLO
from src.ball_motion_detector import MotionBallDetector

COURT_X_MIN = 234
COURT_X_MAX = 1180
COURT_Y_MIN = 150
COURT_Y_MAX = 715

class PadelDetector:
    def __init__(self):
        print("Loading models...")
        self.player_model = YOLO("yolov8n-pose.pt")
        self.motion_detector = MotionBallDetector()
        print("Models loaded successfully!")

    def detect_frame(self, frame):
        results = {
            "players": [],
            "ball": None,
        }

        # Player + pose detection
        player_results = self.player_model(frame, classes=[0], verbose=False)
        for r in player_results:
            boxes = r.boxes
            keypoints = r.keypoints
            if boxes is None or keypoints is None:
                continue
            for box, kps in zip(boxes, keypoints):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # Skip players outside court
                if not (COURT_X_MIN < cx < COURT_X_MAX):
                    continue
                if not (COURT_Y_MIN < cy < COURT_Y_MAX):
                    continue
                if confidence < 0.25:
                    continue

                kp_list = kps.xy[0].tolist()
                results["players"].append({
                    "bbox": [round(x1,1), round(y1,1), round(x2,1), round(y2,1)],
                    "confidence": round(confidence, 3),
                    "keypoints": kp_list,
                    "player_id": None
                })

        # Ball detection — motion first, color as fallback
        motion_ball = self.motion_detector.detect(frame)
        if motion_ball:
            bx, by = motion_ball["x"], motion_ball["y"]
            if COURT_X_MIN < bx < COURT_X_MAX and COURT_Y_MIN < by < COURT_Y_MAX:
                results["ball"] = motion_ball

        if not results["ball"]:
            color_ball = self._detect_ball_hsv(frame)
            if color_ball:
                bx, by = color_ball["x"], color_ball["y"]
                if COURT_X_MIN < bx < COURT_X_MAX and COURT_Y_MIN < by < COURT_Y_MAX:
                    results["ball"] = color_ball

        return results

    def _detect_ball_hsv(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 200])
        upper = np.array([179, 50, 255])
        mask = cv2.inRange(hsv, lower, upper)

        kernel_hline = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        kernel_vline = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        mask_no_hlines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_hline)
        mask_no_vlines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_vline)
        lines_mask = cv2.bitwise_or(mask_no_hlines, mask_no_vlines)
        clean_mask = cv2.subtract(mask, lines_mask)

        kernel = np.ones((3, 3), np.uint8)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        best_candidate = None
        best_score = 0

        for c in contours:
            area = cv2.contourArea(c)
            if not (10 < area < 800):
                continue
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.4:
                continue
            score = circularity * min(area, 200) / 200
            if score > best_score:
                best_score = score
                (x, y), radius = cv2.minEnclosingCircle(c)
                best_candidate = {
                    "x": round(float(x), 1),
                    "y": round(float(y), 1),
                    "radius": round(float(radius), 1),
                    "circularity": round(circularity, 3)
                }
        return best_candidate

    def draw_detections(self, frame, results):
        # Draw court boundary
        cv2.rectangle(frame,
            (COURT_X_MIN, COURT_Y_MIN),
            (COURT_X_MAX, COURT_Y_MAX),
            (255, 100, 0), 2)
        cv2.putText(frame, "Court ROI", (COURT_X_MIN + 5, COURT_Y_MIN - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

        # Draw players
        for i, player in enumerate(results["players"]):
            x1, y1, x2, y2 = [int(v) for v in player["bbox"]]
            conf = player["confidence"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            pid = player['player_id'] if player['player_id'] is not None else i
            cv2.putText(frame, f"Player {pid} ({conf:.2f})", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            for j, kp in enumerate(player["keypoints"]):
                x, y = int(kp[0]), int(kp[1])
                if x == 0 and y == 0:
                    continue
                color = (0, 0, 255) if j in [15, 16] else (255, 255, 0)
                cv2.circle(frame, (x, y), 4, color, -1)

        # Draw ball
        if results["ball"]:
            bx = int(results["ball"]["x"])
            by = int(results["ball"]["y"])
            br = max(int(results["ball"]["radius"]), 8)
            cv2.circle(frame, (bx, by), br, (0, 255, 255), 2)
            cv2.circle(frame, (bx, by), 3, (0, 255, 255), -1)
            method = results["ball"].get("method", "color")
            cv2.putText(frame, f"Ball ({method})", (bx + br + 5, by),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        return frame