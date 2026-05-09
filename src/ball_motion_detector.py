import cv2
import numpy as np
from collections import deque

class MotionBallDetector:
    def __init__(self):
        # More history = better background model
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=25,
            detectShadows=False
        )
        self.frame_count = 0
        self.prev_ball = None
        self.ball_history = deque(maxlen=10)

    def detect(self, frame):
        fg_mask = self.bg_subtractor.apply(frame)
        self.frame_count += 1

        # Need more warmup frames
        if self.frame_count < 60:
            return None

        # Stronger cleanup
        kernel_small = np.ones((2, 2), np.uint8)
        kernel_med   = np.ones((4, 4), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel_small)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_med)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []

        for c in contours:
            area = cv2.contourArea(c)

            # Ball area range in pixels (tune if needed)
            if not (5 < area < 400):
                continue

            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.35:
                continue

            (x, y), radius = cv2.minEnclosingCircle(c)

            # Bonus: if this position is close to last known ball position
            continuity_bonus = 0
            if self.prev_ball:
                dist = np.sqrt(
                    (x - self.prev_ball["x"])**2 +
                    (y - self.prev_ball["y"])**2
                )
                # Ball shouldn't teleport — max ~150px between frames
                if dist < 150:
                    continuity_bonus = 0.3
                else:
                    continue  # too far from last position, skip

            score = circularity + continuity_bonus
            candidates.append({
                "x": round(float(x), 1),
                "y": round(float(y), 1),
                "radius": round(float(radius), 1),
                "circularity": round(circularity, 3),
                "score": score,
                "method": "motion"
            })

        if not candidates:
            # Ball not found this frame — allow 5 frame gap
            self.prev_ball = None
            return None

        # Pick best candidate
        best = max(candidates, key=lambda c: c["score"])
        self.prev_ball = best
        self.ball_history.append((best["x"], best["y"]))
        return best

    def get_smoothed_position(self):
        """Returns average position over last N detections."""
        if not self.ball_history:
            return None
        xs = [p[0] for p in self.ball_history]
        ys = [p[1] for p in self.ball_history]
        return {
            "x": round(sum(xs) / len(xs), 1),
            "y": round(sum(ys) / len(ys), 1)
        }