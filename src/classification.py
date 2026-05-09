import numpy as np

# YOLOv8 pose keypoint indices (COCO format)
NOSE        = 0
SHOULDER_L  = 5;  SHOULDER_R = 6
ELBOW_L     = 7;  ELBOW_R    = 8
WRIST_L     = 9;  WRIST_R    = 10
HIP_L       = 11; HIP_R      = 12
KNEE_L      = 13; KNEE_R     = 14

class ShotClassifier:
    """
    Classifies padel shots from body keypoints.
    Uses wrist position relative to body center and height.
    """

    def classify(self, keypoints):
        if not keypoints or len(keypoints) < 15:
            return "unknown"

        kp = np.array(keypoints)

        # Get key joints (x, y)
        wrist_r    = kp[WRIST_R]
        wrist_l    = kp[WRIST_L]
        shoulder_r = kp[SHOULDER_R]
        shoulder_l = kp[SHOULDER_L]
        elbow_r    = kp[ELBOW_R]
        hip_r      = kp[HIP_R]
        hip_l      = kp[HIP_L]

        # Skip if keypoints not detected (zeros)
        if wrist_r[0] == 0 and wrist_r[1] == 0:
            return "unknown"

        # Body center x
        body_cx = (shoulder_r[0] + shoulder_l[0]) / 2

        # Shoulder height (y increases downward in image coords)
        shoulder_y = (shoulder_r[1] + shoulder_l[1]) / 2
        hip_y      = (hip_r[1] + hip_l[1]) / 2
        body_height = abs(hip_y - shoulder_y)

        # Use right wrist as racket hand (assume right-handed)
        # Switch to left wrist if right wrist is undetected
        if wrist_r[0] == 0:
            racket_wrist = wrist_l
        else:
            racket_wrist = wrist_r

        wrist_x = racket_wrist[0]
        wrist_y = racket_wrist[1]

        # How high is the wrist relative to shoulder?
        wrist_above_shoulder = wrist_y < shoulder_y
        wrist_height_above   = shoulder_y - wrist_y  # positive = above shoulder

        # --- Classification rules ---

        # Smash: wrist high above shoulder + elbow also raised
        if wrist_above_shoulder and wrist_height_above > body_height * 0.5:
            return "smash"

        # Serve: wrist above shoulder (less extreme than smash)
        if wrist_above_shoulder and wrist_height_above > body_height * 0.2:
            return "serve"

        # Forehand: wrist on same side as dominant shoulder (right side)
        if wrist_x > body_cx + 20:
            return "forehand"

        # Backhand: wrist crosses to non-dominant side
        if wrist_x < body_cx - 20:
            return "backhand"

        # Default: volley (wrist near center, at net)
        return "volley"

    def classify_sequence(self, keypoint_sequence):
        """Majority vote over multiple frames for stability."""
        votes = [self.classify(kp) for kp in keypoint_sequence if kp]
        votes = [v for v in votes if v != "unknown"]
        if not votes:
            return "unknown"
        return max(set(votes), key=votes.count)