import numpy as np
from collections import deque

class BallTracker:
    """Smooths ball position across frames and detects hit events."""

    def __init__(self, history_len=30):
        self.positions  = deque(maxlen=history_len)
        self.frames     = deque(maxlen=history_len)
        self.velocities = deque(maxlen=history_len)

    def update(self, detection, frame_num):
        if detection:
            x, y = detection["x"], detection["y"]
            prev = [p for p in self.positions if p is not None]
            vel  = np.sqrt((x-prev[-1][0])**2 + (y-prev[-1][1])**2) if prev else 0.0
            self.positions.append((x, y))
            self.velocities.append(vel)
        else:
            self.positions.append(None)
            self.velocities.append(0.0)
        self.frames.append(frame_num)

    def detect_hit_event(self):
        valid_vels = [v for v in self.velocities if v > 0]
        if len(valid_vels) < 5:
            return False
        recent   = np.mean(list(valid_vels)[-3:])
        previous = np.mean(list(valid_vels)[-8:-3])
        if previous > 0 and recent > previous * 1.8:
            return True
        valid_pos = [p for p in self.positions if p is not None]
        if len(valid_pos) >= 6:
            v1 = np.array(valid_pos[-3]) - np.array(valid_pos[-6])
            v2 = np.array(valid_pos[-1]) - np.array(valid_pos[-3])
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 5 and n2 > 5:
                cos_a = np.clip(np.dot(v1,v2)/(n1*n2), -1, 1)
                if np.degrees(np.arccos(cos_a)) > 45:
                    return True
        return False

    def get_trajectory(self):
        return [(f, p) for f, p in zip(self.frames, self.positions) if p]


# Court boundary imported for position-based assignment
COURT_X_MIN = 234
COURT_X_MAX = 1180
COURT_MID_X = (COURT_X_MIN + COURT_X_MAX) // 2   # 707
COURT_Y_MIN = 150
COURT_Y_MAX = 715
COURT_MID_Y = (COURT_Y_MIN + COURT_Y_MAX) // 2   # 432

class PlayerTracker:
    """
    Assigns stable IDs based on court position zones.
    Padel court has 4 quadrants — one player per quadrant max.

    IDs:
      0 = near left  (bottom-left quadrant)
      1 = near right (bottom-right quadrant)
      2 = far left   (top-left quadrant)
      3 = far right  (top-right quadrant)
    """

    def __init__(self):
        # Fixed zone centers
        self.zone_centers = {
            0: (COURT_X_MIN + (COURT_MID_X - COURT_X_MIN)//2,
                COURT_MID_Y + (COURT_Y_MAX - COURT_MID_Y)//2),   # near-left
            1: (COURT_MID_X + (COURT_X_MAX - COURT_MID_X)//2,
                COURT_MID_Y + (COURT_Y_MAX - COURT_MID_Y)//2),   # near-right
            2: (COURT_X_MIN + (COURT_MID_X - COURT_X_MIN)//2,
                COURT_Y_MIN + (COURT_MID_Y - COURT_Y_MIN)//2),   # far-left
            3: (COURT_MID_X + (COURT_X_MAX - COURT_MID_X)//2,
                COURT_Y_MIN + (COURT_MID_Y - COURT_Y_MIN)//2),   # far-right
        }
        # Last known position per zone
        self.last_positions = {0: None, 1: None, 2: None, 3: None}

    def _center(self, bbox):
        return ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)

    def _assign_zone(self, bbox):
        """Assign player to zone 0-3 based on court position."""
        cx, cy = self._center(bbox)

        # Determine quadrant
        left  = cx < COURT_MID_X
        near  = cy > COURT_MID_Y   # near = closer to camera = higher y

        if near and left:   return 0
        if near and not left: return 1
        if not near and left: return 2
        return 3

    def update(self, detections, frame_num):
        if not detections:
            return detections

        # Assign each detection to a zone
        zone_assignments = {}
        for d in detections:
            zone = self._assign_zone(d["bbox"])
            cx, cy = self._center(d["bbox"])

            # If zone already taken, pick next closest zone
            if zone in zone_assignments:
                # Find best unoccupied zone
                taken = set(zone_assignments.keys())
                best_zone = zone
                best_dist = float("inf")
                for z, zc in self.zone_centers.items():
                    if z not in taken:
                        dist = (cx-zc[0])**2 + (cy-zc[1])**2
                        if dist < best_dist:
                            best_dist = dist
                            best_zone = z
                zone = best_zone

            zone_assignments[zone] = d
            d["player_id"] = zone
            self.last_positions[zone] = (cx, cy)

        return detections

    def _iou(self, b1, b2):
        xi1 = max(b1[0], b2[0]); yi1 = max(b1[1], b2[1])
        xi2 = min(b1[2], b2[2]); yi2 = min(b1[3], b2[3])
        inter = max(0, xi2-xi1) * max(0, yi2-yi1)
        area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        return inter / (area1 + area2 - inter + 1e-6)