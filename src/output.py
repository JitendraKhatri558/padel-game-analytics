import json
import csv
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class ShotEvent:
    frame:      int
    timestamp:  float
    player_id:  Optional[int]
    shot_type:  str
    ball_x:     Optional[float]
    ball_y:     Optional[float]
    confidence: float

class OutputWriter:
    def __init__(self, fps: float):
        self.fps    = fps
        self.events = []

    def add_event(self, frame, player_id, shot_type,
                  ball_pos=None, confidence=1.0):
        event = ShotEvent(
            frame      = frame,
            timestamp  = round(frame / self.fps, 3),
            player_id  = player_id,
            shot_type  = shot_type,
            ball_x     = ball_pos["x"] if ball_pos else None,
            ball_y     = ball_pos["y"] if ball_pos else None,
            confidence = round(confidence, 3),
        )
        self.events.append(event)
        return event

    def to_json(self, path="shot_analysis.json"):
        with open(path, "w") as f:
            json.dump([asdict(e) for e in self.events], f, indent=2)
        print(f"Saved {len(self.events)} events → {path}")

    def to_csv(self, path="shot_analysis.csv"):
        if not self.events:
            print("No events to save.")
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=asdict(self.events[0]).keys()
            )
            writer.writeheader()
            writer.writerows([asdict(e) for e in self.events])
        print(f"Saved {len(self.events)} events → {path}")

    def print_summary(self):
        if not self.events:
            print("No shot events detected.")
            return

        print(f"\n{'='*40}")
        print(f"SHOT ANALYSIS SUMMARY")
        print(f"{'='*40}")
        print(f"Total shots detected: {len(self.events)}")

        # Count by shot type
        from collections import Counter
        shot_counts = Counter(e.shot_type for e in self.events)
        print(f"\nShot breakdown:")
        for shot, count in shot_counts.most_common():
            pct = count / len(self.events) * 100
            print(f"  {shot:<12}: {count:>3} ({pct:.1f}%)")

        # Count by player
        player_counts = Counter(e.player_id for e in self.events)
        print(f"\nShots by player:")
        for pid, count in player_counts.most_common():
            print(f"  Player {pid}: {count} shots")

        print(f"{'='*40}")