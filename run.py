import cv2
from collections import deque
from src.detection      import PadelDetector
from src.tracking       import BallTracker, PlayerTracker
from src.classification import ShotClassifier
from src.output         import OutputWriter

def analyze_video(video_path: str, output_prefix: str = "shot_analysis",
                  max_frames: int = None):

    cap = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_frames:
        total_frames = min(total_frames, max_frames)

    print(f"Video : {video_path}")
    print(f"Frames: {total_frames} @ {fps:.1f} fps")
    print(f"Length: {total_frames/fps:.1f} seconds\n")

    detector       = PadelDetector()
    ball_tracker   = BallTracker(history_len=30)
    player_tracker = PlayerTracker()
    classifier     = ShotClassifier()
    writer         = OutputWriter(fps=fps)

    # Keep recent keyframes per player for sequence classification
    player_pose_buffer = {}   # player_id -> deque of keypoints

    frame_num    = 0
    hit_cooldown = 0   # prevent double-counting same hit

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_num >= total_frames:
            break

        # --- Detect ---
        dets    = detector.detect_frame(frame)
        players = player_tracker.update(dets["players"], frame_num)

        # --- Track ball ---
        ball_tracker.update(dets["ball"], frame_num)

        # --- Buffer player poses ---
        for player in players:
            pid = player["player_id"]
            if pid not in player_pose_buffer:
                player_pose_buffer[pid] = deque(maxlen=8)
            player_pose_buffer[pid].append(player["keypoints"])

        # --- Detect hit and classify ---
        if hit_cooldown == 0 and ball_tracker.detect_hit_event():
            # Find closest player to ball
            ball = dets["ball"]
            best_player = None

            if ball and players:
                min_dist = float("inf")
                for p in players:
                    px = (p["bbox"][0] + p["bbox"][2]) / 2
                    py = (p["bbox"][1] + p["bbox"][3]) / 2
                    dist = ((px - ball["x"])**2 + (py - ball["y"])**2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        best_player = p
            elif players:
                best_player = players[0]

            if best_player:
                pid        = best_player["player_id"]
                pose_seq   = list(player_pose_buffer.get(pid, []))
                shot_type  = classifier.classify_sequence(pose_seq)
                confidence = best_player["confidence"]

                event = writer.add_event(
                    frame      = frame_num,
                    player_id  = pid,
                    shot_type  = shot_type,
                    ball_pos   = dets["ball"],
                    confidence = confidence,
                )
                print(f"  Frame {frame_num:>5} | {shot_type:<12} | "
                      f"Player {pid} | conf={confidence:.2f}")

                hit_cooldown = 15   # skip next 15 frames after a hit

        if hit_cooldown > 0:
            hit_cooldown -= 1

        frame_num += 1
        if frame_num % 500 == 0:
            print(f"[{frame_num}/{total_frames}] "
                  f"shots so far: {len(writer.events)}")

    cap.release()

    writer.print_summary()
    writer.to_json(f"{output_prefix}.json")
    writer.to_csv(f"{output_prefix}.csv")

    return writer.events


if __name__ == "__main__":
    import sys
    video      = sys.argv[1] if len(sys.argv) > 1 \
                 else "data/raw/input_sample_video.mp4"
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 500

    events = analyze_video(video, max_frames=max_frames)
    print(f"\nDone. {len(events)} shot events saved.")