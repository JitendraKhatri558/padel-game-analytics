import pandas as pd
import numpy as np
import cv2
import json
from collections import Counter

def generate_dashboard(csv_path="shot_analysis.csv",
                       video_path="data/raw/input_sample_video.mp4",
                       output_path="dashboard.jpg"):

    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} shot events")

    # Get video frame for background
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
    ret, court_frame = cap.read()
    cap.release()

    # Canvas size
    W, H = 1600, 900
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 30  # dark background

    # ── Helper functions ──────────────────────────────────────────
    def draw_text(img, text, x, y, scale=0.7, color=(255,255,255),
                  thickness=1, bold=False):
        t = thickness + 1 if bold else thickness
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, t, cv2.LINE_AA)

    def draw_rect(img, x1, y1, x2, y2, color, filled=False, alpha=1.0):
        if filled:
            overlay = img.copy()
            cv2.rectangle(overlay, (x1,y1), (x2,y2), color, -1)
            cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 1)

    # ── Title bar ────────────────────────────────────────────────
    draw_rect(canvas, 0, 0, W, 60, (45, 45, 45), filled=True)
    draw_text(canvas, "PADEL GAME ANALYTICS", 20, 40,
              scale=1.1, color=(0, 220, 180), bold=True)
    draw_text(canvas, f"Total shots: {len(df)}   |   "
              f"Duration: {df['timestamp'].max():.1f}s   |   "
              f"Players: {df['player_id'].nunique()}",
              500, 40, scale=0.65, color=(180, 180, 180))

    # ── Section 1: Shot type bar chart (top left) ─────────────────
    shot_counts = df['shot_type'].value_counts()
    colors_map = {
        "forehand": (80,  200, 80),
        "backhand": (80,  140, 220),
        "serve":    (220, 180, 80),
        "smash":    (220, 80,  80),
        "volley":   (180, 80,  220),
        "unknown":  (120, 120, 120),
    }

    panel_x, panel_y = 20, 80
    draw_rect(canvas, panel_x, panel_y,
              panel_x+380, panel_y+340, (60,60,60), filled=True, alpha=0.5)
    draw_text(canvas, "Shot Distribution", panel_x+10, panel_y+28,
              scale=0.7, color=(200,200,200), bold=True)

    max_count = shot_counts.max()
    bar_x     = panel_x + 110
    bar_w_max = 220

    for i, (shot, count) in enumerate(shot_counts.items()):
        y       = panel_y + 60 + i * 45
        bar_w   = int(bar_w_max * count / max_count)
        color   = colors_map.get(shot, (150,150,150))
        pct     = count / len(df) * 100

        draw_text(canvas, shot.capitalize(), panel_x+10, y+14,
                  scale=0.6, color=(200,200,200))
        draw_rect(canvas, bar_x, y, bar_x+bar_w, y+28,
                  color, filled=True, alpha=0.8)
        draw_rect(canvas, bar_x, y, bar_x+bar_w_max, y+28, (80,80,80))
        draw_text(canvas, f"{count} ({pct:.0f}%)",
                  bar_x+bar_w+8, y+18, scale=0.55, color=(200,200,200))

    # ── Section 2: Player comparison (bottom left) ────────────────
    panel2_y = panel_y + 360
    draw_rect(canvas, panel_x, panel2_y,
              panel_x+380, panel2_y+200, (60,60,60), filled=True, alpha=0.5)
    draw_text(canvas, "Player Breakdown", panel_x+10, panel2_y+28,
              scale=0.7, color=(200,200,200), bold=True)

    players = df['player_id'].unique()
    p_colors = [(80,200,80), (80,140,220), (220,180,80), (220,80,80)]

    for i, pid in enumerate(sorted(players)):
        pdata  = df[df['player_id'] == pid]
        py     = panel2_y + 55 + i * 65
        pcolor = p_colors[i % len(p_colors)]

        draw_text(canvas, f"Player {pid}", panel_x+10, py+15,
                  scale=0.65, color=pcolor, bold=True)
        draw_text(canvas, f"{len(pdata)} shots", panel_x+10, py+38,
                  scale=0.55, color=(160,160,160))

        # Mini shot breakdown for this player
        x_offset = panel_x + 120
        for shot, cnt in pdata['shot_type'].value_counts().items():
            sc = colors_map.get(shot, (150,150,150))
            bw = int(60 * cnt / len(pdata))
            draw_rect(canvas, x_offset, py+5,
                      x_offset+bw, py+28, sc, filled=True, alpha=0.8)
            draw_text(canvas, shot[:3], x_offset+2, py+22,
                      scale=0.4, color=(20,20,20))
            x_offset += bw + 4

    # ── Section 3: Shot timeline (top right) ──────────────────────
    tl_x, tl_y = 420, 80
    tl_w, tl_h = 1160, 180
    draw_rect(canvas, tl_x, tl_y, tl_x+tl_w, tl_y+tl_h,
              (60,60,60), filled=True, alpha=0.5)
    draw_text(canvas, "Shot Timeline", tl_x+10, tl_y+28,
              scale=0.7, color=(200,200,200), bold=True)

    t_min = df['timestamp'].min()
    t_max = df['timestamp'].max()
    t_range = max(t_max - t_min, 1)

    # Time axis
    cv2.line(canvas,
             (tl_x+20, tl_y+tl_h-30),
             (tl_x+tl_w-20, tl_y+tl_h-30),
             (100,100,100), 1)

    for _, row in df.iterrows():
        t      = row['timestamp']
        shot   = row['shot_type']
        pid    = row['player_id']
        tx     = tl_x + 20 + int((t - t_min) / t_range * (tl_w-40))
        color  = colors_map.get(shot, (150,150,150))
        p_idx  = list(sorted(players)).index(pid)
        ty     = tl_y + 50 + p_idx * 40

        cv2.circle(canvas, (tx, ty), 10, color, -1)
        cv2.circle(canvas, (tx, ty), 10, (200,200,200), 1)
        draw_text(canvas, shot[:3], tx-10, ty+25,
                  scale=0.4, color=(180,180,180))
        cv2.line(canvas, (tx, ty+12), (tx, tl_y+tl_h-30),
                 (80,80,80), 1)
        draw_text(canvas, f"{t:.1f}s", tx-12, tl_y+tl_h-12,
                  scale=0.38, color=(140,140,140))

    # Player labels on timeline
    for i, pid in enumerate(sorted(players)):
        ty = tl_y + 50 + i * 40
        draw_text(canvas, f"P{pid}", tl_x+2, ty+5,
                  scale=0.5, color=p_colors[i % len(p_colors)])

    # ── Section 4: Court map (bottom right) ───────────────────────
    court_x, court_y = 420, 280
    court_w, court_h = 560, 580

    # Draw mini court
    draw_rect(canvas, court_x, court_y,
              court_x+court_w, court_y+court_h,
              (30,120,30), filled=True, alpha=0.6)
    draw_rect(canvas, court_x, court_y,
              court_x+court_w, court_y+court_h, (255,255,255))

    # Court lines
    mid_x = court_x + court_w // 2
    mid_y = court_y + court_h // 2
    cv2.line(canvas, (mid_x, court_y), (mid_x, court_y+court_h),
             (255,255,255), 1)
    cv2.line(canvas, (court_x, mid_y), (court_x+court_w, mid_y),
             (255,255,255), 2)  # net

    draw_text(canvas, "Shot Heatmap", court_x+10, court_y-10,
              scale=0.7, color=(200,200,200), bold=True)
    draw_text(canvas, "NET", mid_x-15, mid_y+15,
              scale=0.45, color=(200,200,200))

    # Video resolution for coordinate mapping
    vid_w, vid_h = 1920, 1080
    from src.detection import COURT_X_MIN, COURT_X_MAX, COURT_Y_MIN, COURT_Y_MAX

    # Plot ball positions on court
    for _, row in df.iterrows():
        if pd.isna(row['ball_x']) or pd.isna(row['ball_y']):
            continue
        bx = row['ball_x']
        by = row['ball_y']
        shot = row['shot_type']
        color = colors_map.get(shot, (150,150,150))

        # Map from video coords to court diagram coords
        nx = court_x + int((bx - COURT_X_MIN) /
                           (COURT_X_MAX - COURT_X_MIN) * court_w)
        ny = court_y + int((by - COURT_Y_MIN) /
                           (COURT_Y_MAX - COURT_Y_MIN) * court_h)

        nx = max(court_x+5, min(court_x+court_w-5, nx))
        ny = max(court_y+5, min(court_y+court_h-5, ny))

        cv2.circle(canvas, (nx, ny), 8, color, -1)
        cv2.circle(canvas, (nx, ny), 8, (255,255,255), 1)

    # ── Section 5: Stats panel (far right) ────────────────────────
    stats_x = 1000
    draw_rect(canvas, stats_x, 280, stats_x+580, 860,
              (60,60,60), filled=True, alpha=0.5)
    draw_text(canvas, "Match Statistics", stats_x+10, 310,
              scale=0.7, color=(200,200,200), bold=True)

    avg_conf = df['confidence'].mean()
    stats = [
        ("Total Shots",     str(len(df))),
        ("Duration",        f"{df['timestamp'].max():.1f}s"),
        ("Shots/min",       f"{len(df)/(df['timestamp'].max()/60):.1f}"),
        ("Avg Confidence",  f"{avg_conf:.2f}"),
        ("Most common",     df['shot_type'].mode()[0].capitalize()),
        ("Top player",      f"Player {df['player_id'].value_counts().index[0]}"),
        ("Ball tracked",    f"{df['ball_x'].notna().sum()}/{len(df)} shots"),
    ]

    for i, (label, value) in enumerate(stats):
        sy = 340 + i * 65
        draw_rect(canvas, stats_x+10, sy, stats_x+560, sy+50,
                  (50,50,50), filled=True, alpha=0.5)
        draw_text(canvas, label, stats_x+20, sy+18,
                  scale=0.55, color=(150,150,150))
        draw_text(canvas, value, stats_x+20, sy+40,
                  scale=0.7, color=(0,220,180), bold=True)

    # ── Legend ────────────────────────────────────────────────────
    legend_y = 830
    draw_text(canvas, "Legend:", 420, legend_y,
              scale=0.55, color=(150,150,150))
    lx = 490
    for shot, color in colors_map.items():
        if shot == "unknown":
            continue
        cv2.circle(canvas, (lx+8, legend_y-5), 7, color, -1)
        draw_text(canvas, shot.capitalize(), lx+18, legend_y,
                  scale=0.45, color=(180,180,180))
        lx += 110

    # Save
    cv2.imwrite(output_path, canvas)
    print(f"\nDashboard saved → {output_path}")
    print("Open dashboard.jpg in VS Code to view!")
    return canvas


if __name__ == "__main__":
    generate_dashboard()