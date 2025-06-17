import os
import cv2

INPUT_VIDEO = './clips/yy_1.vrs.mp4'
OUTPUT_DIR = './sliding_clips'
WINDOW_SEC = 2
STRIDE_SEC = 1

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

window_size = int(WINDOW_SEC * fps)
stride = int(STRIDE_SEC * fps)

idx = 0
for start in range(0, total_frames - window_size + 1, stride):
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(window_size):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    if len(frames) == window_size:
        out_path = os.path.join(OUTPUT_DIR, f"clip_{idx:04d}.mp4")
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        for f in frames:
            out.write(f)
        out.release()
        print(f"[✓] Saved: {out_path}")
        idx += 1

cap.release()
print(f"Generated {idx} sliding clips.")
