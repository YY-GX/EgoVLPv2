import os
import cv2
import math
from glob import glob

# Parameters
INPUT_DIR = '/mnt/arc/yygx/pkgs_baselines/EgoVLPv2/EgoVLPv2/data/raw_videos'
OUTPUT_BASE_DIR = '/mnt/arc/yygx/pkgs_baselines/EgoVLPv2/EgoVLPv2/data'
CLIP_DURATION = 2  # seconds

# Get all mp4 files in alphabetical order
video_files = sorted(glob(os.path.join(INPUT_DIR, '*.mp4')))

for idx, video_path in enumerate(video_files):
    video_name = os.path.basename(video_path)
    print(f'Processing {video_name}...')
    output_dir = os.path.join(OUTPUT_BASE_DIR, f'video_{idx}')
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    clip_frames = int(CLIP_DURATION * fps)
    n_clips = math.ceil(total_frames / clip_frames)

    for clip_idx in range(n_clips):
        start_frame = clip_idx * clip_frames
        end_frame = min((clip_idx + 1) * clip_frames, total_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = os.path.join(output_dir, f'clip_{clip_idx:03d}.mp4')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        for f in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        out.release()
    cap.release()
    print(f'Saved {n_clips} clips to {output_dir}')

print('Done!') 