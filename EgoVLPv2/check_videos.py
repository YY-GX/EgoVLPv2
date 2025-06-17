import pandas as pd
import os
from pathlib import Path

# Read the validation CSV file
val_csv = "annotations/val.csv"
df = pd.read_csv(val_csv)

# Base directory for videos
video_dir = "data/video_0"

# Check each video file
missing_videos = []
for idx, row in df.iterrows():
    # Convert video_id to string and ensure it's 3 digits
    video_id = str(row['video_id']).zfill(3)
    video_path = os.path.join(video_dir, f"clip_{video_id}.mp4")
    if not os.path.exists(video_path):
        missing_videos.append({
            'video_id': row['video_id'],
            'action_label': row['action_label'],
            'start_time': row['start_time'],
            'end_time': row['end_time']
        })
        print(f"Missing video: {video_path}")

print(f"\nTotal videos in val.csv: {len(df)}")
print(f"Missing videos: {len(missing_videos)}")
if missing_videos:
    print("\nMissing video details:")
    for video in missing_videos:
        print(f"Video ID: {video['video_id']}, Action: {video['action_label']}, Start: {video['start_time']}, End: {video['end_time']}") 