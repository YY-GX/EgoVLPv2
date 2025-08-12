import os
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
from pathlib import Path
import glob
from collections import Counter

def parse_timestamp(timestamp):
    """Convert timestamp string to seconds."""
    try:
        dt = datetime.strptime(timestamp.strip(), '%M:%S')
        return dt.minute * 60 + dt.second
    except ValueError:
        # Try with hours if minutes:seconds format fails
        dt = datetime.strptime(timestamp.strip(), '%H:%M:%S')
        return dt.hour * 3600 + dt.minute * 60 + dt.second

def get_most_common_action(start_time, end_time, annotations):
    """Get the most common action label between start_time and end_time."""
    start_sec = parse_timestamp(start_time)
    end_sec = parse_timestamp(end_time)
    
    # Get all actions that overlap with this time period
    actions = []
    for i in range(len(annotations)):
        current_start = parse_timestamp(annotations.iloc[i]['timestamp'])
        
        # For the last annotation, assume it continues until the end
        if i == len(annotations) - 1:
            # If this action overlaps with our time period
            if current_start <= end_sec:
                # Calculate overlap duration
                overlap_start = max(current_start, start_sec)
                overlap_duration = end_sec - overlap_start
                
                # Add the action multiple times based on duration
                if overlap_duration > 0:
                    actions.extend([annotations.iloc[i]['action_label']] * int(overlap_duration))
        else:
            next_start = parse_timestamp(annotations.iloc[i + 1]['timestamp'])
            
            # If this action overlaps with our time period
            if (current_start <= end_sec and next_start >= start_sec):
                # Calculate overlap duration
                overlap_start = max(current_start, start_sec)
                overlap_end = min(next_start, end_sec)
                overlap_duration = overlap_end - overlap_start
                
                # Add the action multiple times based on duration
                if overlap_duration > 0:
                    actions.extend([annotations.iloc[i]['action_label']] * int(overlap_duration))
    
    if not actions:
        return None
        
    # Return the most common action
    return Counter(actions).most_common(1)[0][0]

def generate_clip_annotations_for_video(raw_annotation_file, clip_duration, video_id, video_dir):
    """
    Generate clip annotations for a single video based on existing clips.
    
    Args:
        raw_annotation_file (str): Path to the raw annotation CSV file
        clip_duration (int): Duration of each clip in seconds
        video_id (str): ID of the video (e.g., 'video_0')
        video_dir (str): Directory containing the video clips
    
    Returns:
        pd.DataFrame: DataFrame containing clip annotations
    """
    # Read raw annotations
    try:
        annotations = pd.read_csv(raw_annotation_file)
        annotations.columns = annotations.columns.str.strip()
        annotations['action_label'] = annotations['action_label'].str.strip()
        
        # Validate required columns
        required_columns = ['timestamp', 'action_label']
        missing_columns = [col for col in required_columns if col not in annotations.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        print(f"  Loaded {len(annotations)} annotations from {raw_annotation_file}")
        
    except Exception as e:
        raise ValueError(f"Error reading annotation file {raw_annotation_file}: {str(e)}")
    
    # Get list of existing clips - clips are directly in video_dir/video_id/clip_*.mp4
    clip_pattern = os.path.join(video_dir, video_id, "clip_*.mp4")
    existing_clips = glob.glob(clip_pattern)
    
    if not existing_clips:
        raise ValueError(f"No clips found matching pattern: {clip_pattern}")
    
    print(f"  Found {len(existing_clips)} clips in {video_dir}/{video_id}")
    
    # Create clips dataframe
    clips = []
    skipped_clips = 0
    
    for clip_path in existing_clips:
        # Extract clip number from filename
        clip_num = int(os.path.basename(clip_path).replace('clip_', '').replace('.mp4', ''))
        start_time = clip_num * clip_duration
        
        # Convert start_time to MM:SS format
        minutes = int(start_time // 60)
        seconds = int(start_time % 60)
        timestamp = f"{minutes:02d}:{seconds:02d}"
        
        # Get action label for this timestamp
        action_label = get_most_common_action(timestamp, f"{int((start_time + clip_duration) // 60):02d}:{int((start_time + clip_duration) % 60):02d}", annotations)
        
        if action_label is None:
            print(f"  Warning: No action label found for clip {clip_path}")
            skipped_clips += 1
            continue
            
        clips.append({
            'video_id': f"{video_id}_clip_{clip_num:03d}",
            'action_label': action_label,
            'start_time': timestamp,
            'source_video': video_id
        })
    
    if skipped_clips > 0:
        print(f"  Skipped {skipped_clips} clips due to missing action labels")
    
    if not clips:
        raise ValueError(f"No valid clips generated for {video_id}")
    
    return pd.DataFrame(clips)

def generate_combined_annotations(annotation_dir, clip_duration, video_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, output_dir=None, split_mode='mode1'):
    """
    Generate train/val/test annotation files from multiple raw timestamp annotations.
    Supports two split modes:
      - mode1: random clip-level split (default)
      - mode2: video_1+ for train, video_0 split in half for val/test
    """
    # Find all raw annotation files (video_*.csv)
    raw_annotation_files = glob.glob(os.path.join(annotation_dir, 'video_*.csv'))
    if not raw_annotation_files:
        raise ValueError(f"No raw annotation files found in {annotation_dir}. Expected video_*.csv files")
    raw_annotation_files.sort()
    print(f"Found {len(raw_annotation_files)} annotation files: {[os.path.basename(f) for f in raw_annotation_files]}")
    # Generate clips for each video
    all_clips = []
    video_id_to_df = {}
    for raw_file in raw_annotation_files:
        video_id = os.path.splitext(os.path.basename(raw_file))[0]  # Get video_0, video_1, etc.
        print(f"Processing {video_id}...")
        try:
            clips_df = generate_clip_annotations_for_video(raw_file, clip_duration, video_id, video_dir)
            all_clips.append(clips_df)
            video_id_to_df[video_id] = clips_df
            print(f"  Generated {len(clips_df)} clips for {video_id}")
        except Exception as e:
            print(f"  Error processing {video_id}: {str(e)}")
            continue
    if not all_clips:
        raise ValueError("No clips were generated from any video")
    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = annotation_dir
    os.makedirs(output_dir, exist_ok=True)
    if split_mode == 'mode1':
        # --- Original random split ---
        combined_clips = pd.concat(all_clips, ignore_index=True)
        combined_clips = combined_clips.sample(frac=1, random_state=42)
        n_clips = len(combined_clips)
        n_train = int(n_clips * train_ratio)
        n_val = int(n_clips * val_ratio)
        train_df = combined_clips.iloc[:n_train].copy()
        val_df = combined_clips.iloc[n_train:n_train + n_val].copy()
        test_df = combined_clips.iloc[n_train + n_val:].copy()
        train_df['split'] = 'train'
        val_df['split'] = 'val'
        test_df['split'] = 'test'
        mode_dir = os.path.join(output_dir, 'mode1')
        os.makedirs(mode_dir, exist_ok=True)
        train_df.to_csv(os.path.join(mode_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(mode_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(mode_dir, 'test.csv'), index=False)
        print(f"\n[MODE1] Saved annotation files in {mode_dir}")
        # Print statistics
        print(f"\nGenerated annotation files in {mode_dir}:")
        print(f"Total clips: {n_clips}")
        print(f"Train: {len(train_df)} clips")
        print(f"Val: {len(val_df)} clips")
        print(f"Test: {len(test_df)} clips")
        # Print statistics per video
        print("\nClips per video:")
        for video_id in combined_clips['source_video'].unique():
            video_clips = combined_clips[combined_clips['source_video'] == video_id]
            print(f"\n{video_id}:")
            print(f"  Total: {len(video_clips)} clips")
            print(f"  Train: {len(train_df[train_df['source_video'] == video_id])} clips")
            print(f"  Val: {len(val_df[val_df['source_video'] == video_id])} clips")
            print(f"  Test: {len(test_df[test_df['source_video'] == video_id])} clips")
        # Print action distribution
        print("\nAction distribution:")
        action_counts = combined_clips['action_label'].value_counts()
        for action, count in action_counts.items():
            print(f"  {action}: {count} clips")
    elif split_mode == 'mode2':
        # --- New video-based split ---
        train_clips = []
        val_clips = []
        test_clips = []
        for vid, df in video_id_to_df.items():
            if vid == 'video_0':
                n = len(df)
                half = n // 2
                val_df = df.iloc[:half].copy()
                test_df = df.iloc[half:].copy()
                val_df['split'] = 'val'
                test_df['split'] = 'test'
                val_clips.append(val_df)
                test_clips.append(test_df)
            else:
                train_df = df.copy()
                train_df['split'] = 'train'
                train_clips.append(train_df)
        train_df = pd.concat(train_clips, ignore_index=True) if train_clips else pd.DataFrame()
        val_df = pd.concat(val_clips, ignore_index=True) if val_clips else pd.DataFrame()
        test_df = pd.concat(test_clips, ignore_index=True) if test_clips else pd.DataFrame()
        mode_dir = os.path.join(output_dir, 'mode2')
        os.makedirs(mode_dir, exist_ok=True)
        train_df.to_csv(os.path.join(mode_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(mode_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(mode_dir, 'test.csv'), index=False)
        print(f"\n[MODE2] Saved annotation files in {mode_dir}")
        # Print statistics
        print(f"\nGenerated annotation files in {mode_dir}:")
        print(f"Total clips: {len(train_df) + len(val_df) + len(test_df)}")
        print(f"Train: {len(train_df)} clips")
        print(f"Val: {len(val_df)} clips")
        print(f"Test: {len(test_df)} clips")
        # Print statistics per video
        print("\nClips per video:")
        all_splits = pd.concat([train_df, val_df, test_df], ignore_index=True)
        if isinstance(all_splits, pd.DataFrame) and not all_splits.empty and 'source_video' in all_splits.columns:
            for video_id in all_splits['source_video'].unique():
                video_clips = all_splits[all_splits['source_video'] == video_id]
                print(f"\n{video_id}:")
                print(f"  Total: {len(video_clips)} clips")
                print(f"  Train: {len(video_clips[video_clips['split'] == 'train'])} clips")
                print(f"  Val: {len(video_clips[video_clips['split'] == 'val'])} clips")
                print(f"  Test: {len(video_clips[video_clips['split'] == 'test'])} clips")
            # Print action distribution
            print("\nAction distribution:")
            action_counts = all_splits['action_label'].value_counts()
            for action, count in action_counts.items():
                print(f"  {action}: {count} clips")
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

def main():
    parser = argparse.ArgumentParser(description='Generate train/val/test annotation files from raw timestamps')
    parser.add_argument('--annotation_dir', type=str, required=True, 
                      help='Directory containing raw annotation CSV files (video_*.csv)')
    parser.add_argument('--clip_duration', type=int, required=True, 
                      help='Duration of each clip in seconds')
    parser.add_argument('--video_dir', type=str, required=True,
                      help='Directory containing the video clips')
    parser.add_argument('--train_ratio', type=float, default=0.7, 
                      help='Ratio of clips for training')
    parser.add_argument('--val_ratio', type=float, default=0.15, 
                      help='Ratio of clips for validation')
    parser.add_argument('--test_ratio', type=float, default=0.15, 
                      help='Ratio of clips for testing')
    parser.add_argument('--output_dir', type=str, 
                      help='Directory to save the generated annotation files')
    parser.add_argument('--split_mode', type=str, default='mode1', choices=['mode1', 'mode2'],
                      help='Split mode: mode1 (random clip split), mode2 (video_0 for val/test, others for train)')
    args = parser.parse_args()
    generate_combined_annotations(
        args.annotation_dir,
        args.clip_duration,
        args.video_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.output_dir,
        args.split_mode
    )

if __name__ == '__main__':
    main() 